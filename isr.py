from sympy.physics.secondquant import (
    F, Fd, wicks, NO, AntiSymmetricTensor, Dagger
)
from sympy import symbols, latex, nsimplify, Rational, diff, S

import numpy as np
from math import factorial

from indices import (
    n_ov_from_space, repeated_indices, index_space, split_idx_string, indices
)
from misc import (cached_member, Inputerror, transform_to_tuple,
                  validate_input, process_arguments)
from simplify import simplify
from func import evaluate_deltas


class intermediate_states:
    def __init__(self, mp, variant="pp"):
        self.gs = mp
        self.indices = indices()

        variants = {
            "pp": ["ph", "hp"],
            "ea": ["p"],
            "ip": ["h"],
            "dip": ["hh"],
            "dea": ["pp"],
        }
        if variant not in variants.keys():
            raise Inputerror(f"The ADC variant {variant} is not valid. "
                             "Supported variants are "
                             f"{list(variants.keys())}.")
        self.variant = variant
        self.min_space = variants[variant]

    @process_arguments
    @cached_member
    def precursor(self, order, space, braket, indices):
        """Method used to obtain precursor states of specified order
           for the specified space. The indices of the resulting precursor
           wavefunction need to be provided as string in the input
           (e.g. indices='ia' produces |PSI_{ia}^#>).
           """

        # check input parameters
        space = transform_to_tuple(space)
        indices = transform_to_tuple(indices)
        validate_input(order=order, space=space, braket=braket,
                       indices=indices)
        if len(indices) != 1:
            raise Inputerror(f"{indices} are not valid for constructing a "
                             "precursor state.")
        space = space[0]
        indices = indices[0]
        # check that the space is valid for the given ADC variant
        if not self.check_valid_space(space):
            raise Inputerror(f"{space} is not a valid space for "
                             f"{self.variant} ADC.")

        # check compatibility of indices and space
        idx_ov = {'occ': 0, 'virt': 0, 'general': 0}
        for idx in split_idx_string(indices):
            idx_ov[index_space(idx)] += 1
        if idx_ov['general']:
            raise Inputerror(f"The provided indices {indices} include a "
                             "general index.")
        n_ov_space = n_ov_from_space(space)
        if idx_ov["occ"] != n_ov_space["n_occ"] or \
                idx_ov["virt"] != n_ov_space["n_virt"]:
            raise Inputerror(f"The provided indices {indices} (occ: "
                             f"{idx_ov['occ']} / virt: {idx_ov['virt']}) do "
                             "not match the required amount of indices for the"
                             f" space {space} (occ:{n_ov_space['n_occ']} / "
                             f"virt: {n_ov_space['n_virt']}).")

        # get the target symbols of the precursor state
        idx = self.indices.get_indices(indices)
        # in contrast to the gs, here the operators are ordered as
        # abij instead of abji in order to stay consistent with the
        # ADC results.
        operators = 1
        if idx.get('virt', False):
            for symbol in idx['virt']:
                operators *= Fd(symbol)
        if idx.get('occ', False):
            for symbol in idx['occ']:
                operators *= F(symbol)
        if braket == "bra":
            operators = Dagger(operators)

        # leading term:
        # no need to differentiate bra/ket here, because
        # operators * mp = mp * operators (there is always an equal number of
        # p/h operators in mp that needs to be moved to the other side.
        # Will always give +.)
        max_gs = self.gs.psi(order, braket)
        res = (NO(operators) * max_gs).expand()

        # get all terms of a*b of the desired order (ground state norm)
        orders = get_order_two(order)

        # orthogonalise with respect to the ground state for pp ADC.
        # checked up to 4th order!
        if self.variant == "pp":
            # import all ground state wave functions that may not appear twice
            # in |a><b|c>, i.e. all of order > int(order/2)
            gs_psi = {'bra': {}, 'ket': {}}
            gs_psi[braket][order] = max_gs
            for o in range(order//2 + 1, order+1):
                if not gs_psi['bra'].get(o):
                    gs_psi['bra'][o] = self.gs.psi(o, 'bra')
                if not gs_psi['ket'].get(o):
                    gs_psi['ket'][o] = self.gs.psi(o, 'ket')
            def get_gs_wfn(o, bk): return gs_psi[bk][o] if o > order//2 else \
                self.gs.psi(o, bk)
            # 1) iterate through all combinations of norm_factor*projector
            for norm_term in orders:
                norm = self.gs.norm_factor(norm_term[0])
                if norm is S.Zero:
                    continue
                # 2) construct the projector for a given norm_factor
                # the overall order is split between the norm_factor and the
                # projector
                orders_pre = get_orders_three(norm_term[1])
                projection = 0
                for term in orders_pre:
                    # |Y>  <--  -|X><X|Y>
                    if braket == "ket":
                        i1 = (get_gs_wfn(term[1], 'bra') * NO(operators) *
                              get_gs_wfn(term[2], 'ket'))
                        state = get_gs_wfn(term[0], 'ket')
                    # <Y|  <--  -<Y|X><X|
                    elif braket == "bra":
                        i1 = (get_gs_wfn(term[0], 'bra') * NO(operators) *
                              get_gs_wfn(term[1], 'ket'))
                        state = get_gs_wfn(term[2], 'bra')
                    # wicks automatically expands the passed expression
                    i1 = wicks(
                        i1, keep_only_fully_contracted=True,
                        simplify_kronecker_deltas=True,
                    )
                    # print(f"term {term}:", latex(i1))
                    projection += (state * i1).expand()
                projection = evaluate_deltas(projection).sympy
                # print(f"NORM FACTOR {norm_term[0]}:", latex(norm))
                # print(f"PROJECTION {norm_term[1]}:", latex(projection))
                res -= (norm * projection).expand()
            gs_psi.clear()

        # iterate over lower excitated spaces
        lower_spaces = self.__generate_lower_spaces(space)
        for lower_space in lower_spaces:
            # get generic unique indices to generate the lower_isr_states.
            n_ov = n_ov_from_space(lower_space)
            indices_isr = self.indices.get_generic_indices(**n_ov)
            idx_isr = [s.name for symb in indices_isr.values() for s in symb]
            idx_isr = "".join(idx_isr)

            # prefactor due to the sum - sum_J |J><J|I>
            prefactor = Rational(
                1, factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
            )

            # orthogonalise with respsect to the lower excited ISR state
            # 1) iterate through all combinations of norm_factor*projector
            for norm_term in orders:
                norm = self.gs.norm_factor(norm_term[0])
                if norm is S.Zero:
                    continue
                # 2) construct the projector for a given norm factor
                # the overall order is split between he norm_factor and the
                # projector
                orders_isr = get_orders_three(norm_term[1])
                projection = 0
                for term in orders_isr:
                    # |Y#>  <--  -|X><X|Y>
                    if braket == "ket":
                        i1 = (self.intermediate_state(
                                  term[1], lower_space, "bra", indices=idx_isr
                                  )
                              * NO(operators) * self.gs.psi(term[2], "ket")
                              )
                        state = self.intermediate_state(
                            term[0], lower_space, "ket", indices=idx_isr
                        )
                    # <Y#|  <--  -<Y|X><X|
                    elif braket == "bra":
                        i1 = (self.gs.psi(term[0], "bra") * NO(operators) *
                              self.intermediate_state(
                                  term[1], lower_space, "ket", indices=idx_isr
                                  )
                              )
                        state = self.intermediate_state(
                            term[2], lower_space, "bra", indices=idx_isr
                        )
                    i1 = wicks(
                        i1, keep_only_fully_contracted=True,
                        simplify_kronecker_deltas=True,
                    )
                    projection += (prefactor * state * i1).expand()
                projection = evaluate_deltas(projection).sympy
                res -= (norm * projection).expand()

        print(f"Build precursor {space}_({indices})^({order}) {braket}:"
              f" {latex(res)}")
        return res

    @process_arguments
    @cached_member
    def overlap_precursor(self, order, space, indices):
        """Method to obtain precursor overlap matrices
           for a given order and space. Indices of the resulting
           overlap matrix element need to be provided in the form 'ia,jb'
           which will produce S_{ia,jb}.
           """

        # no need to do more validation here -> will be done in precursor
        space = transform_to_tuple(space)
        indices = transform_to_tuple(indices)
        if len(indices) != 2 or len(space) != 2:
            raise Inputerror("Necessary to provide 2 index and space strings "
                             "for contructing a precursor overlap matrix."
                             f"Provided: {indices}")

        if repeated_indices(indices[0], indices[1]):
            raise Inputerror("Repeated index found in indices of precursor "
                             f"overlap matrix: {indices}.")

        orders = get_order_two(order)

        res = 0
        # 1) iterate through all combinations of norm_factor*S
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct S for a given norm factor
            # the overall order is split between he norm_factor and S
            orders_overlap = get_order_two(norm_term[1])
            overlap = 0
            for term in orders_overlap:
                i1 = (self.precursor(term[0], space=space[0], braket="bra",
                      indices=indices[0]) *
                      self.precursor(term[1], space=space[1], braket="ket",
                      indices=indices[1]))
                i1 = wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
                overlap += i1
            res += (norm * overlap).expand()
        # It should be valid to simplifiy the result by permuting contracted
        # indices before returning -> should lower the overall size of the
        # final expression
        res = simplify(res)
        print(f"Build overlap {space} S_{indices}^({order}) = {res}")
        return res.sympy

    @process_arguments
    @cached_member
    def s_root(self, order, space, indices):
        """Method to obtain S^{-0.5} of a given order.
           Indices for the resulting matrix element are required.
           e.g. indices='ia,jb' produces (S^{-0.5})_{ia,jb}
           """

        space = transform_to_tuple(space)
        indices = transform_to_tuple(indices)
        if len(indices) != 2 or len(space) != 2:
            raise Inputerror("Necessary to provide 2 index and space strings "
                             "for contructing a precursor overlap matrix."
                             f"Provided: {indices}")
        if sorted(space[0]) != sorted(space[1]):
            raise NotImplementedError("Did only implement combined terms "
                                      "(S*S) for space I,I'.")

        # TODO: check if min_order is still correct with first order singles
        # (I think it should be)
        prefactors, orders = self.expand_S_taylor(order, min_order=2)
        # assume in the following that both spaces are equal!!
        n_ov = n_ov_from_space(space[0])
        # create an index list: first and last element are the two provided
        # idx strings
        idx = list(indices)
        res = 0
        # iterate over exponents and terms, starting with the lowest exponent
        for exponent, termlist in dict(sorted(orders.items())).items():
            # generate len(term)-1 or exponent-1 index spaces, e.g. for x*x
            # 1 additional space is required: s,s' = s,s''*s'',s'
            # x^3: s,s' = s,s'' * s'',s''' * s''',s' etc.
            if len(idx) - 1 < exponent:
                for i in range(exponent - len(idx) + 1):
                    # insert them in the idx list [0,..., new, last]
                    new = self.indices.get_generic_indices(**n_ov)
                    idx.insert(-1, "".join([s.name for sym in new.values()
                               for s in sym]))
            for term in termlist:
                i1 = prefactors[exponent]
                for n, o in enumerate(term):
                    i1 *= self.overlap_precursor(o, space,
                                                 indices=(idx[n], idx[n+1]))
                # in combined terms S*S delta evaluation might be necessary
                res += evaluate_deltas(i1.expand()).sympy
        print(f"Build {space} S_root_{indices}^({order}) = {latex(res)}")
        return res

    @process_arguments
    @cached_member
    def intermediate_state(self, order, space, braket, indices):
        """Returns an intermediate state. The index string idx_is represents
           the indices of the returned Intermediate State. The index string
           idx_pre is used to generate S^(-0.5), i.e. the sum
           sum_I' |I'> S_{I',I}^(-0.5) // sum_I' S_{I,I'}^(-0.5) <I'|
           runs over idx_pre.
           """

        indices = transform_to_tuple(indices)
        space = transform_to_tuple(space)
        if len(indices) != 1 or len(space) != 1:
            raise Inputerror(f"{space} or {indices} are not valid to "
                             "construct an intermediate state.")
        indices = indices[0]
        space = space[0]

        # generate additional indices for the precursor state
        n_ov = n_ov_from_space(space)
        indices_pre = self.indices.get_generic_indices(**n_ov)
        idx_pre = [s.name for sym in indices_pre.values() for s in sym]
        idx_pre = "".join(idx_pre)

        prefactor = Rational(
            1, factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        # sandwich the IS and precursor indices together
        s_indices = {
            'bra': ",".join([indices, idx_pre]),
            'ket': ",".join([idx_pre, indices])
        }

        orders = get_order_two(order)
        res = 0
        for term in orders:
            i1 = (prefactor *
                  self.s_root(term[0], space=(space, space),
                              indices=s_indices[braket]) *
                  self.precursor(term[1], space=space, braket=braket,
                                 indices=idx_pre))
            res += evaluate_deltas(i1.expand()).sympy
        print(f"Build {space} ISR_({indices}^({order}) "
              f"{braket} = {latex(res)}")
        return res

    @process_arguments
    @cached_member
    def overlap_isr(self, order, block, indices):
        """Computes a block of the overlap matrix in the ISR basis."""

        space = transform_to_tuple(space)
        indices = transform_to_tuple(indices)
        if len(space) != 2 or len(indices) != 2:
            raise Inputerror("Constructing a ISR overlap matrix requires two "
                             f"index and block strings. Provided: {space} / "
                             f"{indices}.")

        orders = get_order_two(order)
        res = 0
        # 1) iterate through all combinations of norm_factor*S
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct S for a given norm factor
            # the overall order is split between he norm_factor and S
            orders_overlap = get_order_two(norm_term[1])
            overlap = 0
            for term in orders_overlap:
                i1 = (self.intermediate_state(term[0], space[0], "bra",
                                              indices=indices[0]) *
                      self.intermediate_state(term[1], space[1], "ket",
                                              indices=indices[1]))
                i1 = wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
                overlap += i1
            res += (norm * overlap).expand()
        print(f"Build ISR overlap {space} S_{indices}^({order}) = ",
              latex(res))
        return res

    @process_arguments
    @cached_member
    def amplitude_vector(self, indices, lr="right"):
        """Returns an amplitude vector using the provided indices.
           They are sorted alphabetically before use.
           """

        idx = self.indices.get_indices(indices)
        # add empty list if e.g. only occ indices have been provided (IP)
        for ov in ["occ", "virt"]:
            if ov not in idx:
                idx[ov] = []

        t_string = {
            "right": "Y",
            "left": "X",
        }
        return AntiSymmetricTensor(
            t_string[lr], tuple(idx["virt"]), tuple(idx["occ"])
        )

    def expand_S_taylor(self, order, min_order=2):
        """Computes the Taylor expansion of 'S^{-0.5} = (1 + x)^{-0.5} with
           'x = S(2) + S(3) + O(4)' to a given order in perturbation theory.
           The lowest order term of x may be defined via the parameter
           min_order.

           Returns two dicts:
           The first one contains the prefactors of the series x + x² + ...
           The second one contains the orders of S that contribute to the
           n-th order term. (like (4,) and (2,2) for fourth order)
           In both dicts the exponent of x in the Taylor expansion is used
           as key.
           """

        x = symbols('x')
        diffs = {0: 1}
        f = (1 + x) ** -0.5
        intermediate = f
        for o in range(1, int(order/min_order) + 1):
            intermediate = diff(intermediate, x)
            diffs[o] = nsimplify(intermediate.subs(x, 0) * 1 / factorial(o))
        orders = gen_order_S(order, min_order=min_order)
        if not orders:
            orders[0] = [(order,)]
        return (diffs, orders)

    def __generate_lower_spaces(self, space_str):
        lower_spaces = []
        n_ov = n_ov_from_space(space_str)
        for i in range(min(n_ov.values())):
            space_str = space_str.replace('p', "", 1)
            space_str = space_str.replace('h', "", 1)
            if not space_str:
                break
            lower_spaces.append(space_str)
        return lower_spaces

    def check_valid_space(self, space_str):
        """Checks wheter the provided space is a valid space for
           the chosen ADC variant.
           """

        if space_str in self.min_space:
            return True

        lower_spaces = self.__generate_lower_spaces(space_str)
        valid = False
        for s in lower_spaces:
            if s in self.min_space:
                valid = True
        return valid

    def pretty_precursor(self, order, space, braket):
        """Returns precursor bra/ket for a given space and order.
           Makes the result pretty by substituting indices. Therefore, no
           further calculations are possible with the resulting expression!
           """

        indices = {
            "ph": "ia",
            "pphh": "ijab",
            "ppphhh": "ijkabc",
            "pppphhhh": "ijklabcd"
        }
        if space not in indices:
            raise Inputerror("Can only build a pretty precursor state for the "
                             f"spaces {list(indices.keys())}.")
        return self.indices.substitute(
            self.precursor(order, space, braket, indices[space])
        )

    def pretty_precursor_overlap(self, order, space):
        """Returns the precursor overlap matrix of a given order
           for the defined space. Makes the result pretty by substituting
           indices. Therefore, no further calculations are possible with the
           resulting expression!
           """

        indices = {
            'ph': "ia,jb",
            'pphh': "iajb,kcld",
            'ppphhh': "iajbkc,ldmenf",
        }
        if space not in indices:
            raise Inputerror("Can only build a pretty overlap matrix for the "
                             f"spaces {list(indices.keys())}.")
        return self.indices.substitute(
            self.overlap_precursor(order, space, indices[space])
        )

    def pretty_s_root(self, order, space):
        """Returns S^{-0.5} of a given order for the defined space.
           Makes the result pretty by substituting indices. Therefore,
           no further calculations are possible with the resulting expression!
           """

        indices = {
            'ph': "ia,jb",
            'pphh': "iajb,kcld",
            'ppphhh': "iajbkc,ldmenf",
        }
        if space not in indices:
            raise Inputerror("Can only build pretty S_root for the spaces",
                             f"{list(indices.keys())}.")
        return self.indices.substitute(
            self.s_root(order, space, indices[space])
        )

    def pretty_intermediate_state(self, order, space, braket):
        """Returns a bra/ket intermediate state for a given space and order.
           Makes the result pretty by substituting indices. Therefore,
           no further calculations are possible with the resulting expression!
           """

        indices = {
            'ph': "ia",
            'pphh': "iajb",
            'ppphhh': "iajbkc",
            "pppphhhh": "ijklabcd"
        }
        if space not in indices:
            raise Inputerror("Can only build pretty intermediate states for "
                             f"spaces {list(indices.keys())}.")
        return self.indices.substitute(
            self.intermediate_state(order, space, braket, indices[space])
        )


def gen_order_S(order, min_order=2):
    """Computes the series x + x² + ... with
       x = S(min) + S(min+1) + ... + S(order).
       Returns all terms of the series that are of a given order
       sorted by the exponent of x:
       {exponent: [(o1, o2, ...), ...]}.
       """

    orders = np.array([2 ** o for o in range(min_order, order + 1)])
    multiplied = {1: orders}
    if order >= 2 * min_order:
        res = orders
        for exponent in range(int(order/min_order) - 1):
            i1 = np.multiply.outer(res, orders)
            multiplied[exponent + 2] = i1
            res = i1

    indices = {}
    for exponent, product in multiplied.items():
        idx = np.where(product == 2 ** order)
        if idx[0].size > 0:
            indices[exponent] = idx

    matching_orders = {}
    for exponent, artuple in indices.items():
        res = []
        for idx in range(artuple[0].size):
            i1 = []
            for ar in range(len(artuple)):
                i1.append(artuple[ar][idx] + min_order)
            res.append(tuple(i1))
        matching_orders[exponent] = res
    return matching_orders


def get_order_two(order):
    """Returns all terms, that may contribute to a nth order
       term that concists of two parts that are expanded
       perturbatively (e.g. S = <a|b>)"""
    ret = []
    for left in range(order + 1):
        for right in range(order + 1):
            if left + right == order:
                ret.append((left, right))
    return ret


def get_orders_three(order):
    """Returns all terms, that may contribute to a nth order
       term that involves three terms that are expanded perturbatively
       (e.g. a><b|c>)"""

    ret = []
    for left in range(order + 1):
        for middle in range(order + 1):
            for right in range(order + 1):
                if left + middle + right == order:
                    ret.append((left, middle, right))
    return ret
