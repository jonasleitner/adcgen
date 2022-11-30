from sympy.physics.secondquant import wicks
from sympy import latex, Rational, S, Mul, sympify

from math import factorial

from .indices import (
    n_ov_from_space, repeated_indices, indices, extract_names
)
from .misc import (cached_member, Inputerror, transform_to_tuple,
                   validate_input, process_arguments)
from .simplify import simplify
from .func import evaluate_deltas, gen_term_orders
from .groundstate import ground_state


class intermediate_states:
    def __init__(self, mp, variant="pp"):
        if not isinstance(mp, ground_state):
            raise Inputerror("Invalid ground state object.")
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
        """Method to obtain precursor states.
           The indices of the precursor wavefunction need to be provided as
           string in the input (e.g. indices='ia' produces |PSI_{ia}^#>).
           """
        from sympy.physics.secondquant import F, Fd, NO, Dagger

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
        if not self.validate_space(space):
            raise Inputerror(f"{space} is not a valid space for "
                             f"{self.variant} ADC.")

        # get the target symbols of the precursor state
        idx = self.indices.get_indices(indices)
        # check compatibility of indices and space
        if idx.get('general'):
            raise Inputerror(f"The provided indices {indices} include a "
                             "general index.")
        n_ov = n_ov_from_space(space)
        if len(idx.get('occ', [])) != n_ov.get('n_occ', 0) or \
                len(idx.get('virt', [])) != n_ov.get('n_virt', 0):
            raise Inputerror(f"The indices {indices} and the space {space} "
                             "are not compatible.")

        # in contrast to the gs, here the operators are ordered as
        # abij instead of abji in order to stay consistent with the
        # ADC results.
        operators = 1
        if idx.get('virt'):
            operators *= Mul(*[Fd(s) for s in idx['virt']])
        if idx.get('occ'):
            operators *= Mul(*[F(s) for s in idx['occ']])
        if braket == "bra":
            operators = Dagger(operators)

        # leading term:
        # no need to differentiate bra/ket here, because
        # operators * mp = mp * operators (there is always an equal number of
        # p/h operators in mp that needs to be moved to the other side.
        # Will always give +.)
        max_gs = self.gs.psi(order=order, braket=braket)
        res = (NO(operators) * max_gs).expand()

        # get all terms of a*b of the desired order (ground state norm)
        orders = gen_term_orders(order=order, term_length=2, min_order=0)

        # orthogonalise with respect to the ground state for pp ADC.
        # checked up to 4th order!
        if self.variant == "pp":
            # import all ground state wave functions that may not appear twice
            # in |a><b|c>, i.e. all of: order > int(order/2)
            gs_psi = {'bra': {}, 'ket': {}}
            gs_psi[braket][order] = max_gs
            for o in range(order//2 + 1, order+1):
                if not gs_psi['bra'].get(o):
                    gs_psi['bra'][o] = self.gs.psi(order=o, braket='bra')
                if not gs_psi['ket'].get(o):
                    gs_psi['ket'][o] = self.gs.psi(order=o, braket='ket')
            def get_gs_wfn(o, bk): return gs_psi[bk][o] if o > order//2 else \
                self.gs.psi(order=o, braket=bk)
            # 1) iterate through all combinations of norm_factor*projector
            for norm_term in orders:
                norm = self.gs.norm_factor(norm_term[0])
                if norm is S.Zero:
                    continue
                # 2) construct the projector for a given norm_factor
                # the overall order is split between the norm_factor and the
                # projector
                orders_projection = gen_term_orders(
                    order=norm_term[1], term_length=3, min_order=0
                )
                projection = sympify(0)
                for term in orders_projection:
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
                    projection += (state * i1).expand()
                projection = evaluate_deltas(projection).sympy
                res -= (norm * projection).expand()
            gs_psi.clear()

        # iterate over lower excitated spaces
        lower_spaces = self.__generate_lower_spaces(space)
        for lower_space in lower_spaces:
            # get generic unique indices to generate the lower_isr_states.
            n_ov = n_ov_from_space(lower_space)
            idx_isr = self.indices.get_generic_indices(**n_ov)
            idx_isr = "".join(extract_names(idx_isr))

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
                orders_projection = gen_term_orders(
                    norm_term[1], term_length=3, min_order=0
                )
                projection = sympify(0)
                for term in orders_projection:
                    # |Y#>  <--  -|X><X|Y>
                    if braket == "ket":
                        i1 = (self.intermediate_state(order=term[1],
                                                      space=lower_space,
                                                      braket="bra",
                                                      indices=idx_isr) *
                              NO(operators) * self.gs.psi(order=term[2],
                                                          braket="ket")
                              )
                        state = self.intermediate_state(
                            order=term[0], space=lower_space, braket="ket",
                            indices=idx_isr
                        )
                    # <Y#|  <--  -<Y|X><X|
                    elif braket == "bra":
                        i1 = (self.gs.psi(order=term[0], braket="bra") *
                              NO(operators) *
                              self.intermediate_state(order=term[1],
                                                      space=lower_space,
                                                      braket="ket",
                                                      indices=idx_isr)
                              )
                        state = self.intermediate_state(
                            order=term[2], space=lower_space, braket="bra",
                            indices=idx_isr
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
    def overlap_precursor(self, order, block, indices):
        """Method to obtain precursor overlap matrices
           for a given order and space. Indices of the resulting
           overlap matrix element need to be provided in the form 'ia,jb'
           which will produce S_{ia,jb}.
           """

        # no need to do more validation here -> will be done in precursor
        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("2 index strings required for an overlap matrix "
                             f"block. Got: {indices}.")

        if repeated_indices(indices[0], indices[1]):
            raise Inputerror("Repeated index found in indices of precursor "
                             f"overlap matrix: {indices}.")

        orders = gen_term_orders(order=order, term_length=2, min_order=0)

        res = 0
        # 1) iterate through all combinations of norm_factor*S
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct S for a given norm factor
            # the overall order is split between he norm_factor and S
            orders_overlap = gen_term_orders(
                order=norm_term[1], term_length=2, min_order=0
            )
            overlap = sympify(0)
            for term in orders_overlap:
                i1 = (self.precursor(order=term[0], space=block[0],
                                     braket="bra", indices=indices[0]) *
                      self.precursor(order=term[1], space=block[1],
                                     braket="ket", indices=indices[1]))
                i1 = wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
                overlap += i1
            res += (norm * overlap).expand()
        # It should be valid to simplifiy the result by permuting contracted
        # indices before returning -> should lower the overall size of the
        # final expression
        res = simplify(res)
        print(f"Build overlap {block} S_{indices}^({order}) = {res}")
        return res.sympy

    @process_arguments
    @cached_member
    def s_root(self, order, block, indices):
        """Method to obtain S^{-0.5} of a given order.
           Indices for the resulting matrix element are required.
           e.g. indices='ia,jb' produces (S^{-0.5})_{ia,jb}
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("2 index strings required for a block of the "
                             "inverse suqare root of the overlap matrix. "
                             f"Got: {indices}.")
        if repeated_indices(indices[0], indices[1]):
            raise Inputerror(f"Repeated index found in indices {indices}.")
        if block[0] != block[1]:
            raise NotImplementedError("Off diagonal blocks of the overlap "
                                      "matrix should be 0 by definition. "
                                      "Simply don't know how to handle the "
                                      "index generation needed in this case.")

        taylor_expansion = self.expand_S_taylor(order, min_order=2)
        # assume in the following that both spaces are equal!!
        n_ov = n_ov_from_space(block[0])
        # create an index list: first and last element are the two provided
        # idx strings
        idx = list(indices)
        res = sympify(0)
        # iterate over exponents and terms, starting with the lowest exponent
        for pref, termlist in taylor_expansion:
            # generate len(termlist)-1 or exponent-1 index spaces, e.g. for x*x
            # 1 additional space is required: s,s' = s,s''*s'',s'
            # x^3: s,s' = s,s'' * s'',s''' * s''',s' etc.
            while len(idx)-1 < len(termlist[0]):
                new = self.indices.get_generic_indices(**n_ov)
                new = "".join(extract_names(new))
                idx.insert(-1, new)
            for term in termlist:
                i1 = pref
                for n, o in enumerate(term):
                    i1 *= self.overlap_precursor(order=o, block=block,
                                                 indices=(idx[n], idx[n+1]))
                    if i1 is S.Zero:
                        break
                # in combined terms S*S delta evaluation might be necessary
                res += evaluate_deltas(i1.expand()).sympy
        print(f"Build {block} S_root_{indices}^({order}) = {latex(res)}")
        return res

    @process_arguments
    @cached_member
    def intermediate_state(self, order, space, braket, indices):
        """Method for constructing an intermediate state using the provided
           indices.
           """

        indices = transform_to_tuple(indices)
        space = transform_to_tuple(space)
        validate_input(order=order, space=space, braket=braket,
                       indices=indices)
        if len(indices) != 1:
            raise Inputerror(f"{indices} are not valid for "
                             "constructing an intermediate state.")
        indices = indices[0]
        space = space[0]

        # generate additional indices for the precursor state
        n_ov = n_ov_from_space(space)
        idx_pre = self.indices.get_generic_indices(**n_ov)
        idx_pre = "".join(extract_names(idx_pre))

        prefactor = Rational(
            1, factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        # sandwich the IS and precursor indices together
        s_indices = {
            'bra': ",".join([indices, idx_pre]),
            'ket': ",".join([idx_pre, indices])
        }

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = sympify(0)
        for term in orders:
            i1 = (prefactor *
                  self.s_root(order=term[0], block=(space, space),
                              indices=s_indices[braket]) *
                  self.precursor(order=term[1], space=space, braket=braket,
                                 indices=idx_pre))
            res += evaluate_deltas(i1.expand()).sympy
        print(f"Build {space} ISR_({indices}^({order}) "
              f"{braket} = {latex(res)}")
        return res

    @process_arguments
    @cached_member
    def overlap_isr(self, order, block, indices):
        """Computes a block of the overlap matrix in the ISR basis."""

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("Constructing a ISR overlap matrix block requires"
                             f" 2 index strings. Provided: {indices}.")

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = sympify(0)
        # 1) iterate through all combinations of norm_factor*S
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct S for a given norm factor
            # the overall order is split between he norm_factor and S
            orders_overlap = gen_term_orders(
                order=norm_term[1], term_length=2, min_order=0
            )
            overlap = sympify(0)
            for term in orders_overlap:
                i1 = (self.intermediate_state(order=term[0], space=block[0],
                                              braket="bra",
                                              indices=indices[0]) *
                      self.intermediate_state(order=term[1], space=block[1],
                                              braket="ket",
                                              indices=indices[1]))
                i1 = wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
                overlap += i1
            res += (norm * overlap).expand()
        print(f"Build ISR overlap {block} S_{indices}^({order}) = ",
              latex(res))
        return res

    @process_arguments
    @cached_member
    def amplitude_vector(self, indices, lr="right"):
        """Returns an amplitude vector using the provided indices.
           """
        from sympy.physics.secondquant import AntiSymmetricTensor

        validate_input(indices=indices, lr=lr)

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
        """Computes the all n-h order contributions to the Taylor expansion of
           'S^{-0.5} = (1 + x)^{-0.5} with 'x = sum_{n=1} S^(n)'.
           min_order defines the lowest order at n where the overlap matrix
           is non-zero, excluding the zeroth order contribution.

           Returns two dicts:
           The first one contains the prefactors of the series x + x² + ...
           The second one contains the orders of S that contribute to the
           n-th order term. (like (4,) and (2,2) for fourth order)
           In both dicts the exponent of x in the Taylor expansion is used
           as key.
           """
        from sympy import diff, nsimplify, symbols

        validate_input(order=order, min_order=min_order)
        if min_order == 0:
            raise Inputerror("A minimum order of 0 does not make sense here.")

        # below min_order all orders - except the zeroth order contribution -
        # should be zero. Should be handled automatically if the corresponding
        # orders are forwarded to the overlap method.
        if order < min_order:
            return [(1, [(order,)])]

        x = symbols('x')
        f = (1 + x) ** -0.5
        ret = []
        for exp in range(1, order//min_order + 1):
            f = diff(f, x)
            pref = nsimplify(f.subs(x, 0) / factorial(exp), rational=True)
            orders = gen_term_orders(
                order=order, term_length=exp, min_order=min_order
            )
            ret.append((pref, orders))
        return ret

    def __generate_lower_spaces(self, space_str):
        """Generate all lower spaces for the provided space string, e.g.
           ['ph'] for 'pphh'."""
        lower_spaces = []
        for _ in range(min(space_str.count('p'), space_str.count('h'))):
            space_str = space_str.replace('p', '', 1).replace('h', '', 1)
            if not space_str:
                break
            lower_spaces.append(space_str)
        return lower_spaces

    def validate_space(self, space_str):
        """Checks wheter the provided space is a valid space for
           the current ADC variant.
           """

        if space_str in self.min_space:
            return True

        lower_spaces = self.__generate_lower_spaces(space_str)
        return any([sp in self.min_space for sp in lower_spaces])
