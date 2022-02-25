from sympy.core.function import diff
from sympy.physics.secondquant import (
    F, Fd, evaluate_deltas, wicks, NO, AntiSymmetricTensor,
    Dagger
)
from sympy import symbols, latex, nsimplify, Rational, sqrt

import numpy as np
from math import factorial

from indices import (
    split_idxstring, make_pretty, check_repeated_indices,
    get_n_ov_from_space, assign_index
)
from misc import cached_member


class intermediate_states:
    def __init__(self, mp, variant="pp"):
        self.gs = mp
        self.indices = mp.indices

        variants = ["pp", "ea", "ip"]
        if variant not in variants:
            print(f"The ADC variant {variant} is not valid. "
                  f"Supported variants are {variants}.")
            exit()
        self.variant = variant

    @cached_member
    def precursor(self, order, space, braket, indices):
        """Method used to obtain precursor states of specified order
           for the specified space. The indices of the resulting precursor
           wavefunction need to be provided as string in the input
           (e.g. indices='ia' produces |PSI_{ia}^#>).
           """

        if not self.check_valid_space(space):
            print(f"{space} is not a valid space for a {self.variant} "
                  "ISR state.")
            exit()
        if braket not in ["bra", "ket"]:
            print(f"Unknown precursor wavefuntion type {braket}."
                  "Only 'bra' and 'ket' are valid.")
            exit()

        # check compatibiliity of indices and space
        idx_ov = {'occ': 0, 'virt': 0, 'general': 0}
        for idx in split_idxstring(indices):
            idx_ov[assign_index(idx)] += 1
        if idx_ov['general']:
            print(f"The provided indices {indices} include a general index.")
            exit()
        n_ov_space = get_n_ov_from_space(space)
        if idx_ov["occ"] != n_ov_space["n_occ"] or \
                idx_ov["virt"] != n_ov_space["n_virt"]:
            print(f"The provided indices {indices} (occ: {idx_ov['occ']} / "
                  f"virt: {idx_ov['virt']}) do not match the required amount "
                  f"of indices for the space {space} (occ: "
                  f"{n_ov_space['n_occ']} / virt: {n_ov_space['n_virt']}).")
            exit()

        # import all bra and ket gs wavefunctions up to requested order
        # for nicer indices iterate first over bk
        mp = {}
        for bk in ["bra", "ket"]:
            for o in range(order + 1):
                if o not in mp:
                    mp[o] = {}
                mp[o][bk] = self.gs.psi(o, bk)

        # get all possible combinations a*b*c of the desired order
        orders = get_orders_three(order)

        idx = self.indices.get_indices(indices)
        # in contrast to the gs, here the operators are ordered as
        # abij instead of abji in order to stay consistent with the
        # ADC results.
        operators = 1
        if idx.get('virt'):
            for symbol in idx['virt']:
                operators *= Fd(symbol)
        if idx.get('occ'):
            for symbol in idx['occ']:
                operators *= F(symbol)
        if braket == "bra":
            operators = Dagger(operators)

        # no need to differentiate bra/ket here, because
        # operators * mp = mp * operators (there is always an equal number of
        # p/h operators in mp that needs to be moved to the other side.
        # Will always give +.)
        res = NO(operators) * mp[order][braket]

        # orthogonalise with respect to the ground state for pp ADC.
        if self.variant == "pp":
            for term in orders:
                # |Y>  <--  -|X><X|Y>
                if braket == "ket":
                    i1 = mp[term[1]]["bra"] * NO(operators) * \
                        mp[term[2]]["ket"]
                    state = mp[term[0]]["ket"]
                # <Y|  <--  -<Y|X><X|
                elif braket == "bra":
                    i1 = mp[term[0]]["bra"] * NO(operators) * \
                        mp[term[1]]["ket"]
                    state = mp[term[2]]["bra"]
                # wicks automatically expands the passed expression
                i1 = wicks(
                    i1, keep_only_fully_contracted=True,
                    simplify_kronecker_deltas=True,
                )
                res -= state * i1

        # iterate over lower excitated spaces
        lower_spaces = self.__generate_lower_spaces(space)
        for lower_space in lower_spaces:
            # get generic unique indices to generate the lower_isr_states.
            n_ov = get_n_ov_from_space(lower_space)
            isr_generic = self.indices.get_isr_indices(
                indices, n_occ=3*n_ov["n_occ"], n_virt=3*n_ov["n_virt"]
            )
            # sort indices into three strings of length n_o + n_v
            isr_idx = {}
            for i in range(3):
                isr_idx[i] = []
                if n_ov["n_occ"]:
                    n_o = n_ov["n_occ"]
                    occ = isr_generic['occ'][i*n_o:(i+1)*n_o]
                    isr_idx[i].extend([s.name for s in occ])
                if n_ov["n_virt"]:
                    n_v = n_ov["n_virt"]
                    virt = isr_generic['virt'][i*n_v:(i+1)*n_v]
                    isr_idx[i].extend([s.name for s in virt])
                isr_idx[i] = "".join(isr_idx[i])

            # generate the isr states for the lower_space
            lower_isr = {}
            for o in range(order + 1):
                lower_isr[o] = {}
                lower_isr[o]['bra'] = self.intermediate_state(
                    o, lower_space, 'bra', idx_is=isr_idx[0],
                    idx_pre=isr_idx[2]
                )
                lower_isr[o]['ket'] = self.intermediate_state(
                    o, lower_space, 'ket', idx_is=isr_idx[0],
                    idx_pre=isr_idx[1]
                )

            # orthogonalise with respsect to the lower excited ISR state
            for term in orders:
                # |Y>  <--  -|X><X|Y>
                prefactor = Rational(
                    1, factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
                )
                if braket == "ket":
                    i1 = prefactor * lower_isr[term[1]]["bra"] * \
                        NO(operators) * mp[term[2]]["ket"]
                    state = lower_isr[term[0]]["ket"]
                # <Y|  <--  -<Y|X><X|
                elif braket == "bra":
                    i1 = prefactor * mp[term[0]]["bra"] * NO(operators) \
                        * lower_isr[term[1]]["ket"]
                    state = lower_isr[term[2]]["bra"]
                i1 = wicks(
                    i1, keep_only_fully_contracted=True,
                    simplify_kronecker_deltas=True,
                )
                res -= state * i1

        print(f"Build precursor {space}_({indices})^({order}) {braket}:",
              latex(res))
        return res

    @cached_member
    def overlap_precursor(self, order, space, indices):
        """Method to obtain precursor overlap matrices
           for a given order and space. Indices of the resulting
           overlap matrix element need to be provided in the form 'ia,jb'
           which will produce S_{ia,jb}.
           """

        if not isinstance(indices, str):
            print("Indices for precursor overlap matrix must be of type",
                  f"str, not {type(indices)}")
            exit()

        splitted = indices.split(",")
        if len(splitted) != 2:
            print("Only provide 2 strings of indices separated by a ','"
                  f"e.g. 'ia,jb'. {indices} is not valid.")
            exit()
        if check_repeated_indices(splitted[0], splitted[1]):
            print("Repeated index found in indices of overlap matrix "
                  f"{splitted[0]}, {splitted[1]}.")
            exit()

        # calculate the required precursor states.
        get_idx = {
            'bra': indices.split(",")[0],
            'ket': indices.split(",")[1]
        }
        precursor = {}
        for o in range(order + 1):
            precursor[o] = {}
            for braket in ["bra", "ket"]:
                precursor[o][braket] = self.precursor(
                    o, space, braket, indices=get_idx[braket]
                )

        orders = get_order_two(order)

        res = 0
        for term in orders:
            res += precursor[term[0]]["bra"] * precursor[term[1]]["ket"]
        res = wicks(res, keep_only_fully_contracted=True,
                    simplify_kronecker_deltas=True)
        print(f"Build overlap {space} S_({indices})^({order}) = {latex(res)}")
        return res

    @cached_member
    def s_root(self, order, space, indices):
        """Method to obtain S^{-0.5} of a given order.
           Indices for the resulting matrix element are required.
           e.g. indices='ia,jb' produces (S^{-0.5})_{ia,jb}
           """

        if not isinstance(indices, str):
            print("Indices for S_root must be of type "
                  f"str, not {type(indices)}")
            exit()

        prefactors, orders = self.expand_S_taylor(order, min_order=2)
        res = 0
        for exponent, termlist in orders.items():
            for term in termlist:
                i1 = prefactors[exponent]
                for o in term:
                    i1 *= self.overlap_precursor(o, space, indices=indices)
                res += i1.expand()
        print(f"Build {space} S_root_({indices})^({order}) = {latex(res)}")
        return res

    @cached_member
    def intermediate_state(self, order, space, braket, idx_is, idx_pre):
        """Returns an intermediate state. The index string idx_is represents
           the indices of the returned Intermediate State. The index string
           idx_pre is used to generate S^(-0.5), i.e. the sum
           sum_I' |I'> S_{I',I}^(-0.5) // sum_I' S_{I,I'}^(-0.5) <I'|
           runs over idx_pre.
           """

        get_s_indices = {
            'bra': ",".join([idx_is, idx_pre]),
            'ket': ",".join([idx_pre, idx_is])
        }
        precursor = {}
        s_root = {}
        for o in range(order + 1):
            precursor[o] = self.precursor(
                o, space, braket, indices=idx_pre
            )
            s_root[o] = self.s_root(
                o, space, indices=get_s_indices[braket]
            )

        orders = get_order_two(order)
        n_ov = get_n_ov_from_space(space)
        prefactor = Rational(
            1, factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        res = 0
        for term in orders:
            res += (prefactor * s_root[term[0]] * precursor[term[1]]).expand()
        res = evaluate_deltas(res)
        print(f"Build {space} ISR_({idx_is} <- {idx_pre})^({order}) {braket} "
              f"= {latex(res)}")
        return res

    def overlap_isr(self, order, space, indices):
        pass

    def amplitude_vector(self, space, indices, lr="right"):
        """Returns an amplitude vector for the requested space using
           the provided indices.
           Note that, also a prefactor is returned that keeps the
           normalization of Y if the index restrictions are dropped.
           """

        if len(space) != len(split_idxstring(indices)):
            print(f"Indices {indices} not valid for space {space}.")
            exit()

        idx = self.indices.get_indices(indices)
        for ov in ["occ", "virt"]:
            if ov not in idx:
                idx[ov] = []

        # normally when lifting the index restrictions a prefactor of
        # p = 1/(no! * nv!) is necessary
        # However, in order to keep the amplitude vector normalized
        # sqrt(p) is packed in each amplitude vector.
        # For PP ADC this gives 1/(n!)^2 * <Y|Y> which keeps the normalization
        # of Y. The remaining factor sqrt(p) is visible in the MVP expression
        n_ov = get_n_ov_from_space(space)
        prefactor = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        t_string = {
            "right": "Y",
            "left": "X",
        }
        print("prefactor from ampl side: ", prefactor)
        return prefactor * AntiSymmetricTensor(
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
        n_ov = get_n_ov_from_space(space_str)
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

        smallest = {
            "pp": "ph",
            "ip": "h",
            "ea": "p",
        }
        if space_str == smallest[self.variant]:
            return True

        lower_spaces = self.__generate_lower_spaces(space_str)
        valid = False
        for s in lower_spaces:
            if s == smallest[self.variant]:
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
            print("Can only build a pretty precursor state for the spaces",
                  f"{list(indices.keys())}.")
            exit()
        return make_pretty(
            self.precursor(order, space, braket, indices[space])
        )

    def pretty_overlap(self, order, space):
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
            print("Can only build a pretty overlap matrix for the spaces",
                  f"{list(indices.keys())}.")
            exit()
        return make_pretty(
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
            print("Can only build pretty S_root for the spaces",
                  f"{list(indices.keys())}.")
            exit()
        return make_pretty(
            self.s_root(order, space, indices[space])
        )

    def pretty_intermediate_state(self, order, space, braket):
        """Returns a bra/ket intermediate state for a given space and order.
           Makes the result pretty by substituting indices. Therefore,
           no further calculations are possible with the resulting expression!
           """

        indices = {
            'ph': {'is': "ia", 'pre': "jb"},
            'pphh': {'is': "iajb", 'pre': "klcd"},
            'ppphhh': {'is': "iajbkc", 'pre': "ldmenf"},
        }
        if space not in indices:
            print("Can only build pretty intermediate states for spaces",
                  f"{list(indices.keys())}.")
        idx = indices[space]
        return make_pretty(
            self.intermediate_state(order, space, braket, idx["is"],
                                    idx["pre"])
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
