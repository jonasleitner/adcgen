from sympy.core.function import diff
from sympy.physics.secondquant import (
    F, Fd, evaluate_deltas, wicks, NO, AntiSymmetricTensor
)
from sympy import symbols, latex, nsimplify

import numpy as np
from math import factorial

from indices import split_idxstring, make_pretty, check_repeated_indices
from misc import cached_member


class intermediate_states:
    def __init__(self, mp):
        self.gs = mp
        self.indices = mp.indices
        self.order_spaces = {
            "ph": 1,
            "pphh": 2,
            "ppphhh": 3,
            "pppphhhh": 4,
        }

    @cached_member
    def precursor(self, order, space, braket, indices):
        """Method used to obtain precursor states of arbitrary order
           for an arbitrary space. The indices of the resulting precursor
           wavefunction need to be provided as string in the input
           (e.g. indices='ia' produces |PSI_{ia}^#).
           """

        if space not in self.order_spaces:
            print(f"{space} is not a valid space. Valid spaces are"
                  f"{self.order_spaces}.")
            exit()
        if braket not in ["bra", "ket"]:
            print(f"Unknown precursor wavefuntion type {braket}."
                  "Only 'bra' and 'ket' are valid.")
            exit()
        if len(split_idxstring(indices)) != len(space):
            print("Number of Indices are not adequate for constructing a",
                  f"Precursor state for space {space}. Indices: {indices}.")
            exit()

        # split the index str ij2a42b into [i,j2,a42,b]
        # and sort alphabetically afterwards
        indices = "".join(sorted(split_idxstring(indices)))
        idx = self.indices.get_indices(indices)

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

        get_ov = {
            'ket': lambda ov: ov,
            'bra': lambda ov: [other for other in ["occ", "virt"]
                               if other != ov]
        }
        ov = get_ov[braket]

        # in contrast to the gs, here the operators are ordered as
        # abij instead of abji in order to stay consistent with the
        # ADC results.
        operators = 1
        for symbol in idx["".join(ov('virt'))]:
            operators *= Fd(symbol)
        for symbol in idx["".join(ov('occ'))]:
            operators *= F(symbol)
        res = NO(operators) * mp[order][braket]

        # orthogonalise with respect to the ground state
        for term in orders:
            # |Y>  <--  -|X><X|Y>
            if braket == "ket":
                i1 = mp[term[1]]["bra"] * NO(operators) * mp[term[2]]["ket"]
                state = mp[term[0]]["ket"]
            # <Y|  <--  -<Y|X><X|
            elif braket == "bra":
                i1 = mp[term[0]]["bra"] * NO(operators) * mp[term[1]]["ket"]
                state = mp[term[2]]["bra"]
            i1 = wicks(
                i1, keep_only_fully_contracted=True,
                simplify_kronecker_deltas=True,
            )
            res -= state * i1

        # iterate over lower excitated spaces
        for n in range(1, self.order_spaces[space]):
            # generate name string of the lower excited space
            lower_space = ["p" for i in range(n)]
            lower_space.extend(["h" for i in range(n)])
            lower_space = "".join(lower_space)

            # get generic unique indices to generate the lower_isr_states.
            isr_generic = self.indices.get_isr_indices(
                indices, n_occ=3*n, n_virt=3*n
            )
            # sort indices into three strings of length n
            isr_idx = {}
            for i in range(3):
                occ = isr_generic['occ'][i*n:(i+1)*n]
                virt = isr_generic['virt'][i*n:(i+1)*n]
                # extract the names of the generated symbols
                idx_string = [s.name for s in virt]
                idx_string.extend([s.name for s in occ])
                isr_idx[i] = "".join(idx_string)

            # generate the isr states for the lower_space
            lower_isr = {}
            for o in range(order + 1):
                lower_isr[o] = {}
                lower_isr[o]['bra'] = self.intermediate_state(
                    o, lower_space, 'bra', isr_idx[0], isr_idx[2]
                )
                lower_isr[o]['ket'] = self.intermediate_state(
                    o, lower_space, 'ket', isr_idx[0], isr_idx[1]
                )

            # orthogonalise with respsect to the lower excited ISR state
            for term in orders:
                # |Y>  <--  -|X><X|Y>
                if braket == "ket":
                    i1 = lower_isr[term[1]]["bra"] * \
                        NO(operators) * mp[term[2]]["ket"]
                    state = lower_isr[term[0]]["ket"]
                # <Y|  <--  -<Y|X><X|
                elif braket == "bra":
                    i1 = mp[term[0]]["bra"] * NO(operators) \
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
    def overlap(self, order, space, indices):
        """Method to obtain precursor overlap matrices
           for a given order and space. Indices of the resulting
           overlap matrix element need to be provided in the form 'ia,jb'
           which will produce S_{ia,jb}.
           """

        if not isinstance(indices, str):
            print("Indices for precursor overlap matrix must be of type",
                  f"str, not {type(indices)}")
            exit()

        # sort both parts of the index string alphabetically
        splitted = indices.split(",")
        if len(splitted) != 2:
            print("Only provide 2 strings of indices separated by a ','"
                  f"e.g. 'ia,jb'. {indices} is not valid.")
            exit()
        if check_repeated_indices(splitted[0], splitted[1]):
            print("Repeated index found in indices of overlap matrix "
                  f"{splitted[0]}, {splitted[1]}.")
            exit()
        sorted_idx = []
        for idxstring in splitted:
            split = sorted(split_idxstring(idxstring))
            sorted_idx.append("".join(split))
        if sorted_idx[0] == sorted_idx[1]:
            print("Indices for overlap matrix should not be equal."
                  f"Provided indices {indices}.")
            exit()
        indices = ",".join(sorted_idx)

        # calculate the precursor states.
        get_idx = {
            'bra': indices.split(",")[0],
            'ket': indices.split(",")[1]
        }
        precursor = {}
        for o in range(order + 1):
            precursor[o] = {}
            for braket in ["bra", "ket"]:
                precursor[o][braket] = self.precursor(
                    o, space, braket, get_idx[braket]
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
            print("Indices for S_root must be of type"
                  f"str, not {type(indices)}")
            exit()

        # sort indices in each part of the string alphabetically
        sorted_idx = []
        for idxstring in indices.split(","):
            sorted_idx.append("".join(sorted(split_idxstring(idxstring))))
        indices = ",".join(sorted_idx)

        overlap = {}
        for o in range(order + 1):
            overlap[o] = self.overlap(order, space, indices)

        prefactors, orders = self.__expand_S_taylor(order)
        res = 0
        for exponent, termlist in orders.items():
            for term in termlist:
                for o in range(len(term)):
                    res += prefactors[exponent] * overlap[term[o]]
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

        idx_is = "".join(sorted(split_idxstring(idx_is)))
        idx_pre = "".join(sorted(split_idxstring(idx_pre)))

        get_s_indices = {
            'bra': ",".join([idx_is, idx_pre]),
            'ket': ",".join([idx_pre, idx_is])
        }
        precursor = {}
        s_root = {}
        for o in range(order + 1):
            precursor[o] = self.precursor(
                o, space, braket, idx_pre
            )
            s_root[o] = self.s_root(
                o, space, get_s_indices[braket]
            )

        orders = get_order_two(order)
        res = 0
        for term in orders:
            res += s_root[term[0]] * precursor[term[1]]
        res = evaluate_deltas(res)
        print(f"Build {space} ISR_({idx_is}, {idx_pre})^({order}) {braket} = ",
              latex(res))
        return res

    def amplitude_vector(self, space, indices):
        """Returns an amplitude vector for the requested space using
           the provided indices.
           """

        if len(space) != len(split_idxstring(indices)):
            print(f"Indices {indices} not valid for space {space}.")
            exit()

        idx = self.indices.get_indices(indices)

        return AntiSymmetricTensor("Y", tuple(idx["virt"]), tuple(idx["occ"]))

    def __expand_S_taylor(self, order):
        """Computes the Taylor expansion of 'S^{-0.5} = (1 + x)^{-0.5} with
           'x = S(2) + S(3) + O(4)' to a given order in perturbation theory.

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
        for o in range(1, int(order / 2) + 1):
            intermediate = diff(intermediate, x)
            diffs[o] = nsimplify(intermediate.subs(x, 0) * 1 / factorial(o))
        orders = gen_order_S(order)
        if not orders:
            orders[0] = [(order,)]
        return (diffs, orders)

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
            self.overlap(order, space, indices[space])
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


def gen_order_S(order):
    """Computes the series x + x² + ... with x = S(2) + S(3) + O(4).
       Returns all terms of the series that are of a given order
       sorted by the exponent of x, i.e.
       {exponent: [(o1, o2, ...), ...]}.
       """

    orders = np.array([2 ** o for o in range(2, order + 1)])
    multiplied = {}
    if (int(order / 2) - 1) > 0:
        res = orders
        for exponent in range(int(order / 2) - 1):
            i1 = np.multiply.outer(res, orders)
            multiplied[exponent + 2] = i1
            res = i1
    else:
        multiplied[1] = orders

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
                i1.append(artuple[ar][idx] + 2)
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
