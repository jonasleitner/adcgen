from sympy.core.function import diff
from sympy.physics.secondquant import (
    F, Fd, evaluate_deltas, wicks, substitute_dummies, NO
)
from sympy import symbols, latex, nsimplify

import numpy as np
from math import factorial

from indices import pretty_indices, split_idxstring
from groundstate import ground_state, Hamiltonian


class intermediate_states:
    def __init__(self, mp):
        self.gs = mp
        self.indices = mp.indices
        # self.invoked_spaces = mp.indices.invoked_spaces
        # {order: {'excitation_space': {"ket/bra": {indices: }}}}
        self.precursor = {}
        # {order: {'excitation_space': {'indices': x}}}
        self.overlap = {}
        self.S_root = {}
        # {order: {'excitation_space': {"ket/bra": x}}}
        self.isr = {}
        self.order_spaces = {
            "gs": 0,
            "ph": 1,
            "pphh": 2,
            "ppphhh": 3,
        }

    def get_precursor(self, order, space, braket, indices):
        """Method used to obtain precursor states of arbitrary order
           for an arbitrary space. The indices of the resulting precursor
           wavefunction need to be provided as string in the input
           (e.g. indices='ia' produces |PSI_{ia}^#).
           """

        # maybe interchange the space naming convention from 'ph' etc to just
        # the number of excited electrons 1, 2 etc.
        if space not in self.order_spaces:
            print(f"{space} is not a valid space. Valid spaces need to be",
                  f"in the self.order_spaces dict: {self.order_spaces}")
            exit()
        if braket not in ["bra", "ket"]:
            print(f"Unknown precursor wavefuntion type {braket}.",
                  "Only 'bra' and 'ket' are valid.")
            exit()
        if len(split_idxstring(indices)) != len(space):
            print("Number of Indices are not adequate for constructing a",
                  f"Precursor state for space {space}. Indices: {indices}.")
            exit()

        if order not in self.precursor:
            self.precursor[order] = {}
        if space not in self.precursor[order]:
            self.precursor[order][space] = {}
        if braket not in self.precursor[order][space]:
            self.precursor[order][space][braket] = {}

        # split the index str ij2a42b into [i,j2,a42,b]
        # and sort alphabetically afterwards
        indices = "".join(sorted(split_idxstring(indices)))

        # if space not in self.invoked_spaces:
        #    self.indices.invoke_space(space)
        if indices not in self.precursor[order][space][braket]:
            self.__build_precursor(order, space, braket, indices)
        return self.precursor[order][space][braket][indices]

    def __build_precursor(self, order, space, braket, indices):
        idx = self.indices.get_indices(indices)
        # import all bra and ket gs wavefunctions up to requested order
        # for nicer indices iterate first over bk
        isr = {}
        mp = {}
        for bk in ["bra", "ket"]:
            for o in range(order + 1):
                if o not in mp:
                    mp[o] = {}
                mp[o][bk] = self.gs.get_psi(o, bk)
        isr["gs"] = mp
        orders = get_orders_three(order)

        get_ov = {
            'ket': lambda ov: ov,
            'bra': lambda ov: [other for other in ["occ", "virt"]
                               if other != ov]
        }
        ov = get_ov[braket]

        # in contrast to the gs, here the operators are ordered as
        # abij instead of abji in order to stay consistent with the
        # published ADC results.
        operators = 1
        for symbol in idx["".join(ov('virt'))]:
            operators *= Fd(symbol)
        for symbol in idx["".join(ov('occ'))]:
            operators *= F(symbol)
        res = NO(operators) * mp[order][braket]
        # iterate over lower excitatied spaces (including gs)
        for lower_space, n in self.order_spaces.items():
            if space == lower_space:
                break
            if lower_space not in isr:
                isr[lower_space] = {}
                # get generic unique indices. Saved for reuse using
                # the alphabetically sorted indices of the parent precursor
                # state.
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
                    for s in occ:
                        idx_string += s.name
                    isr_idx[i+1] = "".join(idx_string)
                # generate the isr states for the lower_space
                for o in range(order + 1):
                    isr[lower_space][o] = {}
                    isr[lower_space][o]['bra'] = self.get_is(
                        o, lower_space, 'bra', idx_is=isr_idx[1],
                        idx_pre=isr_idx[3]
                    )
                    isr[lower_space][o]['ket'] = self.get_is(
                        o, lower_space, 'ket', idx_is=isr_idx[1],
                        idx_pre=isr_idx[2]
                    )
            lower_isr = isr[lower_space]
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
        self.precursor[order][space][braket][indices] = res
        print(f"Build precursor {space}_({indices})^({order}) {braket}:",
              latex(res))

    def get_pretty_precursor(self, order, space, braket):
        """Returns precursor bra/ket for a given space and order.
           Indices are not required. Default indices will be used,
           because they will be replaced anyway.
           """

        indices = {
            "ph": "ia",
            "pphh": "ijab",
            "ppphhh": "ijkabc",
        }
        if space not in indices:
            print("Can only build a pretty precursor state for the spaces",
                  f"{list(indices.keys())}.")
            exit()
        return self.make_pretty(
                self.get_precursor(order, space, braket, indices[space])
            )

    def get_overlap(self, order, space, indices):
        """Method that constructs and returns precursor overlap matrices
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
        if len(splitted) > 2:
            print("Only provide 2 strings of indices separated by a ','",
                  f"e.g. 'ia,jb'. {indices} is not valid.")
            exit()
        sorted_idx = []
        for idxstring in splitted:
            sorted_idx.append("".join(sorted(split_idxstring(idxstring))))
        if sorted_idx[0] == sorted_idx[1]:
            print("Indices for overlap matrix should not be equal.",
                  "Use e.g. ia,jb and not ia,ia.")
            exit()
        indices = ",".join(sorted_idx)

        if order not in self.overlap:
            self.overlap[order] = {}
        if space not in self.overlap[order]:
            self.overlap[order][space] = {}
        if indices not in self.overlap[order][space]:
            self.__build_overlap(order, space, indices)
        return self.overlap[order][space][indices]

    def __build_overlap(self, order, space, indices):
        # indices are sorted alphabetically already: ai,bj

        get_idx = {
            'bra': indices.split(",")[0],
            'ket': indices.split(",")[1]
        }
        precursor = {}
        for o in range(order + 1):
            precursor[o] = {}
            for braket in ["bra", "ket"]:
                precursor[o][braket] = self.get_precursor(
                    o, space, braket, indices=get_idx[braket]
                )

        orders = get_order_two(order)
        res = 0
        for term in orders:
            res += precursor[term[0]]["bra"] * precursor[term[1]]["ket"]
        res = wicks(res, keep_only_fully_contracted=True,
                    simplify_kronecker_deltas=True)
        self.overlap[order][space][indices] = res
        print(f"Build overlap {space} S_({indices})^({order}) = {latex(res)}")

    def get_pretty_overlap(self, order, space):
        """Returns the precursor overlap matrix of a given order
           for the defined space. There is no need to provide any indices,
           because they will be substituted and replaced anyway.
           Therefore, a default list with indices is included in the function,
           that provides some indices to construct some overlap matrix and
           precursor states.
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
        return self.make_pretty(
            self.get_overlap(order, space, indices=indices[space])
        )

    def make_pretty(self, expression):
        """Because it is not possible to make the indices of the
           precursor states and their overlap matrix pretty when building
           a method is created for this purpose.
           When indices are substituted sympy does not recognize them as
           equal anymore and therefore evaluates expressions wrong.
           """

        return substitute_dummies(
            expression, new_indices=True, pretty_indices=pretty_indices)

    def get_S_root(self, order, space, indices):
        """Method to obtain S^{-0.5} of a given order.
           Indices for the resulting matrix element are required.
           e.g. indices='ia,jb' produces (S^{-0.5})_{ia,jb}
           """

        if not isinstance(indices, str):
            print("Indices for S_root must be of type",
                  f"str, not {type(indices)}")
            exit()

        # sort indices in each part of the string alphabetically
        sorted_idx = []
        for idxstring in indices.split(","):
            sorted_idx.append("".join(sorted(split_idxstring(idxstring))))
        indices = ",".join(sorted_idx)

        if order not in self.S_root:
            self.S_root[order] = {}
        if space not in self.S_root[order]:
            self.S_root[order][space] = {}
        if indices not in self.S_root[order][space]:
            self.__build_S_root(order, space, indices)
        return self.S_root[order][space][indices]

    def __build_S_root(self, order, space, indices):
        overlap = {}
        for o in range(order + 1):
            overlap[o] = self.get_overlap(order, space, indices=indices)

        prefactors, orders = self.__expand_S_taylor(order)
        res = 0
        for exponent, termlist in orders.items():
            for term in termlist:
                for o in range(len(term)):
                    res += prefactors[exponent] * overlap[term[o]]
        self.S_root[order][space][indices] = res
        print(f"Build {space} S_root_({indices})^({order}) = {latex(res)}")

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
        print("prefactors of taylor expansion: ", diffs)
        orders = gen_order_S(order)
        if not orders:
            orders[0] = [(order,)]
        print("orders of overlap matrix: ", orders)
        return (diffs, orders)

    def get_pretty_S_root(self, order, space):
        """Returns S^{-0.5} of a given order for the defined space.
           Indices will be picked from the default list and replaced
           by "pretty" indices.
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
        return self.make_pretty(
            self.get_S_root(order, space, indices=indices[space])
        )

    def get_is(self, order, space, braket, idx_is, idx_pre):
        # idx_is='ia', idx_pre='jb'
        # target indices, second indices for precursor and overlap
        # if bra: S_{ia,jb} * <jb|
        # elif ket: S_{jb,ia} * |jb>
        idx_is = "".join(sorted(split_idxstring(idx_is)))
        idx_pre = "".join(sorted(split_idxstring(idx_pre)))

        if order not in self.isr:
            self.isr[order] = {}
        if space not in self.isr[order]:
            self.isr[order][space] = {}
        if braket not in self.isr[order][space]:
            self.isr[order][space][braket] = {}
        if ",".join([idx_is, idx_pre]) not in self.isr[order][space][braket]:
            self.__build_is(order, space, braket, idx_is, idx_pre)
        return self.isr[order][space][braket][",".join([idx_is, idx_pre])]

    def __build_is(self, order, space, braket, idx_is, idx_pre):
        get_s_indices = {
            'bra': ",".join([idx_is, idx_pre]),
            'ket': ",".join([idx_pre, idx_is])
        }
        precursor = {}
        s_root = {}
        for o in range(order + 1):
            precursor[o] = self.get_precursor(
                o, space, braket, indices=idx_pre
            )
            s_root[o] = self.get_S_root(
                o, space, indices=get_s_indices[braket]
            )

        orders = get_order_two(order)
        res = 0
        for term in orders:
            res += s_root[term[0]] * precursor[term[1]]
        res = evaluate_deltas(res)
        self.isr[order][space][braket][",".join([idx_is, idx_pre])] = res
        print(f"Build {space} ISR_({idx_is}, {idx_pre})^({order}) {braket} =",
              latex(res))

    def get_pretty_is(self, order, space, braket):
        indices = {
            'ph': {'is': "ia", 'pre': "jb"},
            'pphh': {'is': "iajb", 'pre': "klcd"},
            'ppphhh': {'is': "iajbkc", 'pre': "ldmenf"},
        }
        if space not in indices:
            print("Can only build pretty intermediate states for spaces",
                  f"{list(indices.keys())}.")
        idx = indices[space]
        return self.make_pretty(
            self.get_is(order, space, braket, idx["is"], idx["pre"])
        )


def gen_order_S(order):
    """Computed the series x + x² + ... with x = S(2) + S(3) + O(4).
       Returns all terms of the series that are of a given order as
       sorted by the exponent of x as key.
       {exponent: [(o1, o2, ...), ...]}
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


h = Hamiltonian()
mp = ground_state(h)
isr = intermediate_states(mp)
a = isr.get_precursor(1, "pphh", "ket", indices="iajb")
# a = isr.get_overlap(2, "ph", indices="ia,jb")
# a = isr.get_S_root(2, "ph", indices="ia,jb")
# a = isr.get_is(2, "ph", "ket", idx_is="ia", idx_pre="jb")
# a = isr.get_pretty_is(1, "ph", "ket")
b = isr.indices.substitute_indices(a)
print(latex(b))
