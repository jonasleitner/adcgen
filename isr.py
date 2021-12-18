from sympy.core.function import diff
from sympy.physics.secondquant import (
    F, Fd, evaluate_deltas, wicks, substitute_dummies, NO
)
from sympy import symbols, latex, nsimplify

import numpy as np
from math import factorial

from groundstate import ground_state, Hamiltonian
from indices import pretty_indices


class intermediate_states:
    def __init__(self, mp):
        self.gs = mp
        self.indices = mp.indices
        self.invoked_spaces = mp.indices.invoked_spaces
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
        self.default_indices = {
            "ph": {"bra": "ia", "ket": "jb"},
            "pphh": {"bra": "ijab", "ket": "klcd"},
            "ppphhh": {"bra": "ijkabc", "ket": "lmndef"}
        }

    def get_precursor(self, order, space, braket, indices=None):
        # input order, space, bra/ket, indices="ia" o.ä.
        # order: int, space: ph/pphh etc, braket: bra or ket, indices: iajb

        if space not in self.order_spaces:
            print(f"{space} is not a valid space. Valid spaces need to be",
                  f"in the self.order_spaces dict: {self.order_spaces}")
            exit()
        if braket not in ["bra", "ket"]:
            print(f"Unknown precursor wavefuntion type {braket}.",
                  "Only 'bra' and 'ket' are valid.")
            exit()

        if order not in self.precursor:
            self.precursor[order] = {}
        if space not in self.precursor[order]:
            self.precursor[order][space] = {}
        if braket not in self.precursor[order][space]:
            self.precursor[order][space][braket] = {}

        if not indices:
            try:
                indices = self.default_indices[space][braket]
            except KeyError:
                print(f"No default indices available for space {space}",
                      f"{braket}. Either add some default indices or",
                      "provide another space.")
                exit()

        indices = "".join(sorted(indices))
        # only works if namin spaces ph, pphh etc.
        if len(indices) != len(space):
            print(f"{indices} are not adequate for space {space}.",
                  "Too much or few indices provded")
            exit()

        if space not in self.invoked_spaces:
            self.indices.invoke_space(space)
        if indices not in self.precursor[order][space][braket]:
            self.__build_precursor(order, space, braket, indices)
        return self.precursor[order][space][braket][indices]

    def __build_precursor(self, order, space, braket, indices):
        idx = self.indices.get_indices(space, indices)
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
        get_bk = {
            'ket': lambda bk: bk,
            'bra': lambda bk: [other for other in ["bra", "ket"]
                               if other != bk]
        }
        ov = get_ov[braket]
        bk = get_bk[braket]

        operators = 1
        for symbol in idx["".join(ov('virt'))]:
            operators *= Fd(symbol)
        for symbol in reversed(idx["".join(ov('occ'))]):
            operators *= F(symbol)
        res = NO(operators) * mp[order][braket]
        # iterate over lower excitatied spaces (including gs)
        for lower_space in self.order_spaces:
            if space == lower_space:
                break
            if lower_space != "gs":
                print("not implemented")
                exit()
            lower_isr = isr[lower_space]
            for term in orders:
                i1 = lower_isr[term[1]]["bra"] * \
                    NO(operators) * lower_isr[term[2]]["ket"]
                i1 = wicks(
                    i1, keep_only_fully_contracted=True,
                    simplify_kronecker_deltas=True,
                )
                # simplify_dummies=True  # no idea if this is good here.
                # try without first
                res -= i1 * lower_isr[term[0]]["".join(bk('ket'))]
        self.precursor[order][space][braket][indices] = res
        print(f"Build {space}_({indices})^({order}) {braket}:", latex(res))

    def get_overlap(self, order, space, indices=None):
        if not indices:
            print("Indices are required when requesting precursor overlap",
                  "matrix")
            exit()
        if order not in self.overlap:
            self.overlap[order] = {}
        pass

    def get_overlap_precursor_old(self, order, **kwargs):
        """Returns precursor overlap matrix of a given order
           for all provided spaces.
           E.g. get_overlap_precursor(2, ph=True) returns the second
           order ph overlap matrix as {order: {excitation_space: x}}"""

        if len(kwargs) == 0:
            print("Need to provide the excitation space the precursor overlap",
                  "matrix should be build for. (e.g. ph=True)")
            exit()

        if order not in self.overlap:
            self.overlap[order] = {}

        to_evaluate = {}
        for excitation_space, value in kwargs.items():
            if value and isinstance(value, bool):
                if excitation_space not in self.overlap[order]:
                    to_evaluate[excitation_space] = ["bra", "ket"]
        if len(to_evaluate) > 0:
            self.__build_overlap_precursor(order, to_evaluate)
        return self.overlap[order]

    def get_overlap_precursor_pretty(self, order, **kwargs):
        """Returns the precursor overlap matrix of a given order
           for all requested spaces. (e.g. ph=True)
           The indices are cleaned up so the expression is more readable.
           """

        if len(kwargs) == 0:
            print("Need to provide the excitation space the pretty precursor",
                  "overlap matrix should be build for. (e.g. ph=True)")
            exit()
        pretty_overlap = {}
        overlap = self.get_overlap_precursor(order, **kwargs)
        for space, S in overlap.items():
            pretty_overlap[space] = self.make_pretty(S)
        return pretty_overlap

    def __build_overlap_precursor(self, order, spaces):
        orders = get_order_two(order)
        precursor = {}

        # get all relevant precursor states of all requested
        # excitation spaces
        for o in range(order + 1):
            precursor[o] = self.get_precursor(o, **spaces)

        for excitation_space in spaces:
            res = 0
            for term in orders:
                res += precursor[term[0]][excitation_space]["bra"] * \
                       precursor[term[1]][excitation_space]["ket"]
            res = wicks(res, keep_only_fully_contracted=True,
                        simplify_kronecker_deltas=True)
            #            simplify_dummies=True)
            # substitute dummies required for expression to be correct,
            # but if substituting the dummies here, the indices of the deltas
            # change and then will not match with indices at a later point.
            self.overlap[order][excitation_space] = res
            print(f"build S_{excitation_space}^({order}) = {latex(res)}\n\n")

    def make_pretty(self, expression):
        """Because it is not possible to make the indices of the
           precursor states and their overlap matrix pretty when building.
           Otherwise the ISR may not be constructed automatically."""

        return substitute_dummies(
            expression, new_indices=True, pretty_indices=pretty_indices)

    def get_S_root(self, order, **kwargs):
        """Returns S^{-0.5} of a given order for all provided spaces.
           E.g. get_S_root(2, ph=True) returns the second
           order ph S^{-0.5} as {order: {excitation_space: x}}
           """

        if len(kwargs) == 0:
            print("Need to provide the excitation space the precursor overlap",
                  "matrix should be build for. (e.g. ph=True)")
            exit()

        if order not in self.S_root:
            self.S_root[order] = {}

        spaces_to_evaluate = {}
        for excitation_space, value in kwargs.items():
            if value and isinstance(value, bool):
                if excitation_space not in self.S_root[order]:
                    spaces_to_evaluate[excitation_space] = True
        if len(spaces_to_evaluate) > 0:
            self.__build_S_root(order, spaces_to_evaluate)
        return self.S_root[order]

    def get_S_root_pretty(self, order, **kwargs):
        if len(kwargs) == 0:
            print("Need to provide the excitation space S_root_pretty",
                  "should be build for. (e.g. ph=True)")
            exit()
        pretty_S_root = {}
        S_root = self.get_S_root(order, **kwargs)
        for space, S in S_root.items():
            pretty_S_root[space] = self.make_pretty(S)
        return pretty_S_root

    def __build_S_root(self, order, spaces):
        overlap = {}
        # get all relevant Overlap matrices for all excitation spaces
        for o in range(order + 1):
            overlap[o] = self.get_overlap_precursor(o, **spaces)
        if order < 2:
            for space in spaces:
                self.S_root[order][space] = overlap[order][space]
                print(f"Build S_root_{space}^({order}) = ",
                      f"{latex(overlap[order][space])}\n\n")
        else:
            prefactors, orders = self.__expand_S_taylor(order)

            for excitation_space in spaces:
                res = 0
                for exponent, terms in orders.items():
                    for term in terms:
                        for i in range(len(term)):
                            res += prefactors[exponent] * \
                                overlap[term[i]][excitation_space]
                # res = substitute_dummies(res)
                self.S_root[order][excitation_space] = res
                print(f"Build S_root_{excitation_space}^({order}) = ",
                      f"{latex(res)}\n\n")

    def get_intermediate_states(self, order, **kwargs):
        """Input: order, ph=['ket', 'bra']"""
        if len(kwargs) == 0:
            print("It is necessary to define the excitation space",
                  "and wheter Bra or Ket should be computed when",
                  "requesting intermediate states. E.g. ph='ket'")
        convertor = {
            str: lambda braket: [braket],
            list: lambda braket: braket,
            dict: lambda braket: [
                key for key, v in braket.items() if v and isinstance(v, bool)
            ]
        }
        request = {}
        for excitation_space, braket in kwargs.items():
            conversion = convertor.get(type(braket), False)
            if not conversion:
                raise TypeError("The definition wheter the precursor Ket or",
                                "Bra is requested needs to be of type str,",
                                f"list or dict. Not of {type(braket)}")
            for ele in conversion(braket):
                if ele not in ["bra", "ket"]:
                    raise ValueError("Precursor state must be either bra",
                                     f"or ket, not {ele}")
            if len(conversion(braket)) == 0:
                print(f"{kwargs} is not a valid input for obtaining a ",
                      "Bra/Ket Precursor state. Use 'space'='ket',",
                      "'space'=['ket', 'bra'] or 'space'=dict('bra': True).")
                exit()
            request[excitation_space] = conversion(braket)
        to_evaluate = {}
        if not self.isr.get(order, False):
            self.isr[order] = {}
        # need to test code below when build intermediates work
        for excitation_space, braket in request.items():
            if not self.isr[order].get(excitation_space, False):
                self.isr[order][excitation_space] = {}
            to_evaluate[excitation_space] = \
                [bk for bk in braket if not
                    self.isr[order][excitation_space].get(bk, False)]
        if len(to_evaluate) > 0:
            self.__build_intermediate_states(order, to_evaluate)
        return self.isr[order]

    def __build_intermediate_states(self, order, spaces):
        # spaces = {space: ["bra", "ket"]}
        orders = get_order_two(order)

        # import all S^{-0.5} and precursor states for all spaces requested
        precursor = {}
        s_root = {}
        # convert bra/ket to True since get_S_root does not know bra or ket
        s_root_spaces = {}
        for space in spaces:
            s_root_spaces[space] = True
        for o in range(order + 1):
            s_root[o] = self.get_S_root(o, **s_root_spaces)
            # since all precursor states have already been constructed for the
            # overlap matrix they are already available
            precursor[o] = self.precursor[o]

        for space, braket in spaces.items():
            if "ket" in braket:
                res = 0
                for term in orders:
                    res += s_root[term[0]][space] * \
                         precursor[term[1]][space]["ket"]
                res = substitute_dummies(res)
                res = evaluate_deltas(res)
                res = substitute_dummies(res)
                self.isr[order][space]["ket"] = res
                print(f"ISR_{space}^({order}) ket = ",
                      f"{latex(self.make_pretty(res))}\n\n")
            if "bra" in braket:
                res = 0
                for term in orders:
                    res += s_root[term[0]][space] * \
                         precursor[term[1]][space]["bra"]
                res = evaluate_deltas(res)
                self.isr[order][space]["bra"] = res
                print(f"ISR_{space}^({order}) bra = {latex(res)}\n\n")

    def __expand_S_taylor(self, order):
        """Computes the Taylor expansion of 'S^{-0.5} = (1 + x)^{-0.5} with
           'x = S(2) + S(3) + O(4)' to a given order in perturbation theory.
           Returns two dicts.
           The first one contains the prefactors of the Taylor expansion.
           The second one contains the orders of S that contribute to the
           n-th order term. (like (4,) and (2,2) for fourth order)
           In both dicts the exponent of x in the Taylor expansion is used
           as key."""

        x = symbols('x')
        diffs = {0: 1}
        f = (1 + x) ** -0.5
        intermediate = f
        for o in range(1, int(order / 2) + 1):
            intermediate = diff(intermediate, x)
            diffs[o] = nsimplify(intermediate.subs(x, 0) * 1 / factorial(o))
        print("prefactors of taylor expansion: ", diffs)
        orders = gen_order_S(order)
        print("orders of overlap matrix: ", orders)
        return (diffs, orders)


def gen_order_S(order):
    """expands 'S = 1 + x' in a Taylor series with 'x = S(2) + S(3) + O(4)' to
       the given order.
       Returns a dict that contains the exponents of x as key and a list of
       tuples of the orders of S as values. {exponent: [(o1, o2, ...), ...]}
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
# mp.get_energy(order=1)
# mp.get_psi(2, bra=True)
isr = intermediate_states(mp)
a = isr.get_precursor(2, "ph", "ket", indices="jb")
# a = isr.get_overlap_precursor(2, ph=True)  # , pphh=True)
# a = isr.get_S_root(2, ph=True)
