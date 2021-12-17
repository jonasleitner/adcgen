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
        # {order: {'excitation_space': {"ket/bra": x}}}
        self.precursor = {}
        # {order: {'excitation_space': x}}
        self.overlap = {}
        self.S_root = {}
        # {order: {'excitation_space': {"ket/bra": x}}}
        self.isr = {}

    def get_precursor(self, order, **kwargs):
        """Returns the Precursor states of a given order as dict.
           Requires to specify whether Bra, Ket or both of a given
           excitation space are desired.
           E.g. get_precursor(2, ph=["bra", "ket"]) returns the second
           order ph precursor states as {"ph": {"ket": x, "bra": y}}.
           """

        if len(kwargs) == 0:
            print("Need to provide the excitation space and wheter",
                  "bra and/or ket precursor state are requested.",
                  "(e.g. ph='bra')")
            exit()
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
            try:
                callback = getattr(
                    self, "_build_precursor_" + excitation_space
                )
            except AttributeError:
                print("Precursor states for excitation space",
                      f"{excitation_space} not implemented.")
                exit()
            if len(conversion(braket)) == 0:
                print(f"{kwargs} is not a valid input for obtaining a ",
                      "Bra/Ket Precursor state. Use 'space'='ket',",
                      "'space'=['ket', 'bra'] or 'space'=dict('bra': True).")
                exit()
            request[excitation_space] = (callback, conversion(braket))

        if not self.precursor.get(order, False):
            self.precursor[order] = {}

        for excitation_space, value in request.items():
            callback = value[0]
            braket = value[1]
            # invoke space for indices if not created before
            if excitation_space not in self.invoked_spaces:
                self.indices.create_space(excitation_space)

            if excitation_space not in self.precursor[order]:
                self.precursor[order][excitation_space] = {}
                callback(order, *braket)
            else:
                [callback(order, bk) for bk in braket if
                    bk not in self.precursor[order][excitation_space]]
        return self.precursor[order]

    def _build_precursor_ph(self, order, *args):
        # sort the args (bra, ket) so that bra get the lower indices.
        sorted_args = list(args)
        sorted_args.sort()

        # generate indices and build all the ground state wavefunctions.
        mp = {}
        idx = self.indices.get_indices("ph", occ=2, virt=2)

        for braket in ["bra", "ket"]:
            bk = {braket: True}
            for o in range(order + 1):
                mp[o] = self.gs.get_psi(o, **bk)

        # get all possible combinations for 0><0|s|0> of order n
        orders = get_orders_three(order)

        # <S# = <S* - <S*|0><0
        if "bra" in sorted_args:
            singles = NO(Fd(idx["occ"][0]) * F(idx["virt"][0]))

            res = mp[order]["bra"] * singles
            for term in orders:
                intermediate = mp[term[0]]["bra"] * \
                    singles * mp[term[1]]["ket"]
                intermediate = wicks(
                    intermediate, keep_only_fully_contracted=True,
                    simplify_kronecker_deltas=True,
                    simplify_dummies=True
                )
                intermediate *= mp[term[2]]["bra"]
                res -= intermediate
            self.precursor[order]["ph"]["bra"] = res
            print(f"<S#_{order} = {latex(res)}\n\n")

        # S#> = S> - 0><0|S>
        if "ket" in sorted_args:
            singles = NO(Fd(idx["virt"][1]) * F(idx["occ"][1]))

            res = singles * mp[order]["ket"]
            for term in orders:
                intermediate = mp[term[1]]["bra"] * \
                                  singles * mp[term[2]]["ket"]
                intermediate = wicks(
                    intermediate, keep_only_fully_contracted=True,
                    simplify_kronecker_deltas=True,
                    simplify_dummies=True
                )
                intermediate *= mp[term[0]]["ket"]
                res -= intermediate
            self.precursor[order]["ph"]["ket"] = res
            print(f"S#>_{order} = {latex(res)}\n\n")

    def get_overlap_precursor(self, order, **kwargs):
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
# a = isr.get_precursor(2, ph={"bra": True, "ket": True})
# a = isr.get_overlap_precursor(2, ph=True)  # , pphh=True)
# a = isr.get_S_root(2, ph=True)
# a = isr.get_intermediate_states(1, ph=["ket", "bra"])
b = isr.get_intermediate_states(2, ph=["ket"])
# c = isr.get_intermediate_states(2, ph=["ket"])
