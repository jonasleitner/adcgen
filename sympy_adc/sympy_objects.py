from sympy.physics.secondquant import TensorSymbol, \
    _sort_anticommuting_fermions, ViolationOfPauliPrinciple
from sympy.core.function import Function
from sympy.core.logic import fuzzy_not
from sympy import sympify, Tuple, Symbol, S
from .misc import Inputerror
from .indices import Index, Indices


class AntiSymmetricTensor(TensorSymbol):
    """Based on the AntiSymmetricTensor from sympy.physics.secondquant.
       Differences are:
           - the sorting key for the sorting of the indices.
             Here indices are sorted canonical.
           - Additional support for bra/ket symmetry/antisymmetry.
        """

    def __new__(cls, symbol: str, upper: tuple[Index], lower: tuple[Index],
                bra_ket_sym: int = 0) -> TensorSymbol:
        # sort the upper and lower indices
        try:
            upper, sign_u = _sort_anticommuting_fermions(
                upper, key=cls._sort_canonical
            )
            lower, sign_l = _sort_anticommuting_fermions(
                lower, key=cls._sort_canonical
            )
        except ViolationOfPauliPrinciple:
            return S.Zero
        # additionally account for the bra ket symmetry
        # add the check for Dummy indices for subs to work correctly
        bra_ket_sym = sympify(bra_ket_sym)
        if bra_ket_sym is not S.Zero and all(isinstance(s, Index) for s
                                             in upper+lower):
            if bra_ket_sym not in [S.One, S.NegativeOne]:
                raise Inputerror("Invalid bra ket symmetry given "
                                 f"{bra_ket_sym}. Valid are 0, 1 or -1.")
            if len(upper) != len(lower):
                raise NotImplementedError("Bra Ket symmetry only implemented "
                                          "for tensors with an equal amount "
                                          "of upper and lower indices.")
            space_u = "".join([s.space[0] for s in upper])
            space_l = "".join([s.space[0] for s in lower])
            if space_l < space_u:  # space with more occ should be the lowest
                upper, lower = lower, upper  # swap
                if bra_ket_sym is S.NegativeOne:  # add another -1
                    sign_u += 1
            # diagonal block: compare the names of the indices
            elif space_l == space_u:
                lower_names = [(int(s.name[1:]) if s.name[1:] else 0,
                                s.name[0]) for s in lower]
                upper_names = [(int(s.name[1:]) if s.name[1:] else 0,
                                s.name[0]) for s in upper]
                if lower_names < upper_names:
                    upper, lower = lower, upper  # swap
                    if bra_ket_sym is S.NegativeOne:  # add another -1
                        sign_u += 1
        # import all quantities to sympy
        symbol = sympify(symbol)
        upper, lower = Tuple(*upper), Tuple(*lower)

        # attach -1 if necessary
        if (sign_u + sign_l) % 2:
            return - TensorSymbol.__new__(cls, symbol, upper, lower,
                                          bra_ket_sym)
        else:
            return TensorSymbol.__new__(cls, symbol, upper, lower, bra_ket_sym)

    @classmethod
    def _sort_canonical(cls, idx):
        if isinstance(idx, Index):
            # also add the hash here for wicks, where multiple i are around
            return (idx.space[0],
                    idx.spin,
                    int(idx.name[1:]) if idx.name[1:] else 0,
                    idx.name[0],
                    hash(idx))
        else:  # necessary for subs to work correctly with simultaneous=True
            return ('', 0, str(idx), hash(idx))

    def _latex(self, printer) -> str:
        return "{%s^{%s}_{%s}}" % (
            self.symbol,
            "".join([i._latex(printer) for i in self.args[1]]),
            "".join([i._latex(printer) for i in self.args[2]])
        )

    @property
    def symbol(self) -> Symbol:
        """Returns the symbol of the tensor."""
        return self.args[0]

    @property
    def upper(self) -> Tuple:
        """Returns the upper indices of the tensor."""
        return self.args[1]

    @property
    def lower(self) -> Tuple:
        """Returns the lower indices of the tensor."""
        return self.args[2]

    def __str__(self):
        return "%s(%s,%s)" % self.args[:3]

    @property
    def bra_ket_sym(self):
        return self.args[3]

    def add_bra_ket_sym(self, bra_ket_sym: int):
        """Adds a bra ket symmetry to the tensor if none has been set yet.
           Valid bra ket symmetries are 0, 1 and -1."""

        if bra_ket_sym and self.bra_ket_sym is S.Zero:
            return AntiSymmetricTensor(self.symbol, self.upper, self.lower,
                                       bra_ket_sym)
        elif not bra_ket_sym:
            return self
        else:
            raise Inputerror("bra ket symmetry already set. The original "
                             "indices are no longer available. Can not apply "
                             "any other bra ket sym.")


class NonSymmetricTensor(TensorSymbol):
    """Used to represent tensors that do not have any symmetry."""

    def __new__(cls, symbol: str, indices: tuple[Index]) -> TensorSymbol:
        symbol = sympify(symbol)
        indices = Tuple(*indices)
        return TensorSymbol.__new__(cls, symbol, indices)

    def _latex(self, printer) -> str:
        return "{%s_{%s}}" % (self.symbol, "".join([i._latex(printer)
                                                    for i in self.indices]))

    @property
    def symbol(self) -> Symbol:
        return self.args[0]

    @property
    def indices(self) -> Tuple:
        return self.args[1]

    def __str__(self):
        return "%s%s" % self.args


class KroneckerDelta(Function):
    @classmethod
    def eval(cls, i: Index, j: Index):
        """Evaluates the KroneckerDelta.
           Adapted from sympy to also cover Spin."""

        diff = i - j
        if diff.is_zero or fuzzy_not(diff.is_zero):
            return S.One

        spi, spj = i.space[0], j.space[0]
        if spi != "g" and spj != "g" and spi != spj:  # delta_ov / delta_vo
            return S.Zero
        spi, spj = i.spin, j.spin
        if spi and spj and spi != spj:  # delta_ab / delta_ba
            return S.Zero
        # sort the indices of the delta
        if i != min(i, j, key=cls._sort_canonical):
            return cls(j, i)

    @classmethod
    def _sort_canonical(cls, idx):
        if isinstance(idx, Index):
            # also add the hash here for wicks, where multiple i are around
            return (idx.space[0],
                    idx.spin,
                    int(idx.name[1:]) if idx.name[1:] else 0,
                    idx.name[0],
                    hash(idx))
        else:  # necessary for subs to work correctly with simultaneous=True
            return ('', 0, str(idx), hash(idx))

    def _latex(self, printer):
        return (
            "\\delta_{" + " ".join(s._latex(printer) for s in self.args) + "}"
        )

    @property
    def preferred_and_killable(self) -> tuple[Index]:
        """Returns the index which is preferred to keep in the final
           expression."""
        i, j = self.args[0], self.args[1]
        space1, spin1 = i.space[0], i.spin
        space2, spin2 = j.space[0], j.spin

        if spin1 == spin2:  # nn / aa / bb  -> equal information
            if space1 == space2 or space2 == "g":  # oo / vv / gg / og / vg
                return (i, j)
            else:  # go / gv
                return (j, i)
        elif spin2:  # na / nb  -> 2 holds more information
            if space1 == space2 or space1 == "g":  # oo / vv / gg / go / gv
                return (j, i)
            else:  # og / vg  -> 1 holds more space information
                # create a new generic index that combines space and spin
                # information of both indices
                n_ov = {f"n_{space1}_{spin2}": 1}
                pref = Indices().get_generic_indices(**n_ov).popitem()[1][0]
                return (pref, i, j)
        else:  # an / bn  -> 1 holds more information
            if space1 == space2 or space2 == "g":  # oo / vv / gg / og / vg
                return (i, j)
            else:  # go / gv  -> 2 holds more space information
                # create a new generic index that combines space and spin
                # information of both indices
                n_ov = {f"n_{space2}_{spin1}": 1}
                pref = Indices().get_generic_indices(**n_ov).popitem()[1][0]
                return (pref, i, j)

    @property
    def indices_contain_equal_information(self):
        return self.args[0].space == self.args[1].space and \
            self.args[0].spin == self.args[1].spin


class SingleSymmetryTensor(TensorSymbol):
    def __new__(cls, symbol: str, indices: tuple[Index],
                perms: list[tuple[int]], factor: int) -> TensorSymbol:
        from itertools import chain

        # ensure that we have no intersecting permutations, i.e., each
        # index occurs only once
        idx = list(chain.from_iterable(perms))
        if len(idx) != len(set(idx)):
            raise NotImplementedError("SpecialSymTensor not implemented for"
                                      f"intersecting permutations {perms}.")
        factor = sympify(factor)
        if factor not in [S.One, S.NegativeOne]:
            raise Inputerror(f"Invalid factor {factor}. Valid are 1 and -1.")

        # each permutation can be applied independently of the others
        permuted = list(indices)
        apply = []
        min_apply = None
        min_not_apply = None
        for perm in perms:
            i, j = sorted(perm)
            p, q = indices[i], indices[j]  # p occurs before q
            if factor is S.NegativeOne and p == q:
                return S.Zero
            p_val, q_val = cls._sort_canonical(p), cls._sort_canonical(q)
            if q_val < p_val:
                apply.append(True)
                if min_apply is None or q_val < min_apply:
                    min_apply = q_val
            else:
                if min_not_apply is None or q_val < min_not_apply:
                    min_not_apply = p_val
            permuted[i], permuted[j] = q, p
        attach_minus = False
        if len(apply) == len(perms):
            indices = permuted
            attach_minus = factor is S.NegativeOne
        elif len(apply) >= len(perms) / 2 and min_apply < min_not_apply:
            indices = permuted
            attach_minus = factor is S.NegativeOne

        symbol = sympify(symbol)
        indices = Tuple(*indices)
        perms = Tuple(*perms)

        if attach_minus:
            return - TensorSymbol.__new__(cls, symbol, indices, perms, factor)
        else:
            return TensorSymbol.__new__(cls, symbol, indices, perms, factor)

    @classmethod
    def _sort_canonical(cls, idx):
        if isinstance(idx, Index):
            # also add the hash here for wicks, where multiple i are around
            return (idx.space[0],
                    idx.spin,
                    int(idx.name[1:]) if idx.name[1:] else 0,
                    idx.name[0],
                    hash(idx))
        else:  # necessary for subs to work correctly with simultaneous=True
            return ('', 0, str(idx), hash(idx))

    def _latex(self, printer) -> str:
        return "{%s_{%s}}" % (self.symbol, "".join([i._latex(printer)
                                                    for i in self.indices]))

    @property
    def symbol(self) -> Symbol:
        return self.args[0]

    @property
    def indices(self) -> Tuple:
        return self.args[1]

    @property
    def sym(self) -> Tuple:
        return Tuple(*self.args[2:])

    def __str__(self):
        return "%s%s" % self.args[:2]
