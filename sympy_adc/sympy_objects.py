from sympy.physics.secondquant import TensorSymbol, \
    _sort_anticommuting_fermions, ViolationOfPauliPrinciple
from sympy import sympify, Tuple, Symbol, Dummy, S
from .misc import Inputerror
from .indices import index_space


class AntiSymmetricTensor(TensorSymbol):
    """Based on the AntiSymmetricTensor from sympy.physics.secondquant.
       Differences are:
           - the sorting key for the sorting of the indices.
             Here indices are sorted canonical.
           - Additional support for bra/ket symmetry/antisymmetry.
        """

    def __new__(cls, symbol: str, upper: tuple[Dummy], lower: tuple[Dummy],
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
        if bra_ket_sym is not S.Zero and all(isinstance(s, Dummy) for s
                                             in upper+lower):
            if bra_ket_sym not in [S.One, S.NegativeOne]:
                raise Inputerror("Invalid bra ket symmetry given "
                                 f"{bra_ket_sym}. Valid are 0, 1 or -1.")
            if len(upper) != len(lower):
                raise NotImplementedError("Bra Ket symmetry only implemented "
                                          "for tensors with an equal amount "
                                          "of upper and lower indices.")
            space_u = "".join([index_space(s.name)[0] for s in upper])
            space_l = "".join([index_space(s.name)[0] for s in lower])
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
        if isinstance(idx, Dummy):
            # also add the hash here for wicks, where multiple i are around
            return (index_space(idx.name)[0],
                    int(idx.name[1:]) if idx.name[1:] else 0,
                    idx.name[0],
                    hash(idx))
        else:  # necessary for subs to work correctly with simultaneous=True
            return ('', 0, str(idx), hash(idx))

    def _latex(self, printer) -> str:
        return "{%s^{%s}_{%s}}" % (
            self.symbol,
            "".join([i.name for i in self.args[1]]),
            "".join([i.name for i in self.args[2]])
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

    def __new__(cls, symbol: str, indices: tuple[Dummy]) -> TensorSymbol:
        symbol = sympify(symbol)
        indices = Tuple(*indices)
        return TensorSymbol.__new__(cls, symbol, indices)

    def _latex(self, printer) -> str:
        return "{%s_{%s}}" % (self.symbol,
                              "".join([i.name for i in self.indices]))

    @property
    def symbol(self) -> Symbol:
        return self.args[0]

    @property
    def indices(self) -> Tuple:
        return self.args[1]

    def __str__(self):
        return "%s%s" % self.args


class Index(Dummy):
    """Class to represent Indices. Inherits it's behaviour from the sympy
       'Dummy' class, i.e.,
        Index("x") == Index("x")
        will evaluate to False.
        Additional functionality:
         - assigning a spin to the variable via the spin keyword ('a'/'b')
    """
    def __new__(cls, name: str = None, dummy_index=None, spin: str = None,
                **assumptions):
        if spin is None:
            return super().__new__(cls, name, dummy_index, **assumptions)
        elif spin == "a":
            return super().__new__(cls, name, dummy_index, alpha=True,
                                   **assumptions)
        elif spin == "b":
            return super().__new__(cls, name, dummy_index, beta=True,
                                   **assumptions)
        else:
            raise Inputerror(f"Invalid spin {spin}. Valid values are 'a' and "
                             "'b'.")

    @property
    def spin(self) -> None | str:
        if self.assumptions0.get("alpha"):
            return "a"
        elif self.assumptions0.get("beta"):
            return "b"

    @property
    def space(self) -> str:
        if self.assumptions0.get("below_fermi"):
            return "occ"
        elif self.assumptions0.get("above_fermi"):
            return "virt"
        else:
            return "general"


class SingleSymmetryTensor(TensorSymbol):
    def __new__(cls, symbol: str, indices: tuple[Dummy],
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
        if isinstance(idx, Dummy):
            # also add the hash here for wicks, where multiple i are around
            return (index_space(idx.name)[0],
                    int(idx.name[1:]) if idx.name[1:] else 0,
                    idx.name[0],
                    hash(idx))
        else:  # necessary for subs to work correctly with simultaneous=True
            return ('', 0, str(idx), hash(idx))

    def _latex(self, printer) -> str:
        return "{%s_{%s}}" % (self.symbol,
                              "".join([i.name for i in self.indices]))

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
