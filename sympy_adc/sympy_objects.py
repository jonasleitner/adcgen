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
            # if diagonal block: compare the lowest index of each space
            elif space_l == space_u and (cls._sort_canonical(lower[0]) <
                                         cls._sort_canonical(upper[0])):
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
            return ('', hash(idx), str(idx), 0)

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
