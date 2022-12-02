from sympy.physics.secondquant import TensorSymbol, \
    _sort_anticommuting_fermions, ViolationOfPauliPrinciple
from sympy import sympify, Tuple, Symbol, Dummy, S
from .misc import Inputerror
from .indices import sort_idx_canonical, index_space


class AntiSymmetricTensor(TensorSymbol):
    """Based on the AntiSymmetricTensor from sympy.physics.secondquant.
       Differences are:
           - the sorting key for the sorting of the indices.
             Here indices are sorted canonical.
           - Additional support for bra/ket symmetry/antisymmetry.
        """

    def __new__(cls, symbol: str, upper: tuple[Dummy], lower: tuple[Dummy],
                bra_ket_sym: str = None) -> TensorSymbol:
        # sort the upper and lower indices
        try:
            upper, sign_u = _sort_anticommuting_fermions(
                upper, key=sort_idx_canonical
            )
            lower, sign_l = _sort_anticommuting_fermions(
                lower, key=sort_idx_canonical
            )
        except ViolationOfPauliPrinciple:
            return S.Zero
        # additionally account for the bra ket symmetry
        # add the check for Dummy indices for subs to work correctly
        if bra_ket_sym is not None and all(isinstance(s, Dummy) for s in
                                           upper+lower):
            if bra_ket_sym not in ['+', '-']:
                raise Inputerror("Invalid bra ket symmetry given "
                                 f"{bra_ket_sym}. Valid are '+' or '-'.")
            if len(upper) != len(lower):
                raise NotImplementedError("Bra Ket symmetry only implemented "
                                          "for tensors with an equal amount "
                                          "of upper and lower indices.")
            space_u = "".join([index_space(s.name)[0] for s in upper])
            space_l = "".join([index_space(s.name)[0] for s in lower])
            if space_l < space_u:  # space with more occ should be the lowest
                upper, lower = lower, upper  # swap
                if bra_ket_sym == '-':  # add another -1
                    sign_u += 1
            # if diagonal block: compare the lowest index of each space
            elif space_l == space_u and (sort_idx_canonical(lower[0]) <
                                         sort_idx_canonical(upper[0])):
                upper, lower = lower, upper  # swap
                if bra_ket_sym == '-':  # add another -1
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

    def add_bra_ket_sym(self, bra_ket_sym: str):
        """Adds a bra ket symmetry to the tensor if none has been set yet.
           Valid bra ket symmetries are '+' and '-'."""

        if self.bra_ket_sym is not None:
            raise Inputerror("bra ket symmetry already set. The original "
                             "indices are no longer available. Can not apply "
                             "another bra ket sym.")
        return AntiSymmetricTensor(self.symbol, self.upper, self.lower,
                                   bra_ket_sym)


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
