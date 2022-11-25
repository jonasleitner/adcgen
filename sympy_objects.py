from sympy.physics.secondquant import TensorSymbol
from sympy import sympify, Tuple, Symbol


class NonSymmetricTensor(TensorSymbol):
    """Used to represent tensors that do not have any symmetry."""

    def __new__(cls, symbol, indices) -> TensorSymbol:
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
