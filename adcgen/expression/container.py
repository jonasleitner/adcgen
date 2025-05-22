from collections.abc import Iterable
from typing import Any, TYPE_CHECKING

from sympy import Expr, latex, S, sympify

from ..indices import Index, order_substitutions
from ..sympy_objects import AntiSymmetricTensor

# imports only required for type checking (avoid circular imports)
if TYPE_CHECKING:
    from .expr_container import ExprContainer


class Container:
    """
    Base class for all container classes that wrap a native
    sympy object.

    Parameters
    ----------
    inner: Expr | Container | Any
        The algebraic expression to wrap, e.g., a sympy.Add or sympy.Mul object
    target_idx: Iterable[Index] | None, optional
        Target indices of the expression. By default the Einstein sum
        convention will be used to identify target and contracted indices,
        which is not always sufficient.
    """

    def __init__(self, inner: "Expr | Container | Any",
                 target_idx: Iterable[Index] | None = None) -> None:
        # possibly extract or import the expression to wrap
        if isinstance(inner, Container):
            inner = inner.inner
        if not isinstance(inner, Expr):
            inner = sympify(inner)
            assert isinstance(inner, Expr)
        self._inner: Expr = inner
        # set the assumptions: the target indices
        if target_idx is not None and not isinstance(target_idx, tuple):
            target_idx = tuple(target_idx)
        self._target_idx: tuple[Index, ...] | None = target_idx

    def __str__(self) -> str:
        return latex(self.inner)

    @property
    def assumptions(self) -> dict[str, Any]:
        return {
            "target_idx": self.provided_target_idx,
        }

    @property
    def provided_target_idx(self) -> tuple[Index, ...] | None:
        return self._target_idx

    @property
    def inner(self) -> Expr:
        return self._inner

    @property
    def braket_sym_tensors(self) -> tuple[str, ...]:
        """
        Returns the names of all tensors with bra-ket-symmetry, i.e.,
        all tensors with d^{pq}_{rs} = d^{rs}_{pq}.
        """
        return tuple(sorted(set(
            tensor.name for tensor in self.inner.atoms(AntiSymmetricTensor)
            if tensor.bra_ket_sym is S.One
        )))

    @property
    def braket_antisym_tensors(self) -> tuple[str, ...]:
        """
        Returns the names of all tensors with bra-ket-antisymmetry, i.e.,
        all tensors with d^{pq}_{rs} = -d^{rs}_{pq}.
        """
        return tuple(sorted(set(
            tensor.name for tensor in self.inner.atoms(AntiSymmetricTensor)
            if tensor.bra_ket_sym is S.NegativeOne
        )))

    def permute(self, *perms: tuple[Index, Index]) -> "ExprContainer":
        """
        Permute indices by applying permutation operators P_pq.

        Parameters
        ----------
        *perms : tuple[Index, Index]
            Permutations to apply to the wrapped object. Permutations are
            applied one after another in the order they are provided.
        """
        sub = {}
        for p, q in perms:
            addition = {p: q, q: p}
            for old, new in sub.items():
                if new is p:
                    sub[old] = q
                    del addition[p]
                elif new is q:
                    sub[old] = p
                    del addition[q]
            if addition:
                sub.update(addition)
        return self.subs(order_substitutions(sub))

    ################################
    # Forwards some calls to inner #
    ################################
    def expand(self) -> "ExprContainer":
        """
        Forwards the expand call to inner and wraps the result in a new
        Container
        """
        from .expr_container import ExprContainer
        return ExprContainer(inner=self.inner.expand(), **self.assumptions)

    def doit(self, *args, **kwargs) -> "ExprContainer":
        """
        Forwards the doit call to inner and wraps the result in a new Container
        """
        from .expr_container import ExprContainer
        return ExprContainer(
            inner=self.inner.doit(*args, **kwargs), **self.assumptions
        )

    def subs(self, *args, **kwargs) -> "ExprContainer":
        """
        Forwards the subs call to inner and wraps the result in a new Container
        """
        from .expr_container import ExprContainer
        return ExprContainer(
            inner=self.inner.subs(*args, **kwargs), **self.assumptions
        )

    #############
    # Operators #
    #############
    def __add__(self, other: Any) -> "ExprContainer":
        from .expr_container import ExprContainer

        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.inner
        return ExprContainer(self.inner + other, **self.assumptions)

    def __iadd__(self, other: Any) -> "ExprContainer":
        return self.__add__(other)

    def __radd__(self, other: Any) -> "ExprContainer":
        from .expr_container import ExprContainer
        # other: some sympy stuff or some number
        return ExprContainer(other + self.inner, **self.assumptions)

    def __sub__(self, other: Any) -> "ExprContainer":
        from .expr_container import ExprContainer

        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.inner
        return ExprContainer(self.inner - other, **self.assumptions)

    def __isub__(self, other: Any) -> "ExprContainer":
        return self.__sub__(other)

    def __rsub__(self, other: Any) -> "ExprContainer":
        from .expr_container import ExprContainer
        # other: some sympy stuff or some number
        return ExprContainer(other - self.inner, **self.assumptions)

    def __mul__(self, other: Any) -> "ExprContainer":
        from .expr_container import ExprContainer

        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.inner
        return ExprContainer(self.inner * other, **self.assumptions)

    def __imul__(self, other: Any) -> "ExprContainer":
        return self.__mul__(other)

    def __rmul__(self, other: Any) -> "ExprContainer":
        from .expr_container import ExprContainer
        # other: some sympy stuff or some number
        return ExprContainer(other * self.inner, **self.assumptions)

    def __truediv__(self, other: Any) -> "ExprContainer":
        from .expr_container import ExprContainer

        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.inner
        return ExprContainer(self.inner / other, **self.assumptions)

    def __itruediv__(self, other: Any) -> "ExprContainer":
        return self.__truediv__(other)

    def __rtruediv__(self, other: Any) -> "ExprContainer":
        from .expr_container import ExprContainer
        # other: some sympy stuff or some number
        return ExprContainer(other / self.inner, **self.assumptions)
