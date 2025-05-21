from collections.abc import Sequence

from sympy.physics.secondquant import (
    _sort_anticommuting_fermions, ViolationOfPauliPrinciple
)
from sympy.core.logic import fuzzy_not
from sympy.core.function import DefinedFunction
from sympy import sympify, Tuple, Symbol, S, Number, Expr

from .misc import Inputerror
from .indices import Index, _is_index_tuple, sort_idx_canonical


class SymbolicTensor(Expr):
    """Base class for symbolic tensors."""

    is_commutative = True

    @property
    def symbol(self) -> Symbol:
        """Returns the symbol of the tensor."""
        symbol = self.args[0]
        assert isinstance(symbol, Symbol)
        return symbol

    @property
    def name(self) -> str:
        """Returns the name of the tensor."""
        return self.symbol.name

    @property
    def idx(self) -> tuple[Index, ...]:
        """Returns all indices of the tensor."""
        raise NotImplementedError("'idx' not implemented on "
                                  f"{self.__class__.__name__}")


class AntiSymmetricTensor(SymbolicTensor):
    """
    Represents antisymmetric tensors
    d^{pq}_{rs} = - d^{qp}_{rs} = - d^{pq}_{sr} = d^{qp}_{sr}.
    Based on the implementation in 'sympy.physics.secondquant'.

    Parameters
    ----------
    name : str | Symbol
        The name of the tensor.
    upper : Sequence[Index] | Tuple
        The upper indices of the tensor.
    lower : Sequence[Index] | Tuple
        The lower indices of the tensor.
    bra_ket_sym : int | Number, optional
        The bra-ket symmetry of the tensor:
        - 0 no bra-ket-symmetry (d^{i}_{j} != d^{j}_{i})
        - 1 bra-ket symmetry (d^{i}_{j} = d^{j}_{i})
        - -1 bra-ket antisymmetry (d^{i}_{j} = - d^{j}_{i})
        (default: 0)
    """

    def __new__(cls, name: str | Symbol, upper: Sequence[Index] | Tuple,
                lower: Sequence[Index] | Tuple, bra_ket_sym: int | Number = 0):
        # sort the upper and lower indices
        try:
            upper_sorted, sign_u = _sort_anticommuting_fermions(
                upper, key=sort_idx_canonical
            )
            lower_sorted, sign_l = _sort_anticommuting_fermions(
                lower, key=sort_idx_canonical
            )
        except ViolationOfPauliPrinciple:
            return S.Zero
        # additionally account for the bra ket symmetry
        # add the Index check for subs to work correctly
        bra_ket_sym_imported = sympify(bra_ket_sym)
        if bra_ket_sym_imported is not S.Zero and \
                all(isinstance(s, Index) for s in upper_sorted+lower_sorted):
            if bra_ket_sym_imported not in [S.One, S.NegativeOne]:
                raise Inputerror("Invalid bra ket symmetry given "
                                 f"{bra_ket_sym}. Valid are 0, 1 or -1.")
            if cls._need_bra_ket_swap(upper_sorted, lower_sorted):
                upper_sorted, lower_sorted = lower_sorted, upper_sorted  # swap
                if bra_ket_sym_imported is S.NegativeOne:  # add another -1
                    sign_u += 1
        # import all quantities
        name_imported = sympify(name)
        upper_imported = Tuple(*upper_sorted)
        lower_imported = Tuple(*lower_sorted)
        # attach -1 if necessary
        if (sign_u + sign_l) % 2:
            return - super().__new__(
                cls, name_imported, upper_imported, lower_imported,
                bra_ket_sym_imported
            )
        else:
            return super().__new__(
                cls, name_imported, upper_imported, lower_imported,
                bra_ket_sym_imported
            )

    @classmethod
    def _need_bra_ket_swap(cls, upper: Sequence, lower: Sequence) -> bool:
        if len(upper) != len(lower):
            raise NotImplementedError("Bra Ket symmetry only implemented "
                                      "for tensors with an equal amount "
                                      "of upper and lower indices.")
        # Build the sort key for each index and collect the first, second, ...
        # entries of the keys
        # -> Compare each component of the sort keys individually and abort
        # if it is clear, that we need or don't need to swap
        # Assumes that upper indices should have the smaller keys.
        upper_sort_keys = (sort_idx_canonical(s) for s in upper)
        lower_sort_keys = (sort_idx_canonical(s) for s in lower)
        for upper_keys, lower_keys in \
                zip(zip(*upper_sort_keys), zip(*lower_sort_keys)):
            if lower_keys < upper_keys:
                return True
            elif upper_keys < lower_keys:
                return False
        return False

    def _latex(self, printer) -> str:
        upper = self.upper.args
        lower = self.lower.args
        assert _is_index_tuple(upper) and _is_index_tuple(lower)
        return "{%s^{%s}_{%s}}" % (
            self.symbol,
            "".join(i._latex(printer) for i in upper),
            "".join(i._latex(printer) for i in lower)
        )

    def __str__(self):
        return f"{self.symbol}({self.upper},{self.lower})"

    @property
    def upper(self) -> Tuple:
        """Returns the upper indices of the tensor."""
        upper = self.args[1]
        assert isinstance(upper, Tuple)
        return upper

    @property
    def lower(self) -> Tuple:
        """Returns the lower indices of the tensor."""
        lower = self.args[2]
        assert isinstance(lower, Tuple)
        return lower

    @property
    def bra_ket_sym(self) -> Number:
        """Returns the bra-ket symmetry of the tensor."""
        braketsym = self.args[3]
        assert isinstance(braketsym, Number)
        return braketsym

    def add_bra_ket_sym(self, bra_ket_sym: int | Number
                        ) -> 'AntiSymmetricTensor':
        """
        Adds bra-ket symmetry to the tensor if none has been set yet.

        Parameters
        ----------
        bra_ket_sym : int
            The bra-ket symmetry to set (0, 1 and -1 are valid.)
        """

        if bra_ket_sym == self.bra_ket_sym:
            return self
        elif self.bra_ket_sym is S.Zero:
            return self.__class__(self.symbol, self.upper, self.lower,
                                  bra_ket_sym)
        else:
            raise Inputerror("bra ket symmetry already set. The original "
                             "indices are no longer available. Can not apply "
                             "any other bra ket sym.")

    @property
    def idx(self) -> tuple[Index, ...]:
        """
        Returns all indices of the tensor. The upper indices are listed before
        the lower indices.
        """
        idx = self.upper.args + self.lower.args
        assert _is_index_tuple(idx)
        return idx


class Amplitude(AntiSymmetricTensor):
    """
    Represents antisymmetric Amplitudes.
    """

    @property
    def idx(self) -> tuple[Index, ...]:
        """
        Returns all indices of the amplitude. The lower indices are
        listed before the upper indices.
        """
        idx = self.lower.args + self.upper.args
        assert _is_index_tuple(idx)
        return idx


class SymmetricTensor(AntiSymmetricTensor):
    """
    Represents symmetric tensors
    d^{pq}_{rs} = d^{qp}_{rs} = d^{pq}_{sr} = d^{qp}_{sr}.

    Parameters
    ----------
    name : str | Symbol
        The name of the tensor.
    upper : Sequence[Index]
        The upper indices of the tensor.
    lower : Sequence[Index]
        The lower indices of the tensor.
    bra_ket_sym : int, optional
        The bra-ket symmetry of the tensor:
        - 0 no bra-ket-symmetry (d^{i}_{j} != d^{j}_{i})
        - 1 bra-ket symmetry (d^{i}_{j} = d^{j}_{i})
        - -1 bra-ket antisymmetry (d^{i}_{j} = - d^{j}_{i})
        (default: 0)
    """

    def __new__(cls, name: str | Symbol, upper: Sequence[Index],
                lower: Sequence[Index], bra_ket_sym: int = 0):
        # sort upper and lower. No need to track the number of swaps
        upper = sorted(upper, key=sort_idx_canonical)
        lower = sorted(lower, key=sort_idx_canonical)
        # account for the bra ket symmetry
        # add the Index check for subs to work correctly
        negative_sign = False
        bra_ket_sym_imported = sympify(bra_ket_sym)
        if bra_ket_sym_imported is not S.Zero and \
                all(isinstance(s, Index) for s in upper+lower):
            if bra_ket_sym_imported not in [S.One, S.NegativeOne]:
                raise Inputerror("Invalid bra ket symmetry given "
                                 f"{bra_ket_sym}. Valid are 0, 1 or -1.")
            if cls._need_bra_ket_swap(upper, lower):
                upper, lower = lower, upper  # swap
                if bra_ket_sym_imported is S.NegativeOne:
                    negative_sign = True
        # import all quantities to sympy
        name_imported = sympify(name)
        upper_imported, lower_imported = Tuple(*upper), Tuple(*lower)
        # attach -1 if necessary
        if negative_sign:
            return - super(AntiSymmetricTensor, cls).__new__(
                cls, name_imported, upper_imported, lower_imported,
                bra_ket_sym_imported
            )
        else:
            return super(AntiSymmetricTensor, cls).__new__(
                cls, name_imported, upper_imported, lower_imported,
                bra_ket_sym_imported
            )


class NonSymmetricTensor(SymbolicTensor):
    """
    Represents tensors that do not have any symmetry.

    Parameters
    ----------
    name : str | Symbol
        The name of the tensor.
    indices : Sequence[Index] | Tuple
        The indices of the tensor.
    """

    def __new__(cls, name: str | Symbol, indices: Sequence[Index] | Tuple):
        symbol_imported = sympify(name)
        indices_imported = Tuple(*indices)
        return super().__new__(cls, symbol_imported, indices_imported)

    def _latex(self, printer) -> str:
        indices = self.indices.args
        assert _is_index_tuple(indices)
        return "{%s_{%s}}" % (
            self.symbol,
            "".join(i._latex(printer) for i in indices)
        )

    def __str__(self):
        return "%s%s" % self.args

    @property
    def indices(self) -> Tuple:
        """Returns the indices of the tensor."""
        indices = self.args[1]
        assert isinstance(indices, Tuple)
        return indices

    @property
    def idx(self) -> tuple[Index, ...]:
        """Returns the indices of the tensor."""
        idx = self.args[1].args
        assert _is_index_tuple(idx)
        return idx


class KroneckerDelta(DefinedFunction):
    """
    Represents a Kronecker delta.
    Based on the implementation in 'sympy.functions.special.tensor_functions'.
    """

    @classmethod
    def eval(cls, i: Expr, j: Expr) -> Expr | None:  # type: ignore[override]
        """
        Evaluates the KroneckerDelta. Adapted from sympy to also cover Spin.
        """
        # This is needed for subs with simultaneous=True
        if not isinstance(i, Index) or not isinstance(j, Index):
            return None

        diff = i - j
        if diff.is_zero:
            return S.One
        elif fuzzy_not(diff.is_zero):
            return S.Zero

        spi, spj = i.space[0], j.space[0]
        valid_spaces = ["o", "v", "g", "c", "a"]
        assert spi in valid_spaces and spj in valid_spaces
        if (spi == "g" and spj == "a") or (spi == "a" or spj == "g"):
            return S.Zero
        if spi != "g" and spj != "g" and spi != spj:  # delta_ov / delta_vo
            return S.Zero
        spi, spj = i.spin, j.spin
        assert spi in ["", "a", "b"] and spj in ["", "a", "b"]
        if spi and spj and spi != spj:  # delta_ab / delta_ba
            return S.Zero
        # sort the indices of the delta
        if i != min(i, j, key=sort_idx_canonical):
            return cls(j, i)
        return None

    def _eval_power(self, exp) -> Expr:  # type: ignore[override]
        # we don't want exponents > 1 on deltas!
        if exp.is_positive:
            return self
        elif exp.is_negative and exp is not S.NegativeOne:
            return S.One / self

    def _latex(self, printer) -> str:
        return (
            "\\delta_{" + " ".join(s._latex(printer) for s in self.idx) + "}"
        )

    @property
    def idx(self) -> tuple[Index, Index]:
        """Returns the indices of the Kronecker delta."""
        idx = self.args
        assert _is_index_tuple(idx) and len(idx) == 2
        return idx

    @property
    def preferred_and_killable(self) -> tuple[Index, Index] | None:
        """
        Returns the preferred (first) and killable (second) index of the
        kronecker delta. The preferred index contains at least as much
        information as the killable index. Therefore, 'evaluate_deltas'
        will always try to keep the preferred index in the expression.
        """
        i, j = self.args
        assert isinstance(i, Index) and isinstance(j, Index)
        space1, spin1 = i.space[0], i.spin
        space2, spin2 = j.space[0], j.spin
        # ensure we have no unexpected space and spin
        assert (
            space1 in ["o", "v", "g", "c", "a"]
            and space2 in ["o", "v", "g", "c", "a"]
        )
        assert spin1 in ["", "a", "b"] and spin2 in ["", "a", "b"]

        if spin1 == spin2:  # nn / aa / bb  -> equal spin information
            # oo / vv / cc / gg / og / vg / cg / aa
            # RI indices will always end up here
            if space1 == space2 or space2 == "g":
                return (i, j)
            else:  # go / gv / gc
                return (j, i)
        elif spin2:  # na / nb  -> 2 holds more spin information
            # oo / vv / cc / gg / go / gv / gc
            if space1 == space2 or space1 == "g":
                return (j, i)
            else:  # og / vg / cg -> 1 holds more space information
                return None
        else:  # an / bn  -> 1 holds more spin information
            # oo / vv / cc / gg / og / vg / cg
            if space1 == space2 or space2 == "g":
                return (i, j)
            else:  # go / gv / gc -> 2 holds more space information
                return None

    @property
    def indices_contain_equal_information(self) -> bool:
        """Whether both indices contain the same amount of information."""
        i, j = self.args
        assert isinstance(i, Index) and isinstance(j, Index)
        return i.space == j.space and i.spin == j.spin
