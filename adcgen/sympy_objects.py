from sympy.physics.secondquant import (
    _sort_anticommuting_fermions, ViolationOfPauliPrinciple
)
from sympy.core.function import Function
from sympy.core.expr import Expr
from sympy.core.logic import fuzzy_not
from sympy import sympify, Tuple, Symbol, S
from .misc import Inputerror
from .indices import Index, sort_idx_canonical


class SymbolicTensor(Expr):
    """Base class for symbolic tensors."""

    is_commutative = True

    @property
    def symbol(self) -> Symbol:
        """Returns the symbol of the tensor."""
        return self.args[0]

    @property
    def name(self) -> str:
        """Returns the name of the tensor."""
        return self.symbol.name

    @property
    def idx(self) -> tuple[Index]:
        """Returns all indices of the tensor."""
        raise NotImplementedError(f"'idx' not implemented on {self.__class__}")


class AntiSymmetricTensor(SymbolicTensor):
    """
    Represents antisymmetric tensors
    d^{pq}_{rs} = - d^{qp}_{rs} = - d^{pq}_{sr} = d^{qp}_{sr}.
    Based on the implementation in 'sympy.physics.secondquant'.

    Parameters
    ----------
    name : str
        The name of the tensor.
    upper : tuple[Index]
        The upper indices of the tensor.
    lower : tuple[Index]
        The lower indices of the tensor.
    bra_ket_sym : int, optional
        The bra-ket symmetry of the tensor:
        - 0 no bra-ket-symmetry (d^{i}_{j} != d^{j}_{i})
        - 1 bra-ket symmetry (d^{i}_{j} = d^{j}_{i})
        - -1 bra-ket antisymmetry (d^{i}_{j} = - d^{j}_{i})
        (default: 0)
    """

    def __new__(cls, name: str, upper: tuple[Index], lower: tuple[Index],
                bra_ket_sym: int = 0):
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
        # add the Index check for subs to work correctly
        bra_ket_sym = sympify(bra_ket_sym)
        if bra_ket_sym is not S.Zero and \
                all(isinstance(s, Index) for s in upper+lower):
            if bra_ket_sym not in [S.One, S.NegativeOne]:
                raise Inputerror("Invalid bra ket symmetry given "
                                 f"{bra_ket_sym}. Valid are 0, 1 or -1.")
            if cls._need_bra_ket_swap(upper, lower):
                upper, lower = lower, upper  # swap
                if bra_ket_sym is S.NegativeOne:  # add another -1
                    sign_u += 1
        # import all quantities to sympy
        name = sympify(name)
        upper, lower = Tuple(*upper), Tuple(*lower)

        # attach -1 if necessary
        if (sign_u + sign_l) % 2:
            return - super().__new__(cls, name, upper, lower,
                                     bra_ket_sym)
        else:
            return super().__new__(cls, name, upper, lower, bra_ket_sym)

    @classmethod
    def _need_bra_ket_swap(cls, upper: tuple[Index],
                           lower: tuple[Index]) -> bool:
        if len(upper) != len(lower):
            raise NotImplementedError("Bra Ket symmetry only implemented "
                                      "for tensors with an equal amount "
                                      "of upper and lower indices.")
        # compare the space of upper and lower indices
        space_u = [s.space[0] for s in upper]
        space_l = [s.space[0] for s in lower]
        if space_l < space_u:  # space with more occ should be upper
            return True
        elif space_l == space_u:  # diagonal block
            # compare the spin of both index blocks:
            # space with more spin orbitals or alpha spin should be upper.
            spin_u = [s.spin for s in upper]
            spin_l = [s.spin for s in lower]
            if spin_l < spin_u:
                return True
            elif spin_l == spin_u:  # diagonal spin block
                # compare the names of indices
                lower_names = [(int(s.name[1:]) if s.name[1:] else 0,
                               s.name[0]) for s in lower]
                upper_names = [(int(s.name[1:]) if s.name[1:] else 0,
                               s.name[0]) for s in upper]
                if lower_names < upper_names:
                    return True
        return False

    def _latex(self, printer) -> str:
        return "{%s^{%s}_{%s}}" % (
            self.symbol,
            "".join([i._latex(printer) for i in self.args[1]]),
            "".join([i._latex(printer) for i in self.args[2]])
        )

    def __str__(self):
        return f"{self.symbol}({self.upper},{self.lower})"

    @property
    def upper(self) -> Tuple:
        """Returns the upper indices of the tensor."""
        return self.args[1]

    @property
    def lower(self) -> Tuple:
        """Returns the lower indices of the tensor."""
        return self.args[2]

    @property
    def bra_ket_sym(self):
        """Returns the bra-ket symmetry of the tensor."""
        return self.args[3]

    def add_bra_ket_sym(self, bra_ket_sym: int) -> 'AntiSymmetricTensor':
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
    def idx(self) -> tuple[Index]:
        """
        Returns all indices of the tensor. The upper indices are listed before
        the lower indices.
        """
        return self.upper.args + self.lower.args


class Amplitude(AntiSymmetricTensor):
    """
    Represents antisymmetric Amplitudes.
    """

    @property
    def idx(self) -> tuple[Index]:
        """
        Returns all indices of the amplitude. The lower indices are
        listed before the upper indices.
        """
        return self.lower.args + self.upper.args


class SymmetricTensor(AntiSymmetricTensor):
    """
    Represents symmetric tensors
    d^{pq}_{rs} = d^{qp}_{rs} = d^{pq}_{sr} = d^{qp}_{sr}.

    Parameters
    ----------
    name : str
        The name of the tensor.
    upper : tuple[Index]
        The upper indices of the tensor.
    lower : tuple[Index]
        The lower indices of the tensor.
    bra_ket_sym : int, optional
        The bra-ket symmetry of the tensor:
        - 0 no bra-ket-symmetry (d^{i}_{j} != d^{j}_{i})
        - 1 bra-ket symmetry (d^{i}_{j} = d^{j}_{i})
        - -1 bra-ket antisymmetry (d^{i}_{j} = - d^{j}_{i})
        (default: 0)
    """

    def __new__(cls, name: str, upper: tuple[Index], lower: tuple[Index],
                bra_ket_sym: int = 0):
        # sort upper and lower. No need to track the number of swaps
        upper = sorted(upper, key=sort_idx_canonical)
        lower = sorted(lower, key=sort_idx_canonical)
        # account for the bra ket symmetry
        # add the Index check for subs to work correctly
        negative_sign = False
        bra_ket_sym = sympify(bra_ket_sym)
        if bra_ket_sym is not S.Zero and \
                all(isinstance(s, Index) for s in upper+lower):
            if bra_ket_sym not in [S.One, S.NegativeOne]:
                raise Inputerror("Invalid bra ket symmetry given "
                                 f"{bra_ket_sym}. Valid are 0, 1 or -1.")
            if cls._need_bra_ket_swap(upper, lower):
                upper, lower = lower, upper  # swap
                if bra_ket_sym is S.NegativeOne:
                    negative_sign = True
        # import all quantities to sympy
        name = sympify(name)
        upper, lower = Tuple(*upper), Tuple(*lower)
        # attach -1 if necessary
        if negative_sign:
            return - super(AntiSymmetricTensor, cls).__new__(
                cls, name, upper, lower, bra_ket_sym
            )
        else:
            return super(AntiSymmetricTensor, cls).__new__(
                cls, name, upper, lower, bra_ket_sym
            )


class NonSymmetricTensor(SymbolicTensor):
    """
    Represents tensors that do not have any symmetry.

    Parameters
    ----------
    name : str
        The name of the tensor.
    indices : tuple[Index]
        The indices of the tensor.
    """

    def __new__(cls, name: str, indices: tuple[Index]):
        symbol = sympify(name)
        indices = Tuple(*indices)
        return super().__new__(cls, symbol, indices)

    def _latex(self, printer) -> str:
        return "{%s_{%s}}" % (self.symbol, "".join([i._latex(printer)
                                                    for i in self.indices]))

    def __str__(self):
        return "%s%s" % self.args

    @property
    def indices(self) -> Tuple:
        """Returns the indices of the tensor."""
        return self.args[1]

    @property
    def idx(self) -> tuple[Index]:
        """Returns all indices of the tensor."""
        return self.args[1].args


class KroneckerDelta(Function):
    """
    Represents a Kronecker delta.
    Based on the implementation in 'sympy.functions.special.tensor_functions'.
    """

    @classmethod
    def eval(cls, i: Index, j: Index):
        """
        Evaluates the KroneckerDelta. Adapted from sympy to also cover Spin.
        """

        diff = i - j
        if diff.is_zero or fuzzy_not(diff.is_zero):  # same index
            return S.One

        spi, spj = i.space[0], j.space[0]
        assert spi in ["o", "v", "g"] and spj in ["o", "v", "g"]
        if spi != "g" and spj != "g" and spi != spj:  # delta_ov / delta_vo
            return S.Zero
        spi, spj = i.spin, j.spin
        assert spi in ["", "a", "b"] and spj in ["", "a", "b"]
        if spi and spj and spi != spj:  # delta_ab / delta_ba
            return S.Zero
        # sort the indices of the delta
        if i != min(i, j, key=sort_idx_canonical):
            return cls(j, i)

    def _eval_power(self, exp):
        # we don't want exponents > 1 on deltas!
        if exp.is_positive:
            return self
        elif exp.is_negative and exp is not S.NegativeOne:
            return 1/self

    def _latex(self, printer):
        return (
            "\\delta_{" + " ".join(s._latex(printer) for s in self.args) + "}"
        )

    @property
    def idx(self) -> tuple[Index]:
        """Returns the indices of the Kronecker delta."""
        return self.args

    @property
    def preferred_and_killable(self) -> tuple[Index, Index] | None:
        """
        Returns the preferred (first) and killable (second) index of the
        kronecker delta. The preferred index contains at least as much
        information as the killable index. Therefore, 'evaluate_deltas'
        will always try to keep the preferred index in the expression.
        """
        i, j = self.args
        space1, spin1 = i.space[0], i.spin
        space2, spin2 = j.space[0], j.spin
        # ensure we have no unexpected space and spin
        assert space1 in ["o", "v", "g"] and space2 in ["o", "v", "g"]
        assert spin1 in ["", "a", "b"] and spin2 in ["", "a", "b"]

        if spin1 == spin2:  # nn / aa / bb  -> equal information
            if space1 == space2 or space2 == "g":  # oo / vv / gg / og / vg
                return (i, j)
            else:  # go / gv
                return (j, i)
        elif spin2:  # na / nb  -> 2 holds more information
            if space1 == space2 or space1 == "g":  # oo / vv / gg / go / gv
                return (j, i)
            else:  # og / vg  -> 1 holds more space information
                return None
        else:  # an / bn  -> 1 holds more information
            if space1 == space2 or space2 == "g":  # oo / vv / gg / og / vg
                return (i, j)
            else:  # go / gv  -> 2 holds more space information
                return None

    @property
    def indices_contain_equal_information(self) -> bool:
        """Whether both indices contain the same amount of information."""
        i, j = self.args
        return i.space == j.space and i.spin == j.spin
