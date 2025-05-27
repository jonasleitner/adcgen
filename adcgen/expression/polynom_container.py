from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import Any, TYPE_CHECKING

from sympy import Add, Expr, Mul, Pow, S

from ..indices import Index, sort_idx_canonical
from .container import Container
from .object_container import ObjectContainer

# imports only required for type checking (avoid circular imports)
if TYPE_CHECKING:
    from .term_container import TermContainer
    from .expr_container import ExprContainer


class PolynomContainer(ObjectContainer):
    """
    Wrapper for a polynom of the form (a + b + ...)^x

    Parameters
    ----------
    inner:
        The polynom to wrap
    target_idx: Iterable[Index] | None, optional
        Target indices of the expression. By default the Einstein sum
        convention will be used to identify target and contracted indices,
        which is not always sufficient.
    """
    def __init__(self, inner: Expr | Container | Any,
                 target_idx: Iterable[Index] | None = None) -> None:
        # call init from ObjectContainers parent class
        super(ObjectContainer, self).__init__(
            inner=inner, target_idx=target_idx
        )
        if isinstance(self._inner, Pow):
            assert isinstance(self._inner.args[0], Add)
        else:
            # (a + b + ...)^1 (orbital energy denominator)
            assert isinstance(self._inner, Add)

    def __len__(self) -> int:
        return len(self.base.args)

    @cached_property
    def terms(self) -> "tuple[TermContainer, ...]":
        from .term_container import TermContainer

        return tuple(
            TermContainer(inner=term, **self.assumptions)
            for term in self.base.args
        )

    #################################################
    # compute additional properties for the Polynom #
    #################################################
    @property
    def type_as_str(self) -> str:
        return 'polynom'

    @cached_property
    def idx(self) -> tuple[Index, ...]:
        """
        Returns all indices that occur in the polynom. Indices that occur
        multiple times will be listed multiple times.
        """
        idx = [s for t in self.terms for s in t.idx]
        return tuple(sorted(idx, key=sort_idx_canonical))

    @cached_property
    def order(self):
        raise NotImplementedError("Order not implemented for polynoms: "
                                  f"{self}")

    def crude_pos(self, *args, **kwargs):
        _, _ = args, kwargs
        raise NotImplementedError("crude_pos for determining index positions "
                                  f"not implemented for polynoms: {self}")

    def description(self, *args, **kwargs):
        _, _ = args, kwargs
        raise NotImplementedError("description not implemented for polynoms:",
                                  f"{self}")

    @property
    def allowed_spin_blocks(self) -> None:
        # allowed spin blocks not available for Polynoms
        return None

    @property
    def contains_only_orb_energies(self) -> bool:
        """Whether the poylnom only contains orbital energy tensors."""
        return all(term.contains_only_orb_energies for term in self.terms)

    ####################################
    # methods manipulating the polynom #
    ####################################
    def _apply_tensor_braket_sym(
            self, braket_sym_tensors: Sequence[str] = tuple(),
            braket_antisym_tensors: Sequence[str] = tuple(),
            wrap_result: bool = True) -> "ExprContainer | Expr":
        """
        Applies the tensor bra-ket symmetry defined in braket_sym_tensors and
        braket_antisym_tensors to all tensors in the polynom.
        If wrap_result is set,
        the new term will be wrapped by :py:class:`ExprContainer`.
        """
        from .expr_container import ExprContainer

        with_sym = S.Zero
        for term in self.terms:
            with_sym += term._apply_tensor_braket_sym(
                braket_sym_tensors=braket_sym_tensors,
                braket_antisym_tensors=braket_antisym_tensors,
                wrap_result=False
            )
        assert isinstance(with_sym, Expr)
        with_sym = Pow(with_sym, self.exponent)

        if wrap_result:
            with_sym = ExprContainer(inner=with_sym, **self.assumptions)
        return with_sym

    def _rename_complex_tensors(self, wrap_result: bool = True
                                ) -> "ExprContainer | Expr":
        """
        Renames complex tensors to reflect that the expression is
        represented in a real orbital basis, e.g., complex t-amplitudes
        are renamed t1cc -> t1.

        Parameters
        ----------
        wrap_result: bool, optional
            If set the result will be wrapped with
            :py:class:`ExprContainer`. Otherwise the unwrapped
            object is returned. (default: True)
        """
        from .expr_container import ExprContainer

        real = S.Zero
        for term in self.terms:
            real += term._rename_complex_tensors(wrap_result=False)
        assert isinstance(real, Expr)
        real = Pow(real, self.exponent)

        if wrap_result:
            real = ExprContainer(inner=real, **self.assumptions)
        return real

    def block_diagonalize_fock(self, wrap_result: bool = True
                               ) -> "ExprContainer | Expr":
        """
        Block diagonalize the fock matrix in the polynom by removing terms
        that contain elements of off-diagonal blocks.
        """
        from .expr_container import ExprContainer

        bl_diag = S.Zero
        for term in self.terms:
            bl_diag += term.block_diagonalize_fock(wrap_result=False)
        assert isinstance(bl_diag, Expr)
        bl_diag = Pow(bl_diag, self.exponent)

        if wrap_result:
            bl_diag = ExprContainer(inner=bl_diag, **self.assumptions)
        return bl_diag

    def diagonalize_fock(self, target: Sequence[Index],
                         wrap_result: bool = True
                         ):
        _, _ = target, wrap_result
        raise NotImplementedError("Fock matrix diagonalization not implemented"
                                  f" for polynoms: {self}")

    def rename_tensor(self, current: str, new: str,
                      wrap_result: bool = True
                      ) -> "ExprContainer | Expr":
        """Rename a tensor from current to new."""
        from .expr_container import ExprContainer

        renamed = S.Zero
        for term in self.terms:
            renamed += term.rename_tensor(current, new, wrap_result=False)
        assert isinstance(renamed, Expr)
        renamed = Pow(renamed, self.exponent)

        if wrap_result:
            renamed = ExprContainer(inner=renamed, **self.assumptions)
        return renamed

    def expand_coulomb_ri(self, factorisation: str = 'sym',
                          wrap_result: bool = True) -> "Expr | ExprContainer":
        """
        Expands the Coulomb operators (pq | rs) into RI format

        Parameters
        ----------
        factorisation : str, optional
            The type of factorisation ('sym' or 'asym'), by default 'sym'
        wrap_result : bool, optional
            Whether to wrap the result in an ExprContainer, by default True
        """
        from .expr_container import ExprContainer

        exponent = self.exponent
        if not exponent.is_Integer:
            raise ValueError("Can only apply RI approximation to Polynomials "
                             "with integer exponents. "
                             f"{self} has an invalid exponent.")
        # use a for loop so the contracted aux indices for each x in
        # x * x * ... = x^n are different.
        expanded = S.One
        for _ in range(abs(int(exponent))):
            contrib = S.Zero
            for term in self.terms:
                contrib += term.expand_coulomb_ri(
                    factorisation=factorisation, wrap_result=False
                )
            assert isinstance(contrib, Expr)
            if exponent < S.Zero:
                # a mul object would be simplified as
                # (ab)^-1 -> a^-1 b^-1
                # which is only correct if a and b have no contracted indices.
                assert not isinstance(contrib, Mul)
                contrib = Pow(contrib, -1)
            expanded *= contrib
        if wrap_result:
            expanded = ExprContainer(inner=expanded, **self.assumptions)
        return expanded

    def expand_antisym_eri(self, wrap_result: bool = True):
        """
        Expands the antisymmetric ERI using chemists notation
        <pq||rs> = (pr|qs) - (ps|qr).
        ERI's in chemists notation are by default denoted as 'v'.
        Currently this only works for real orbitals, i.e.,
        for symmetric ERI's <pq||rs> = <rs||pq>.
        """
        from .expr_container import ExprContainer

        expanded = S.Zero
        for term in self.terms:
            expanded += term.expand_antisym_eri(wrap_result=False)
        assert isinstance(expanded, Expr)
        expanded = Pow(expanded, self.exponent)

        if wrap_result:
            expanded = ExprContainer(inner=expanded, **self.assumptions)
        return expanded

    def expand_intermediates(self, target: Sequence[Index],
                             wrap_result: bool = True,
                             fully_expand: bool = True,
                             braket_sym_tensors: Sequence[str] = tuple(),
                             braket_antisym_tensors: Sequence[str] = tuple()
                             ) -> "ExprContainer | Expr":
        """Expands all known intermediates in the polynom."""
        from .expr_container import ExprContainer

        exponent = self.exponent
        if not exponent.is_Integer:
            raise NotImplementedError("Can only expand intermediates for "
                                      "polynoms with integer exponents."
                                      f"{self} has an invalid exponent.")
        # use a for loop so the contracted itmd indices for each x in
        # x * x * ... = x^n are different.
        expanded = S.One
        for _ in range(abs(int(exponent))):
            contrib = S.Zero
            for term in self.terms:
                contrib += term.expand_intermediates(
                    target, wrap_result=False, fully_expand=fully_expand,
                    braket_sym_tensors=braket_sym_tensors,
                    braket_antisym_tensors=braket_antisym_tensors
                )
            assert isinstance(contrib, Expr)
            if exponent < S.Zero:
                # a mul object would be simplified as
                # (ab)^-1 -> a^-1 b^-1
                # which is only correct if a and b have no contracted indices.
                assert not isinstance(contrib, Mul)
                contrib = Pow(contrib, -1)
            expanded *= contrib
        if wrap_result:
            assumptions = self.assumptions
            assumptions["target_idx"] = target
            return ExprContainer(expanded, **assumptions)
        return expanded

    def use_explicit_denominators(self, wrap_result: bool = True
                                  ) -> "ExprContainer | Expr":
        """
        Switch to an explicit representation of orbital energy denominators by
        replacing all symbolic denominators by their explicit counter part,
        i.e., D^{ij}_{ab} -> (e_i + e_j - e_a - e_b)^{-1}.
        """
        from .expr_container import ExprContainer

        explicit_denom = S.Zero
        for term in self.terms:
            explicit_denom += term.use_explicit_denominators(wrap_result=False)
        assert isinstance(explicit_denom, Expr)
        explicit_denom = Pow(explicit_denom, self.exponent)

        if wrap_result:
            explicit_denom = ExprContainer(
                inner=explicit_denom, **self.assumptions
            )
        return explicit_denom

    def to_latex_str(self, only_pull_out_pref: bool = False,
                     spin_as_overbar: bool = False) -> str:
        """Returns a latex string for the polynom."""
        tex_str = " ".join(
            term.to_latex_str(only_pull_out_pref=only_pull_out_pref,
                              spin_as_overbar=spin_as_overbar)
            for term in self.terms
        )
        tex_str = f"\\bigl({tex_str}\\bigr)"
        if self.exponent != 1:
            tex_str += f"^{{{self.exponent}}}"
        return tex_str
