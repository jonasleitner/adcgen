from collections.abc import Iterable, Sequence
from typing import Any, Literal

from sympy import Add, Basic, Expr, Mul, Pow, S, Symbol, factor, nsimplify

from ..indices import (
    Index, get_symbols, sort_idx_canonical,
    _is_str_sequence, _is_index_sequence
)
from ..tensor_names import tensor_names
from .container import Container
from .term_container import TermContainer


class ExprContainer(Container):
    """
    Wraps an arbitrary algebraic expression.

    Parameters
    ----------
    inner:
        The algebraic expression to wrap, e.g., a sympy.Add or sympy.Mul object
    real : bool, optional
        Whether the expression is represented in a real orbital basis.
    sym_tensors: Iterable[str] | None, optional
        Names of tensors with bra-ket-symmetry, i.e.,
        d^{pq}_{rs} = d^{rs}_{pq}. Adjusts the corresponding tensors to
        correctly represent this additional symmetry if they are not aware
        of it yet.
    antisym_tensors: Iterable[str] | None, optional
        Names of tensors with bra-ket-antisymmetry, i.e.,
        d^{pq}_{rs} = - d^{rs}_{pq}. Adjusts the corresponding tensors to
        correctly represent this additional antisymmetry if they are not
        aware of it yet.
    target_idx: Iterable[Index] | None, optional
        Target indices of the expression. By default the Einstein sum
        convention will be used to identify target and contracted indices,
        which is not always sufficient.
    """
    def __init__(self, inner: Expr | Container | Any,
                 real: bool = False,
                 sym_tensors: Iterable[str] = tuple(),
                 antisym_tensors: Iterable[str] = tuple(),
                 target_idx: Iterable[Index] | Iterable[str] | None = None
                 ) -> None:
        # import target index strings
        if target_idx is not None:
            if isinstance(target_idx, str) or isinstance(target_idx, Sequence):
                target_idx = get_symbols(target_idx)
            else:
                target_tpl = tuple(target_idx)
                assert (_is_str_sequence(target_tpl) or
                        _is_index_sequence(target_tpl))
                target_idx = get_symbols(target_tpl)
                del target_tpl
        # set the class attributes and import the inner expression
        super().__init__(
            inner=inner, real=real, sym_tensors=sym_tensors,
            antisym_tensors=antisym_tensors, target_idx=target_idx
        )
        # ensure that sym_tensor and antisym_tensor are immutable tuples and
        # remove duplicates
        self._sym_tensors = tuple(sorted(set(self._sym_tensors)))
        self._antisym_tensors = tuple(sorted(set(self._antisym_tensors)))
        # Now apply the given assumptions: this only happens in this class
        # store target indices as sorted tuple
        if self._target_idx is not None:
            self.set_target_idx(self._target_idx)
        # applying the tensor symmetry has a certain overlap with
        # make_real: make_real will try to add ERI and Fock matrix
        # to sym_tensor and apply the tensor symmetry (but only
        # if the tensors were not already marked as symmetric).
        # Therefore, it makes sense to manually add them here
        # to avoid applying the tensor symmetry twice.
        if self._sym_tensors or self._antisym_tensors:
            if self._real:
                self._sym_tensors = tuple(sorted(set(
                    self._sym_tensors + (tensor_names.fock, tensor_names.eri)
                )))
            self._apply_tensor_braket_sym()
        if self._real:
            self.make_real(force=True)

    def __len__(self) -> int:
        # ExprContainer(0) also has length 1!
        if isinstance(self._inner, Add):
            return len(self._inner.args)
        else:
            return 1

    def copy(self) -> "ExprContainer":
        """
        Creates a new container with the same expression and assumptions.
        The wrapped expression will not be copied.
        """
        return ExprContainer(self.inner, **self.assumptions)

    @property
    def terms(self) -> tuple[TermContainer, ...]:
        """
        Returns all terms of the expression, where a term might be a single
        tensor 'a' or a product of the form 'a * b * c'.
        """
        kwargs = self.assumptions
        if isinstance(self._inner, Add):
            return tuple(
                TermContainer(inner=term, **kwargs)
                for term in self._inner.args
            )
        else:
            return (TermContainer(inner=self._inner, **kwargs),)

    @property
    def idx(self) -> tuple[Index, ...]:
        """
        Returns all indices that occur in the expression. Indices that occur
        multiple times will be listed multiple times.
        """
        idx = [s for t in self.terms for s in t.idx]
        return tuple(sorted(idx, key=sort_idx_canonical))

    ###############################
    # setters for the assumptions #
    ###############################
    def set_target_idx(self, target_idx: Sequence[str] | Sequence[Index] | None
                       ) -> None:
        """
        Set the target indices of the expression. Only necessary if the
        Einstein sum contension is not sufficient to determine them
        automatically.
        """
        if target_idx is None:
            self._target_idx = target_idx
        else:  # import the indices
            self._target_idx = tuple(
                sorted(set(get_symbols(target_idx)), key=sort_idx_canonical)
            )

    @Container.sym_tensors.setter
    def sym_tensors(self, tensors: Iterable[str]) -> None:
        """
        Add bra-ket-symmetry to tensors according to their name.
        Note that it is only formally possible to remove tensors from
        sym_tensors, because the original state of a tensor is lost when the
        bra-ket-symmetry is applied, i.e., after bra-ket-symmetry was added to
        a tensor d^{p}_{q} it is not knwon whether it's original state was
        d^{q}_{p} or d^{p}_{q}.
        """
        if isinstance(tensors, str):
            tensors = {tensors, }
        else:
            assert all(isinstance(t, str) for t in tensors)
            tensors = set(tensors)

        if self.real:
            tensors.update([tensor_names.fock, tensor_names.eri])
        tensors = tuple(sorted(tensors))
        if tensors != self._sym_tensors:
            self._sym_tensors = tensors
            self._apply_tensor_braket_sym()

    @Container.antisym_tensors.setter
    def antisym_tensors(self, tensors: Iterable[str]) -> None:
        """
        Add bra-ket-antisymmetry to tensors according to their name.
        Note that it is only formally possible to remove tensors from
        sym_tensors, because the original state of a tensor is lost when the
        bra-ket-symmetry is applied, i.e., after bra-ket-antisymmetry was
        added to a tensor d^{p}_{q} it is not knwon whether it's original
        state was d^{q}_{p} or d^{p}_{q}.
        """
        if isinstance(tensors, str):
            tensors = (tensors,)
        else:
            assert all(isinstance(t, str) for t in tensors)
            tensors = tuple(sorted(set(tensors)))

        if tensors != self._antisym_tensors:
            self._antisym_tensors = tensors
            self._apply_tensor_braket_sym()

    #################################################
    # methods that modify the expression (in place) #
    #################################################
    def _apply_tensor_braket_sym(self) -> "ExprContainer":
        """
        Adds the bra-ket symmetry and antisymmetry defined in
        sym_tensors and antisym_tensors to the tensor objects
        in the expression.
        """
        if self.inner.is_number:
            return self
        # actually do something
        res = S.Zero
        for term in self.terms:
            res += term._apply_tensor_braket_sym(wrap_result=False)
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def make_real(self, force: bool = False) -> "ExprContainer":
        """
        Represent the expression in a real orbital basis.
        - names of complex conjugate t-amplitudes, for instance t1cc -> t1
        - adds bra-ket-symmetry to the fock matrix and the ERI.

        Parameters
        ----------
        force: bool, optional
            If set the function will also run also if 'real' is already set.
            (default: False)
        """
        if (self.real and not force):
            return self
        # actually so something: first adjust the tensor symmetry
        self._real = True
        sym_tensors = self._sym_tensors
        if tensor_names.fock not in sym_tensors or \
                tensor_names.eri not in sym_tensors:
            self._sym_tensors = tuple(sorted(set(
                sym_tensors + (tensor_names.fock, tensor_names.eri)
            )))
            self._apply_tensor_braket_sym()
        if self.inner.is_number:
            return self
        # and then adjust the tensor names
        res = S.Zero
        for term in self.terms:
            res += term.make_real(wrap_result=False)
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def block_diagonalize_fock(self) -> "ExprContainer":
        """
        Block diagonalize the Fock matrix, i.e. all terms that contain off
        diagonal Fock matrix blocks (f_ov/f_vo) are set to 0.
        """
        self.expand()
        res = S.Zero
        for term in self.terms:
            res += term.block_diagonalize_fock(wrap_result=False)
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def diagonalize_fock(self) -> "ExprContainer":
        """
        Represent the expression in the canonical orbital basis, where the
        fock matrix is diagonal. Because it is not possible to
        determine the target indices in the resulting expression according
        to the Einstein sum convention, the current target indices will
        be set manually in the resulting expression.
        """
        # expand to get rid of polynoms as much as possible
        self.expand()
        diag = S.Zero
        for term in self.terms:
            contrib = term.diagonalize_fock(wrap_result=True)
            assert isinstance(contrib, ExprContainer)
            diag += contrib
        assert isinstance(diag, ExprContainer)
        self._inner = diag.inner
        self._target_idx = diag.provided_target_idx
        return self

    def rename_tensor(self, current: str, new: str) -> 'ExprContainer':
        """Changes the name of a tensor from current to new."""
        assert isinstance(current, str) and isinstance(new, str)

        renamed = S.Zero
        for term in self.terms:
            renamed += term.rename_tensor(current, new, wrap_result=False)
        assert isinstance(renamed, Expr)
        self._inner = renamed
        return self

    def expand_coulomb_ri(self, factorisation: Literal['sym', 'asym'] = 'sym'
                          ) -> 'ExprContainer':
        """
        Factorises symmetric ERIs in chemist notation into RI format.
        This can be done both symmetrically and asymetrically

        Args:
            factorisation : str, optional
                Either 'sym' or 'asym'. Defaults to 'sym'.

        Returns:
            ExprContainer: The factorised result
        """
        res = S.Zero
        for term in self.terms:
            res += term.expand_coulomb_ri(factorisation=factorisation,
                                          wrap_result=False)
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def expand_antisym_eri(self) -> 'ExprContainer':
        """
        Expands the antisymmetric ERI using chemists notation
        <pq||rs> = (pr|qs) - (ps|qr).
        ERI's in chemists notation are by default denoted as 'v'.
        Currently this only works for real orbitals, i.e.,
        for symmetric ERI's <pq||rs> = <rs||pq>."""
        res = S.Zero
        for term in self.terms:
            res += term.expand_antisym_eri(wrap_result=False)
        assert isinstance(res, Expr)
        self._inner = res
        # only update the assumptions if there was an eri to expand
        if Symbol(tensor_names.coulomb) in self.inner.atoms(Symbol):
            self._sym_tensors = tuple(sorted(set(
                self._sym_tensors + (tensor_names.coulomb,)
            )))
        return self

    def use_explicit_denominators(self) -> 'ExprContainer':
        """
        Switch to an explicit representation of orbital energy denominators by
        replacing all symbolic denominators by their explicit counter part,
        i.e., D^{ij}_{ab} -> (e_i + e_j - e_a - e_b)^{-1}.
        """
        res = S.Zero
        for term in self.terms:
            res += term.use_explicit_denominators(wrap_result=False)
        assert isinstance(res, Expr)
        self._inner = res
        # remove the symbolic denom from the assumptions if necessary
        if tensor_names.sym_orb_denom in self._antisym_tensors:
            self._antisym_tensors = tuple(
                t for t in self._antisym_tensors
                if t != tensor_names.sym_orb_denom
            )
        return self

    def substitute_contracted(self) -> 'ExprContainer':
        """
        Tries to substitute all contracted indices with pretty indices, i.e.
        i, j, k instad of i3, n4, o42 etc.
        """
        self.expand()
        res = S.Zero
        for term in self.terms:
            contrib = term.substitute_contracted(wrap_result=False)
            assert isinstance(contrib, Expr)
            res += contrib
        self._inner = res
        return self

    def substitute_with_generic(self) -> 'ExprContainer':
        """
        Subsitutes all contracted indices with new, unused generic indices.
        """
        self.expand()
        res = S.Zero
        for term in self.terms:
            res += term.substitute_with_generic(wrap_result=False)
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def factor(self, num=None) -> 'ExprContainer':
        """
        Tries to factors the expression. Note: this only works for simple cases

        Parameters
        ----------
        num : optional
            Number to factor in the expression.
        """

        if num is None:
            res = factor(self.inner)
        else:
            num = nsimplify(num, rational=True)
            factored = map(
                lambda t: Mul(nsimplify(Pow(num, -1), rational=True), t.inner),
                self.terms
            )
            res = Mul(num, Add(*factored), evaluate=False)
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def expand_intermediates(self, fully_expand: bool = True
                             ) -> 'ExprContainer':
        """
        Expand the known intermediates in the expression.

        Parameters
        ----------
        fully_expand: bool, optional
            True (default): The intermediates are recursively expanded
              into orbital energies and ERI (if possible)
            False: The intermediates are only expanded once, e.g., n'th
              order MP t-amplitudes are expressed by means of (n-1)'th order
              MP t-amplitudes and ERI.
        """
        # TODO: only expand specific intermediates
        # need to adjust the target indices -> not necessarily possible to
        # determine them after expanding intermediates
        expanded = S.Zero
        for t in self.terms:
            expanded += t.expand_intermediates(fully_expand=fully_expand)
        assert isinstance(expanded, ExprContainer)
        self._inner = expanded.inner
        self.set_target_idx(expanded.provided_target_idx)
        return self

    def use_symbolic_denominators(self) -> "ExprContainer":
        """
        Replace all orbital energy denominators in the expression by tensors,
        e.g., (e_a + e_b - e_i - e_j)^{-1} will be replaced by D^{ab}_{ij},
        where D is a SymmetricTensor.
        """
        symbolic_denom = S.Zero
        has_symbolic_denom = False
        for term in self.terms:
            term = term.use_symbolic_denominators()
            symbolic_denom += term.inner
            if tensor_names.sym_orb_denom in term.antisym_tensors:
                has_symbolic_denom = True
        # the symbolic denominators have additional antisymmetry
        # for bra ket swaps
        # -> this is the only possible change in the assumptions
        # -> only set if we replaced a denominator in the expr
        assert isinstance(symbolic_denom, Expr)
        self._inner = symbolic_denom
        if has_symbolic_denom:
            self._antisym_tensors = tuple(sorted(set(
                self._antisym_tensors + (tensor_names.sym_orb_denom,)
            )))
        return self

    ###########################################################
    # Overwrite parent class methods for inplace modification #
    ###########################################################
    def expand(self):
        """
        Forwards the expand call to inner replacing the wrapped
        expression
        """
        res = self._inner.expand()
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def doit(self, *args, **kwargs):
        """
        Forwards the doit call to inner replacing the wrapped
        expression
        """
        res = self._inner.doit(*args, **kwargs)
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def subs(self, *args, **kwargs):
        """
        Forwards the subs call to inner replacing the wrapped
        expression
        """
        res = self._inner.subs(*args, **kwargs)
        assert isinstance(res, Expr)
        self._inner = res
        return self

    #######################################
    # Operators for in-place modification #
    #######################################
    def __iadd__(self, other: Any) -> "ExprContainer":
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.inner
        elif isinstance(other, Basic):
            # Apply the assumptions to the sympy object
            other = ExprContainer(other, **self.assumptions).inner
        res = self.inner + other
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def __isub__(self, other: Any) -> "ExprContainer":
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.inner
        elif isinstance(other, Basic):
            # Apply the assumptions to the sympy object
            other = ExprContainer(other, **self.assumptions).inner
        res = self.inner - other
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def __imul__(self, other: Any) -> "ExprContainer":
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.inner
        elif isinstance(other, Basic):
            # Apply the assumptions to the sympy object
            other = ExprContainer(other, **self.assumptions).inner
        res = self.inner * other
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def __itruediv__(self, other: Any) -> "ExprContainer":
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.inner
        elif isinstance(other, Basic):
            other = ExprContainer(other, **self.assumptions).inner
        res = self.inner / other
        assert isinstance(res, Expr)
        self._inner = res
        return self

    def to_latex_str(self, terms_per_line: int | None = None,
                     only_pull_out_pref: bool = False,
                     spin_as_overbar: bool = False) -> str:
        """
        Transforms the expression to a latex string.

        Parameters
        ----------
        terms_per_line: int, optional
            Returns the expression using the syntax from an 'align'
            environment with the provided number of terms per line.
        only_pull_out_pref: bool, optional
            Use the 'latex' printout from sympy, while prefactors are printed
            in front of each term. This avoids long fractions with a huge
            number of tensors in the numerator and only a factor in the
            denominator.
        spin_as_overbar: bool, optional
            Instead of printing the spin of an index as suffix (idxname_spin)
            use an overbar for beta spin and no indication for alpha. Because
            alpha indices and indices without spin are not distinguishable
            anymore, this only works if all indices have a spin set (the
            expression is completely represented in spatial orbitals).
        """
        tex_terms = [
            term.to_latex_str(only_pull_out_pref, spin_as_overbar)
            for term in self.terms
        ]
        # remove '+' in the first term
        if tex_terms[0].lstrip().startswith("+"):
            tex_terms[0] = tex_terms[0].replace('+', '', 1).lstrip()
        # just the raw output without linebreaks
        if terms_per_line is None:
            return " ".join(tex_terms)
        assert isinstance(terms_per_line, int)
        # only print x terms per line in an align environment
        # create the string of all but the last line
        tex_string = ""
        for i in range(0, len(tex_terms) - terms_per_line, terms_per_line):
            tex_string += (
                "& " + " ".join(tex_terms[i:i+terms_per_line]) +
                " \\nonumber\\\\\n"
            )
        # add the last line. Could ommit this if the equation is not supposed
        # to have a number.
        if len(tex_terms) % terms_per_line:
            remaining = len(tex_terms) % terms_per_line
        else:
            remaining = terms_per_line
        tex_string += "& " + " ".join(tex_terms[-remaining:])
        return tex_string
