from .indices import (get_lowest_avail_indices, get_symbols,
                      order_substitutions, Index, sort_idx_canonical)
from .logger import logger
from .misc import Inputerror, cached_property, cached_member
from .sympy_objects import (
    NonSymmetricTensor, AntiSymmetricTensor, KroneckerDelta, SymmetricTensor,
    SymbolicTensor, Amplitude
)
from .tensor_names import (
    tensor_names, is_t_amplitude, split_t_amplitude_name, is_adc_amplitude,
    is_gs_density, split_gs_density_name
)
from sympy import latex, Add, Mul, Pow, sympify, S, Basic, nsimplify, Symbol
from sympy.physics.secondquant import NO, F, Fd, FermionicOperator


class Container:
    """Base class for all container classes."""

    def expand(self):
        return Expr(self.sympy.expand(), **self.assumptions)

    def doit(self, *args, **kwargs):
        return Expr(self.sympy.doit(*args, **kwargs), **self.assumptions)

    def subs(self, *args, **kwargs):
        return Expr(self.sympy.subs(*args, **kwargs), **self.assumptions)

    def permute(self, *perms):
        """
        Permute indices by applying permutation operators P_pq.

        Parameters
        ----------
        *perms : tuple
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
        # self.subs either modifies in place for expr or creates a new expr
        # object
        return self.subs(order_substitutions(sub))

    def __add__(self, other):
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        return Expr(self.sympy + other, **self.assumptions)

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        # other: some sympy stuff or some number
        return Expr(other + self.sympy, **self.assumptions)

    def __sub__(self, other):
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        return Expr(self.sympy - other, **self.assumptions)

    def __isub__(self, other):
        return self.__sub__(other)

    def __rsub__(self, other):
        # other: some sympy stuff or some number
        return Expr(other - self.sympy, **self.assumptions)

    def __mul__(self, other):
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        return Expr(self.sympy * other, **self.assumptions)

    def __imul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        # other: some sympy stuff or some number
        return Expr(other * self.sympy, **self.assumptions)

    def __truediv__(self, other):
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        return Expr(self.sympy / other, **self.assumptions)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __rtruediv__(self, other):
        # other: some sympy stuff or some number
        return Expr(other / self.sympy, **self.assumptions)


class Expr(Container):
    """
    Wrapper for an algebraic expression.

    Parameters
    ----------
    e :
        The algebraic expression to wrap, e.g., a sympy.Add or sympy.Mul object
    real : bool, optional
        Whether the expression is represented in a real orbital basis.
    sym_tensors: list[str], optional
        Names of tensors with bra-ket-symmetry, i.e.,
        d^{pq}_{rs} = d^{rs}_{pq}. Adjusts the corresponding tensors to
        correctly represent this additional symmetry if they are not aware
        of it yet.
    antisym_tensors: list[str], optional
        Names of tensors with bra-ket-antisymmetry, i.e.,
        d^{pq}_{rs} = - d^{rs}_{pq}. Adjusts the corresponding tensors to
        correctly represent this additional antisymmetry if they are not
        aware of it yet.
    target_idx: list[Index] | list[str], optional
        Target indices of the expression. By default the Einstein sum
        convention will be used to identify target and contracted indices,
        which is not sufficient in all cases.
    """

    def __init__(self, e, real: bool = False, sym_tensors: list[str] = None,
                 antisym_tensors: list[str] = None,
                 target_idx: list[Index] = None):
        if isinstance(e, Container):
            e = e.sympy
        self._expr = sympify(e)
        self._real: bool = False
        self._sym_tensors: set = (set() if sym_tensors is None
                                  else set(sym_tensors))
        self._antisym_tensors: set = (set() if antisym_tensors is None
                                      else set(antisym_tensors))
        self._target_idx: None | tuple[Index] = None
        if target_idx is not None:
            self.set_target_idx(target_idx)
        # first apply the tensor symmetry
        if self._sym_tensors or self._antisym_tensors:
            if real:
                self._sym_tensors.update([tensor_names.fock, tensor_names.eri])
            self._apply_tensor_braket_sym()
        # then check if we are real (make real will adjust the tensor
        # symmetry too - if necessary)
        if real:
            self.make_real()

    def __str__(self):
        return latex(self.sympy)

    def __len__(self):
        # 0 has length 1
        if isinstance(self.sympy, Add):
            return len(self.args)
        else:
            return 1

    def __getattr__(self, attr):
        return getattr(self._expr, attr)

    @property
    def assumptions(self) -> dict:
        """The assumptions currently applied to the expression."""
        return {'real': self.real,
                'sym_tensors': self.sym_tensors,
                'antisym_tensors': self.antisym_tensors,
                'target_idx': self.provided_target_idx}

    @property
    def real(self) -> bool:
        """Whether the expression is represented in a real orbital basis."""
        return self._real

    @property
    def sym_tensors(self) -> tuple:
        """Lists tensors with bra-ket-symmetry the container knows about."""
        return tuple(sorted(self._sym_tensors))

    @property
    def antisym_tensors(self) -> tuple:
        """
        Lists tensors with bra-ket-antisymmetry the container knows about.
        """
        return tuple(sorted(self._antisym_tensors))

    @property
    def provided_target_idx(self) -> None | tuple[Index]:
        """Returns the target indices provided by the user."""
        return self._target_idx

    @property
    def sympy(self):
        """Returns content of the container."""
        return self._expr

    @property
    def type_as_str(self) -> str:
        if isinstance(self.sympy, Add):
            return 'expr'
        elif isinstance(self.sympy, Mul):
            return 'term'
        else:
            return 'obj'

    @property
    def terms(self) -> tuple['Term']:
        """Returns all terms the expression contains."""
        return tuple(Term(self, i) for i in range(len(self)))

    def set_sym_tensors(self, sym_tensors: list[str]) -> None:
        """
        Add bra-ket-symmetry to tensors according to their name.
        Note that it is only formally possible to remove tensors from
        sym_tensors, because the original state of a tensor is lost when the
        bra-ket-symmetry is applied, i.e., after bra-ket-symmetry was added to
        a tensor d^{p}_{p} it is not knwon whether it's original state was
        d^{p}_{p} or d^{p}_{q}.
        """
        if not all(isinstance(t, str) for t in sym_tensors):
            raise Inputerror("Symmetric tensors need to be provided as str.")
        sym_tensors: set = set(sym_tensors)
        if self.real:
            sym_tensors.update([tensor_names.fock, tensor_names.eri])
        if sym_tensors != self._sym_tensors:
            self._sym_tensors = sym_tensors
            self._apply_tensor_braket_sym()

    def set_antisym_tensors(self, antisym_tensors: list[str]) -> None:
        """
        Add bra-ket-antisymmetry to tensors according to their name.
        Note that it is only formally possible to remove tensors from
        antisym_tensors, because the original state of a tensor is lost when
        the bra-ket-antisymmetry is applied, i.e., after bra-ket-symmetry was
        added to a tensor d^{p}_{p} it is not knwon whether it's original state
        was d^{p}_{p} or d^{p}_{q}.
        """
        if not all(isinstance(t, str) for t in antisym_tensors):
            raise Inputerror("Tensors with antisymmetric bra ket symemtry need"
                             "to be provided as string.")
        antisym_tensors = set(antisym_tensors)
        if antisym_tensors != self._antisym_tensors:
            self._antisym_tensors = antisym_tensors
            self._apply_tensor_braket_sym()

    def set_target_idx(self, target_idx: None | list[str | Index]) -> None:
        """
        Set the target indices of the expression. Only necessary if the
        Einstein sum contension is not sufficient to determine them
        automatically.
        """
        if target_idx is None:
            self._target_idx = None
        else:
            target_idx = set(get_symbols(target_idx))
            self._target_idx = tuple(sorted(target_idx,
                                            key=sort_idx_canonical))

    def make_real(self):
        """
        Represent the expression in a real orbital basis.
        - names of complex conjugate t-amplitudes, for instance t1cc -> t1
        - adds bra-ket-symmetry to the fock matrix and the ERI.
        """
        # need to have the option return_sympy at lower levels, because
        # this function may be called upon instantiation

        if self._real:
            return self

        self._real = True
        sym_tensors = self._sym_tensors
        if tensor_names.fock not in sym_tensors or \
                tensor_names.eri not in sym_tensors:
            self._sym_tensors.update([tensor_names.fock, tensor_names.eri])
            self._apply_tensor_braket_sym()
        if self.sympy.is_number:
            return self
        self._expr = Add(*[t.make_real(return_sympy=True)
                           for t in self.terms])
        return self

    def _apply_tensor_braket_sym(self):
        """
        Adds the bra-ket symmetry defined in sym- and antisym_tensors to
        the tensor objects in the expression.
        """
        if self.sympy.is_number:
            return self
        expr_with_sym = Add(*[t._apply_tensor_braket_sym(return_sympy=True)
                              for t in self.terms])
        self._expr = expr_with_sym
        return self

    def block_diagonalize_fock(self):
        """
        Block diagonalize the Fock matrix, i.e. all terms that contain off
        diagonal Fock matrix blocks (f_ov/f_vo) are set to 0.
        """
        self.expand()
        self._expr = Add(*[t.block_diagonalize_fock(return_sympy=True)
                           for t in self.terms])
        return self

    def diagonalize_fock(self):
        """
        Represent the expression in the canonical orbital basis, where the
        Fock matrix is diagonal. Because it is not possible to
        determine the target indices in the resulting expression according
        to the Einstein sum convention, the current target indices will
        be set manually in the resulting expression.
        """
        # expand to get rid of polynoms as much as possible
        self.expand()
        diag = 0
        for term in self.terms:
            diag += term.diagonalize_fock()
        self._expr = diag.sympy
        self._target_idx = diag.provided_target_idx
        return self

    def rename_tensor(self, current, new) -> 'Expr':
        """Changes the name of a tensor from current to new."""
        if not isinstance(current, str) or not isinstance(new, str):
            raise Inputerror("Old and new tensor name need to be provided as "
                             "strings.")
        renamed = 0
        for t in self.terms:
            renamed += t.rename_tensor(current, new, return_sympy=True)
        self._expr = renamed
        return self

    def expand_antisym_eri(self) -> 'Expr':
        """
        Expands the antisymmetric ERI using chemists notation
        <pq||rs> = (pr|qs) - (ps|qr).
        ERI's in chemists notation are by default denoted as 'v'.
        Currently this only works for real orbitals, i.e.,
        for symmetric ERI's <pq||rs> = <rs||pq>."""
        self._expr = Add(*[term.expand_antisym_eri(return_sympy=True)
                           for term in self.terms])
        # only update the assumptions if there was an eri to expand
        if Symbol(tensor_names.coulomb) in self._expr.atoms(Symbol):
            self._sym_tensors.add(tensor_names.coulomb)
        return self

    def use_symbolic_denominators(self):
        """
        Replace all orbital energy denominators in the expression by tensors,
        e.g., (e_a + e_b - e_i - e_j)^{-1} will be replaced by D^{ab}_{ij},
        where D is a SymmetricTensor."""
        symbolic_denom = 0
        has_symbolic_denom = False
        for term in self.terms:
            term = term.use_symbolic_denominators()
            symbolic_denom += term.sympy
            if tensor_names.sym_orb_denom in term.antisym_tensors:
                has_symbolic_denom = True
        # the symbolic denominators have additional antisymmetry
        # for bra ket swaps
        # -> this is the only possible change in the assumptions
        # -> only set if we replaced a denominator in the expr
        self._expr = symbolic_denom
        if has_symbolic_denom:
            self._antisym_tensors.update(tensor_names.sym_orb_denom)
        return self

    def use_explicit_denominators(self):
        """
        Switch to an explicit representation of orbital energy denominators by
        replacing all symbolic denominators by their explicit counter part,
        i.e., D^{ij}_{ab} -> (e_i + e_j - e_a - e_b)^{-1}.
        """
        self._expr = Add(*[term.use_explicit_denominators(return_sympy=True)
                           for term in self.terms])
        # remove the symbolic denom from the assumptions if necessary
        if tensor_names.sym_orb_denom in self._antisym_tensors:
            self._antisym_tensors.remove(tensor_names.sym_orb_denom)
        return self

    def expand(self):
        self._expr = self.sympy.expand()
        return self

    def subs(self, *args, **kwargs):
        self._expr = self.sympy.subs(*args, **kwargs)
        return self

    def doit(self, *args, **kwargs):
        self._expr = self.sympy.doit(*args, **kwargs)
        return self

    def substitute_contracted(self):
        """Tries to substitute all contracted indices with pretty indices, i.e.
           i, j, k instad of i3, n4, o42 etc."""
        self.expand()
        self._expr = Add(*[term.substitute_contracted(return_sympy=True)
                           for term in self.terms])
        return self

    def factor(self, num=None):
        """
        Tries to factors the expression. Note: this only works for simple cases

        Parameters
        ----------
        num : optional
            Number to factor in the expression.
        """
        from sympy import factor, nsimplify
        if num is None:
            self._expr = factor(self.sympy)
            return self
        num = nsimplify(num, rational=True)
        factored = map(lambda t: Mul(nsimplify(Pow(num, -1), rational=True),
                       t.sympy), self.terms)
        self._expr = Mul(num, Add(*factored), evaluate=False)
        return self

    def expand_intermediates(self):
        """Expand all known intermediates in the expression."""
        # TODO: only expand specific intermediates
        # need to adjust the target indices -> not necessarily possible to
        # determine them after expanding intermediates
        expanded: Expr = 0
        for t in self.terms:
            expanded += t.expand_intermediates()
        self._expr = expanded.sympy
        self.set_target_idx(expanded.provided_target_idx)
        return self

    @property
    def idx(self):
        """
        Returns all indices that occur in the expression. Indices that occur
        multiple times will be listed multiple times.
        """
        idx = [s for t in self.terms for s in t.idx]
        return tuple(sorted(idx, key=sort_idx_canonical))

    def copy(self):
        """
        Creates a new container with the same expression and assumptions.
        The wrapped expression will not be copied.
        """
        return Expr(self.sympy, **self.assumptions)

    def to_latex_str(self, terms_per_line: int = None,
                     only_pull_out_pref: bool = False,
                     spin_as_overbar: bool = False) -> str:
        """
        Transforms the expression to a latex string.

        Parameters
        ----------
        terms_per_line: int, optional
            Returns the expression using a 'align' environment with
            the provided number of terms per line.
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
        tex_terms = [term.to_latex_str(only_pull_out_pref, spin_as_overbar)
                     for term in self.terms]
        # remove '+' in the first term
        if tex_terms[0].lstrip().startswith("+"):
            tex_terms[0] = tex_terms[0].replace('+', '', 1).lstrip()
        # just the raw output without linebreaks
        if terms_per_line is None:
            return " ".join(tex_terms)

        # only print x terms per line in an align environment
        if not isinstance(terms_per_line, int):
            raise Inputerror("terms_per_line needs to be an integer.")
        # create the string of all but the last line
        tex_string = ""
        for i in range(0, len(tex_terms) - terms_per_line, terms_per_line):
            tex_string += (
                "& " + " ".join(tex_terms[i:i+terms_per_line]) +
                " \\nonumber\\\\\n"
            )
        # add the last line. Could ommit this if the equation is not supposed
        # to have a number.
        remaining = terms_per_line if len(tex_terms) % terms_per_line == 0 \
            else len(tex_terms) % terms_per_line
        tex_string += "& " + " ".join(tex_terms[-remaining:])
        return tex_string

    def __iadd__(self, other):
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        elif isinstance(other, Basic):
            other = Expr(other, **self.assumptions).sympy
        self._expr = self.sympy + other
        return self

    def __isub__(self, other):
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        elif isinstance(other, Basic):
            other = Expr(other, **self.assumptions).sympy
        self._expr = self.sympy - other
        return self

    def __imul__(self, other):
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        elif isinstance(other, Basic):
            other = Expr(other, **self.assumptions).sympy
        self._expr = self.sympy * other
        return self

    def __itruediv__(self, other):
        if isinstance(other, Container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        elif isinstance(other, Basic):
            other = Expr(other, **self.assumptions).sympy
        self._expr = self.sympy / other
        return self


class Term(Container):
    """
    Wrapper for a single term (a*b*c) that is part of an expression wrapped
    by an instance of the Expr class. Even if the expression only consists
    of a single term, it should be wrapped by the Expr class.

    Parameters
    ----------
    e : Expr
        The 'Expr' the term is contained in.
    pos : int
        The index of the term in the expression (sympy.Add object).
    """
    def __new__(cls, e, pos=None, **assumptions):
        if isinstance(e, (Expr, Polynom)):
            if not isinstance(pos, int):
                raise Inputerror('Position needs to be provided as int.')
            return super().__new__(cls)
        else:
            return Expr(e, **assumptions)

    def __init__(self, e, pos=None, **assumptions):
        self._expr: Expr = e
        self._pos: int = pos
        self._sympy = None

    def __str__(self):
        return latex(self.sympy)

    def __len__(self):
        content = self.sympy
        return len(content.args) if isinstance(content, Mul) else 1

    def __getattr__(self, attr):
        return getattr(self.sympy, attr)

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def assumptions(self) -> dict:
        return self.expr.assumptions

    @property
    def real(self) -> bool:
        return self.expr.real

    @property
    def sym_tensors(self) -> tuple:
        return self.expr.sym_tensors

    @property
    def antisym_tensors(self) -> tuple:
        return self.expr.antisym_tensors

    @property
    def provided_target_idx(self) -> None | tuple:
        return self.expr.provided_target_idx

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def sympy(self):
        """Returns content of the container."""
        # can't cache directly, because getattr is overwritten
        if self._sympy is not None:
            return self._sympy

        if len(self._expr) == 1:
            sympy = self._expr.sympy
        else:
            sympy = self._expr.args[self._pos]
        self._sympy = sympy
        return self._sympy

    @property
    def type_as_str(self) -> str:
        return 'term' if isinstance(self.sympy, Mul) else 'obj'

    @cached_property
    def objects(self) -> tuple['Obj']:
        """Returns all objects the term contains."""
        return tuple(Obj(self, i) for i in range(len(self)))

    @property
    def tensors(self) -> tuple['Obj']:
        """Returns all tensor objects in the term."""
        return tuple(o for o in self.objects
                     if isinstance(o.base, SymbolicTensor))

    @property
    def deltas(self) -> tuple['Obj']:
        """Returns all delta objects of the term."""
        return tuple(o for o in self.objects
                     if isinstance(o.base, KroneckerDelta))

    @property
    def polynoms(self) -> tuple['Polynom']:
        """Returns all polynoms contained in the term."""
        return tuple(o for o in self.objects if isinstance(o, Polynom))

    @cached_property
    def order(self) -> int:
        """Returns the perturbation theoretical order of the term."""
        return sum(t.order for t in self.tensors)

    @cached_property
    def memory_requirements(self):
        """Returns the memory requirements of the largest object of the term.
        """
        from .generate_code import mem_scaling

        mem = []
        for o in self.objects:
            space = o.space
            mem.append(mem_scaling(
                len(space), space.count('g'), space.count('v'),
                space.count('o')
            ))
        return max(mem)

    def make_real(self, return_sympy=False):
        """
        Represent the tern in a real orbital basis.
        - names of complex conjugate t-amplitudes, for instance t1cc -> t1
        - adds bra-ket-symmetry to the fock matrix and the ERI.

        Parameters
        ----------
        return_sympy : bool, optional
            If this is set no Expr object will be returned but the raw
            unwrapped term.
        """
        real_term = Mul(*(o.make_real(return_sympy=True)
                          for o in self.objects))
        if return_sympy:
            return real_term
        assumptions = self.assumptions
        assumptions['real'] = True
        return Expr(real_term, **assumptions)

    def _apply_tensor_braket_sym(self, return_sympy=False):
        term_with_sym = Mul(*[o._apply_tensor_braket_sym(return_sympy=True)
                              for o in self.objects])
        if return_sympy:
            return term_with_sym
        return Expr(term_with_sym, **self.assumptions)

    def block_diagonalize_fock(self, return_sympy=True):
        """
        Block diagonalize the Fock matrix, i.e. if the term contains a off
        diagonal Fock matrix block (f_ov/f_vo) it is set to 0.
        """
        bl_diag = Mul(*[o.block_diagonalize_fock(return_sympy=True)
                        for o in self.objects])
        if return_sympy:
            return bl_diag
        return Expr(bl_diag, **self.assumptions)

    def diagonalize_fock(self, target: tuple[Index] = None,
                         return_sympy: bool = False):
        """
        Represent the term in the canonical orbital basis, where the
        Fock matrix is diagonal. Because it is not possible to
        determine the target indices in the resulting term according
        to the Einstein sum convention, the current target indices will
        be set manually in the resulting term.

        Parameters
        ----------
        return_sympy : bool, optional
            If this is set no Expr object will be returned but the raw
            unwrapped object.
        """
        if target is None:
            target = self.target
        sub = {}
        diag = 1
        for o in self.objects:
            diag_obj, sub_obj = o.diagonalize_fock(target, return_sympy=True)
            diag *= diag_obj
            if any(k in sub and sub[k] != v for k, v in sub_obj.items()):
                raise NotImplementedError("Did not implement the case of "
                                          "multiple fock matrix elements with "
                                          f"intersecting indices: {self}")
            sub.update(sub_obj)
        # if term is part of a polynom -> return the sub dict and perform the
        # substitution in the polynoms parent term object.
        # provide the target indices to the returned expression, because
        # the current target indices might be lost due to the diagonalization
        if return_sympy:
            if isinstance(self.expr, Expr):
                return diag.subs(order_substitutions(sub))
            else:  # polynom
                return diag, sub
        else:
            assumptions = self.assumptions
            assumptions['target_idx'] = target
            if isinstance(self.expr, Expr):
                return Expr(diag.subs(order_substitutions(sub)), **assumptions)
            else:  # polynom
                return Expr(diag, **assumptions), sub

    def substitute_contracted(self, return_sympy: bool = False,
                              only_build_sub: bool = False):
        """Substitute contracted indices in the term by the lowest available
           indices."""

        # 1) determine the target and contracted indices
        #    and split them according to their space
        #    Don't use atoms to obtain the contracted indices! Atoms is a set
        #    and therefore not sorted -> will produce a random result.
        contracted = {}
        for s in self.contracted:
            if (key := s.space_and_spin) not in contracted:
                contracted[key] = []
            contracted[key].append(s)
        used = {}
        for s in set(self.target):
            if (key := s.space_and_spin) not in used:
                used[key] = set()
            used[key].add(s.name)

        # 3) generate new indices the contracted will be replaced with
        #    and build a substitution dictionary
        #    Don't filter out indices that will not change!
        sub = {}
        for (space, spin), idx_list in contracted.items():
            new_idx = get_lowest_avail_indices(len(idx_list),
                                               used.get((space, spin), []),
                                               space)
            if spin:
                new_idx = get_symbols(new_idx, spin * len(idx_list))
            else:
                new_idx = get_symbols(new_idx)
            sub.update({o: n for o, n in zip(idx_list, new_idx)})
        # 4) apply substitutions while ensuring the substitutions are
        # performed in the correct order
        sub = order_substitutions(sub)

        if only_build_sub:  # only build and return the sub_list
            return sub

        substituted = self.sympy.subs(sub)
        # ensure that the substitutions are valid
        if substituted is S.Zero and self.sympy is not S.Zero:
            raise ValueError(f"Invalid substitutions {sub} for {self}.")

        if return_sympy:
            return substituted
        else:
            return Expr(substituted, **self.assumptions)

    def rename_tensor(self, current, new, return_sympy: bool = False):
        """
        Rename tensors in a terms.

        Parameters
        ----------
        return_sympy : bool, optional
            If this is set no Expr object will be returned but the raw
            unwrapped object.
        """
        renamed = Mul(*(o.rename_tensor(current, new, return_sympy=True)
                        for o in self.objects))
        if return_sympy:
            return renamed
        else:
            return Expr(renamed, **self.assumptions)

    def expand_antisym_eri(self, return_sympy: bool = False):
        """
        Expands the antisymmetric ERI using chemists notation
        <pq||rs> = (pr|qs) - (ps|qr).
        ERI's in chemists notation are by default denoted as 'v'.
        Currently this only works for real orbitals, i.e., for
        symmetric ERI's <pq||rs> = <rs||pq>.
        """
        expanded = Mul(*[o.expand_antisym_eri(return_sympy=True)
                         for o in self.objects])
        if return_sympy:
            return expanded
        assumptions = self.assumptions
        # add chemist notation eri to sym_tensors if necessary
        if Symbol(tensor_names.coulomb) in expanded.atoms(Symbol):
            assumptions['sym_tensors'] = (
                assumptions['sym_tensors'] + (tensor_names.coulomb,)
            )
        return Expr(expanded, **assumptions)

    def factor(self):
        from sympy import factor
        return Expr(factor(self.sympy), **self.assumptions)

    def expand_intermediates(self, target: tuple = None,
                             return_sympy: bool = False):
        """Expand all known intermediates in the term."""
        if target is None:
            target = self.target
        expanded = Mul(*[o.expand_intermediates(target, return_sympy=True)
                         for o in self.objects])
        if return_sympy:
            return expanded
        else:
            assumptions = self.assumptions
            assumptions['target_idx'] = target
            return Expr(expanded, **assumptions)

    @cached_member
    def symmetry(self, only_contracted: bool = False,
                 only_target: bool = False) -> dict[tuple, int]:
        """
        Determines the symmetry of the term with respect to index permutations.
        By default all indices of the term are considered. However, by setting
        either only_contracted or only_target the indices may be restricted to
        the respective subset of indices.
        """
        from itertools import combinations, permutations, chain, product
        from math import factorial
        from .indices import split_idx_string
        from .symmetry import Permutation, PermutationProduct

        def permute_str(string, *perms):
            string = split_idx_string(string)
            for perm in perms:
                p, q = [s.name for s in perm]
                sub = {p: q, q: p}
                string = [sub.get(s, s) for s in string]
            return "".join(string)

        def get_perms(*space_perms):
            for perms in chain.from_iterable(space_perms):
                yield perms
            if len(space_perms) > 1:  # form the product
                for perm_tpl in product(*space_perms):
                    yield PermutationProduct(chain.from_iterable(perm_tpl))

        if only_contracted and only_target:
            raise Inputerror("Can not set only_contracted and only_target "
                             "simultaneously.")
        if self.sympy.is_number or isinstance(self.sympy, NonSymmetricTensor):
            return {}  # in both cases we can't find any symmetry

        if only_contracted:
            indices = self.contracted
        elif only_target:
            indices = self.target
        else:
            indices = self.idx

        if len(indices) < 2:  # not enough indices for any permutations
            return {}

        # split in occ and virt indices to only generate P_oo, P_vv and P_gg.
        # Similarly, the spin has to be the same!
        sorted_idx = {}
        for s in indices:
            if (key := s.space_and_spin) not in sorted_idx:
                sorted_idx[key] = []
            sorted_idx[key].append(s)

        space_perms: list[list] = []  # find all permutations within a space
        for idx_list in sorted_idx.values():
            if len(idx_list) < 2:
                continue
            max_n_perms = factorial(len(idx_list))
            # generate idx string that will also be permuted to avoid
            # redundant permutations
            idx_string = "".join([s.name for s in idx_list])
            permuted_str = [idx_string]
            # form all index pairs - all permutations operators
            pairs = [Permutation(*pair) for pair in combinations(idx_list, 2)]
            # form all combinations of permutation operators
            combs = chain.from_iterable(
                permutations(pairs, n) for n in range(1, len(idx_list))
            )
            # remove redundant combinations
            temp = []
            for perms in combs:
                if len(permuted_str) == max_n_perms:
                    break  # did find enough permutations
                perm_str = permute_str(idx_string, *perms)
                if perm_str in permuted_str:  # is the perm redundant?
                    continue
                permuted_str.append(perm_str)
                temp.append(perms)
            space_perms.append(temp)
        # now apply all found perms to the term and determine the symmetry
        # -> add/subtract permuted versions of the term and see if we get 0
        symmetry: dict[tuple, int] = {}
        original_term = self.sympy
        for perms in get_perms(*space_perms):
            permuted = self.permute(*perms).sympy
            if original_term + permuted is S.Zero:
                symmetry[perms] = -1
            elif original_term - permuted is S.Zero:
                symmetry[perms] = +1
        return symmetry

    @property
    def symmetrize(self):
        """
        Symmetrise the term by applying all found symmetries to the term that
        only involve contracted indices and adding up the normalized result.
        """
        from sympy import Rational
        symmetry = self.symmetry(only_contracted=True)
        res = self.sympy
        for perm, factor in symmetry.items():
            res += self.permute(*perm).sympy * factor
        # renormalize the term
        res *= Rational(1, len(symmetry) + 1)
        return Expr(res.expand(), **self.assumptions)

    @property
    def contracted(self) -> tuple[Index]:
        """
        Returns all contracted indices of the term. If no target indices
        have been provided to the parent expression, the Einstein sum
        convention will be applied.
        """

        # target indices have been provided -> no need to count indices
        if (target := self.provided_target_idx) is not None:
            return tuple(s for s, _ in self._idx_counter if s not in target)
        else:  # count indices to determine target and contracted indices
            return tuple(s for s, n in self._idx_counter if n)

    @property
    def target(self) -> tuple[Index]:
        """
        Returns all target indices of the term. If no target indices have been
        provided to the parent expression, the Einstein sum convention will
        be applied.
        """
        # dont cache target and contracted to allow them to react to
        # modifications of the assumptions

        if (target := self.provided_target_idx) is not None:
            return target
        else:
            return tuple(s for s, n in self._idx_counter if not n)

    @cached_property
    def idx(self) -> tuple[Index]:
        """
        Returns all indices that occur in the term. Indices that occur multiple
        times will be listed multiple times.
        """
        return tuple(s for s, n in self._idx_counter for _ in range(n + 1))

    @cached_property
    def _idx_counter(self) -> tuple:
        idx = {}
        for o in self.objects:
            n = abs(o.exponent)  # abs value for denominators
            for s in o.idx:
                if s in idx:
                    idx[s] += n
                else:  # start counting at 0
                    idx[s] = n - 1
        return tuple(sorted(idx.items(),
                            key=lambda tpl: sort_idx_canonical(tpl[0])))

    @cached_property
    def prefactor(self):
        """Returns the prefactor of the term."""
        return nsimplify(
            Mul(*(o.sympy for o in self.objects if o.sympy.is_number)),
            rational=True
        )

    @property
    def sign(self):
        """Returns the sign of the term."""
        return "minus" if self.prefactor < 0 else "plus"

    @cached_property
    def pattern(self) -> dict:
        """
        Determins the pattern of the indices in the term. This is a (kind of)
        readable string hash for each index that is based upon the positions
        the index appears and the coupling of the objects.
        """

        coupl = self.coupling()
        pattern = {}
        for i, o in enumerate(self.objects):
            positions = o.crude_pos()
            c = f"_{'_'.join(sorted(coupl[i]))}" if i in coupl else None
            for s, pos in positions.items():
                key = s.space_and_spin
                if key not in pattern:
                    pattern[key] = {}
                if s not in pattern[key]:
                    pattern[key][s] = []
                if c is None:
                    pattern[key][s].extend((p for p in pos))
                else:
                    pattern[key][s].extend((p + c for p in pos))
        # sort pattern to allow for direct comparison
        for ov, idx_pat in pattern.items():
            for s, pat in idx_pat.items():
                pattern[ov][s] = sorted(pat)
        return pattern

    @cached_member
    def coupling(self, include_target_idx: bool = True,
                 include_exponent: bool = True) -> dict[int, list]:
        """
        Returns the coupling between the objects in the term, where two objects
        are coupled when they share common indices. Only the coupling of non
        unique objects is returned, i.e., the coupling of e.g. a t2_1 amplitude
        is only returned if there is another one in the same term."""
        from collections import Counter
        # 1) collect all the couplings (e.g. if a index s occurs at two tensors
        #    t and V: the crude_pos of s at t will be extended by the crude_pos
        #    of s at V. And vice versa for V.)
        objects: tuple[Obj] = self.objects
        descriptions = [o.description(include_exponent=include_exponent,
                                      include_target_idx=include_target_idx)
                        for o in objects]
        descr_counter = Counter(descriptions)
        positions = [o.crude_pos(include_exponent=include_exponent,
                                 include_target_idx=include_target_idx)
                     for o in objects]
        coupling = {}
        for i, (descr, idx_pos) in enumerate(zip(descriptions, positions)):
            # if the tensor is unique in the term -> no coupling necessary
            if descr_counter[descr] < 2:
                continue
            for other_i, other_idx_pos in enumerate(positions):
                if i == other_i:
                    continue
                matches = [idx for idx in idx_pos if idx in other_idx_pos]
                if not matches:
                    continue
                if i not in coupling:
                    coupling[i] = []
                coupling[i].extend(
                    [p for s in matches for p in other_idx_pos[s]]
                )
        return coupling

    def use_symbolic_denominators(self) -> 'Expr':
        """
        Replace all orbital energy denominators in the expression by tensors,
        e.g., (e_a + e_b - e_i - e_j)^{-1} will be replaced by D^{ab}_{ij},
        where D is a SymmetricTensor."""
        from .eri_orbenergy import EriOrbenergy

        term = EriOrbenergy(self)
        symbolic_denom = term.symbolic_denominator()
        # symbolic denom might additionaly have D set as antisym tensor
        return symbolic_denom * term.pref * term.num.sympy * term.eri.sympy

    def use_explicit_denominators(self, return_sympy: bool = False):
        """
        Switch to an explicit representation of orbital energy denominators by
        replacing all symbolic denominators by their explicit counter part,
        i.e., D^{ij}_{ab} -> (e_i + e_j - e_a - e_b)^{-1}.
        """
        explicit_denom = Mul(*[obj.use_explicit_denominators(return_sympy=True)
                               for obj in self.objects])
        if return_sympy:
            return explicit_denom
        assumptions = self.assumptions
        # remove the tensor from the assumptions
        if tensor_names.sym_orb_denom in self.antisym_tensors:
            assumptions["antisym_tensors"] = tuple(
                n for n in assumptions["antisym_tensors"]
                if n != tensor_names.sym_orb_denom
            )
        return Expr(explicit_denom, **assumptions)

    def split_orb_energy(self) -> dict:
        """
        Splits the term in a orbital energy fraction and a remainder, e.g.
        (e_i + e_j) / (e_i + e_j - e_a - e_b) * (tensor1 * tensor2).
        To this end all polynoms that only contain 'e' tensors are collected to
        form the numerator and denominator, while the rest of the term is
        collected in the remainder. Prefactors are collected in the numerator.
        """

        assumptions = self.assumptions
        assumptions['target_idx'] = self.target
        ret = {"num": Expr(1, **assumptions),
               'denom': Expr(1, **assumptions),
               'remainder': Expr(1, **assumptions)}
        for o in self.objects:
            base, exponent = o.base_and_exponent
            if o.sympy.is_number:
                key = "num"
            elif o.contains_only_orb_energies:
                key = "denom" if exponent < 0 else "num"
            else:
                key = 'remainder'
            ret[key] *= Pow(base, abs(exponent))
        return ret

    @property
    def contains_only_orb_energies(self):
        """Whether the term only contains orbital energies."""
        return all(o.contains_only_orb_energies for o in self.objects
                   if not o.sympy.is_number)

    def to_latex_str(self, only_pull_out_pref: bool = False,
                     spin_as_overbar: bool = False):
        """
        Transforms the term to a latex string.

        Parameters
        ----------
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
        # - sign and prefactor
        pref = self.prefactor
        tex_str = "+ " if pref >= 0 else "- "
        # term only consists of a number (only pref)
        if self.sympy.is_number:
            return tex_str + f"{latex(abs(pref))}"
        # avoid printing +- 1 prefactors
        if pref not in [+1, -1]:
            tex_str += f"{latex(abs(pref))} "

        # - latex strings for the remaining objects
        tex_str += " ".join(
            [o.to_latex_str(only_pull_out_pref, spin_as_overbar)
             for o in self.objects if not o.sympy.is_number]
        )
        return tex_str

    def optimized_contractions(self, target_indices: str = None,
                               max_tensor_dim: int = None):
        """
        Determine the contraction scheme of the term with the lowest
        computational scaling and the lowest memory requirements.

        Parameters
        ----------
        target_indices: str, optional
            The target indices of the term. If not provided the target indices
            of the term will be used after bringing them in canonical order.
        max_tensor_dim: int, optional
            Upper bound for the dimensionality of intermediate results.
        """
        from .generate_code import scaling, contraction_data, mem_scaling
        from collections import Counter
        from itertools import combinations

        def extract_data(o):
            if isinstance(o, Obj):
                return o.idx, o.longname()
            elif isinstance(o, contraction_data):
                return o.target, 'contraction'
            else:
                raise TypeError(f"Can not extract idx and name from {o}")

        def contraction(ob: Obj | contraction_data, i: int | tuple[int],
                        other_ob: Obj | contraction_data,
                        other_i: int | tuple[int], target_indices: list,
                        canonical_target: tuple, max_tensor_dim: int = None):
            """Generate a contraction between two objects or contractions."""

            # extract names and indices of the objects
            indices, name = extract_data(ob)
            other_indices, other_name = extract_data(other_ob)

            # determine target and contracted indices
            idx, other_idx = set(indices), set(other_indices)
            contracted = idx.intersection(other_idx)
            target = idx ^ other_idx  # bitwise XOR

            # if it is not possible to determine the target indices
            # according to the einstein sum_convention it is possible
            # that target indices occur in contracted
            # -> remove them from contracted and add them to target
            if any(i in target_indices for i in contracted):
                target.update({i for i in contracted if i in target_indices})
                contracted = {i for i in contracted if i not in target_indices}

            # check that the result of the contraction does not violate
            # the max_tensor_dim
            if max_tensor_dim is not None and len(target) > max_tensor_dim:
                return None

            # determine the scaling of the contraction
            target_sp = Counter([s.space[0] for s in target])
            contracted_sp = \
                Counter([s.space[0] for s in contracted])

            occ = target_sp['o'] + contracted_sp['o']
            virt = target_sp['v'] + contracted_sp['v']
            general = target_sp['g'] + contracted_sp['g']
            total = occ + virt + general
            # mem = (total, general, virt, occ)
            mem = mem_scaling(len(target), target_sp['g'], target_sp['v'],
                              target_sp['o'])
            scal = scaling(total, general, virt, occ, mem)

            # sort contracted and target indices canonical and store as tuple
            contracted = tuple(sorted(contracted, key=sort_idx_canonical))
            target = tuple(sorted(target, key=sort_idx_canonical))

            # is it the outermost contraction?
            # rather use the provided target indices -> correct order
            if canonical_target == target:
                target = tuple(target_indices)

            return contraction_data((i, other_i), (indices, other_indices),
                                    (name, other_name), contracted, target,
                                    scal)

        def term_contraction(objects: dict, target_indices, canonical_target,
                             max_tensor_dim=None) -> list[list]:
            """Evaluate all possible contraction variants for the term by
               calling this function recursively."""

            if len(objects) < 2:
                raise ValueError("Need at least two objects to determine a "
                                 "contration.")

            ret = []
            for (i, ob), (other_i, other_ob) in \
                    combinations(objects.items(), 2):
                c_data = contraction(ob, i, other_ob, other_i, target_indices,
                                     canonical_target, max_tensor_dim)
                if c_data is None:  # result violates max_tensor_dim
                    continue

                if len(objects) == 2:  # done after this contraction
                    ret.append([c_data])
                else:  # recurse to cover all possible variants
                    # replace the 2 objects by the contraction
                    remaining = {key: val for key, val in objects.items()
                                 if key not in [i, other_i]}
                    remaining[(i, other_i)] = c_data
                    for variant in term_contraction(remaining, target_indices,
                                                    canonical_target,
                                                    max_tensor_dim):
                        # add the current contraction to the list of
                        # contractions found for the remaining objects
                        variant.insert(0, c_data)
                        ret.append(variant)
            return ret

        relevant_objects = {}
        idx_occurences = {}
        n = 0
        for o in self.objects:
            base, exp = o.base_and_exponent
            if o.sympy.is_number or isinstance(base, Symbol):  # skip numbers
                continue
            # antisym-/sym-, nonsymtensors, amplitudes, deltas
            if isinstance(base, (SymbolicTensor, KroneckerDelta)):
                if exp < 0:
                    raise NotImplementedError("Contractions for divisions not "
                                              f"implemented: {self}.")
                # count on how many objects (with exponent 1) an index occures
                for idx in o.idx:
                    if idx not in idx_occurences:
                        idx_occurences[idx] = 0
                    idx_occurences[idx] += exp

                for i in range(n, exp+n):
                    relevant_objects[i] = o
                    n += 1
            else:  # polynom / create / annihilate / NormalOrdered
                raise NotImplementedError("Contractions not implemented for "
                                          "polynoms, creation and annihilation"
                                          f" operators: {self}.")

        # check that no index occure on more than 2 objects
        if any(n > 2 for n in idx_occurences.values()):
            raise Inputerror("Can only optimize contractions for terms where "
                             "each index occures at most on 2 objects. "
                             f"Found: {self}")

        # use the canonical target indices of the term
        if target_indices is None:
            target_indices = self.target  # already sorted canonical
            canonical_target = tuple(target_indices)
        else:  # or transform the provided target indices to sympy symbols
            target_indices = tuple(get_symbols(target_indices))
            canonical_target = tuple(sorted(target_indices,
                                            key=sort_idx_canonical))

        if len(relevant_objects) == 0:
            return []
        elif len(relevant_objects) == 1:  # only a single tensor
            i, o = next(iter(relevant_objects.items()))
            indices = o.idx
            target_sp = Counter(s.space[0] for s in target_indices)
            mem = mem_scaling(len(target_indices), target_sp['g'],
                              target_sp['v'], target_sp['o'])
            scal = scaling(sum(target_sp.values()), target_sp['g'],
                           target_sp['v'], target_sp['o'], mem)
            # no contraction, transpose might be possible
            if set(indices) == set(target_indices):
                contracted = tuple()
            else:  # contraction -> trace is possible
                contracted = tuple(s for s, n in Counter(indices).items()
                                   if n > 1)
            return [contraction_data((i,), (indices,), (o.longname(),),
                                     contracted, target_indices, scal)]

        if max_tensor_dim is not None and not isinstance(max_tensor_dim, int):
            raise Inputerror(f"Invalid max_tensor_dim {max_tensor_dim}.")

        contraction_variants = term_contraction(relevant_objects,
                                                target_indices,
                                                canonical_target,
                                                max_tensor_dim)
        # find the variant with the lowest computational scaling:
        # 1) Maximum: Total scaling
        # 2) Sum of all total scalings  (N^5 + N^3 < N^5 + N^5)
        # 3) Maximum: General scaling (just for generality of the function)
        # 4) Maximum: virt scaling
        # 5) Maximum: occ scaling
        max_scalings = []
        for variant in contraction_variants:
            # the max_dim for intermediates leads to incomplete contraction
            # variants (not all objects could be contracted successfully)
            if len(variant) < len(relevant_objects) - 1:
                continue
            tot_scaling_sum = sum(contr.scaling.total for contr in variant)
            max_scal = list(max(contr.scaling for contr in variant))
            max_scal.insert(1, tot_scaling_sum)
            max_scalings.append(max_scal)
        if not max_scalings:
            raise RuntimeError("Could not find a valid contraction scheme for "
                               f"{self} while restricting the maximum tensor "
                               f"dimension to {max_tensor_dim}.")
        variant, _ = min(
            zip(contraction_variants, max_scalings), key=lambda tpl: tpl[1]
        )
        return variant


class Obj(Container):
    """
    Wrapper for a single Object, e.g., a tensor that is part of a term wrapped
    by the Term class.

    Parameters
    ----------
    t : Term
        The 'Term' instance the object is contained in.
    pos : int
        The index of the object in the term (sympy.Mul object).
    """
    def __new__(cls, t, pos=None, **assumptions):
        types = {
            NO: lambda o: 'no',
            Pow: lambda o: 'polynom' if isinstance(o.args[0], Add) else 'obj',
            # (a + b)^1 is a Add object that is part of a Mul object
            Add: lambda o: 'polynom'
        }
        if isinstance(t, (Term, NormalOrdered)):
            if not isinstance(pos, int):
                raise Inputerror('Position needs to be provided as int.')
            o = t.sympy if len(t) == 1 else t.args[pos]
            obj_type = types.get(type(o), lambda x: 'obj')(o)
            if obj_type == 'obj':
                return super().__new__(cls)
            elif obj_type == 'no':
                return NormalOrdered(t, pos=pos)
            else:
                return Polynom(t, pos=pos)
        else:
            return Expr(t, **assumptions)

    def __init__(self, t, pos=None, **assumptions):
        self._expr: Expr = t.expr
        self._term: Term = t
        self._pos: int = pos
        self._sympy = None

    def __str__(self):
        return latex(self.sympy)

    def __getattr__(self, attr):
        return getattr(self.sympy, attr)

    @property
    def expr(self) -> Expr:
        return self._expr

    @property
    def term(self) -> Term:
        return self._term

    @property
    def assumptions(self) -> dict:
        return self.expr.assumptions

    @property
    def real(self) -> bool:
        return self.expr.real

    @property
    def sym_tensors(self) -> tuple:
        return self.expr.sym_tensors

    @property
    def antisym_tensors(self) -> tuple:
        return self.expr.antisym_tensors

    @property
    def provided_target_idx(self) -> None | tuple:
        return self.expr.provided_target_idx

    @property
    def sympy(self):
        if self._sympy is not None:
            return self._sympy

        if len(self.term) == 1:
            return self.term.sympy
        else:
            return self.term.args[self._pos]

    def make_real(self, return_sympy: bool = False):
        """
        Represent the object in a real orbital basis by renaming the
        complex conjugate t-amplitudes, for instance t1cc -> t1.

        Parameters
        ----------
        return_sympy : bool, optional
            If this is set no Expr object will be returned but the raw
            unwrapped object.
        """

        if self.is_t_amplitude:
            old = self.name
            base_name, ext = split_t_amplitude_name(old)
            new = f"{base_name}{ext.replace('c', '')}"
            if old == new:
                real_obj = self.sympy
            else:
                base, exponent = self.base_and_exponent
                real_obj = Pow(
                    Amplitude(new, base.upper, base.lower, base.bra_ket_sym),
                    exponent
                )
        else:
            real_obj = self.sympy
        if return_sympy:
            return real_obj
        assumptions = self.assumptions
        assumptions['real'] = True
        return Expr(real_obj, **assumptions)

    def _apply_tensor_braket_sym(self, return_sympy: bool = False):
        # antisymtensor, symtensor or amplitude
        base, exponent = self.base_and_exponent
        if isinstance(base, AntiSymmetricTensor):
            bra_ket_sym = None
            if (name := base.name) in self.sym_tensors and \
                    base.bra_ket_sym is not S.One:
                bra_ket_sym = 1
            elif name in self.antisym_tensors and \
                    base.bra_ket_sym is not S.NegativeOne:
                bra_ket_sym = -1
            if bra_ket_sym is None:
                obj_with_sym = self.sympy
            else:
                obj_with_sym = Pow(base.add_bra_ket_sym(bra_ket_sym),
                                   exponent)
        else:
            obj_with_sym = self.sympy
        if return_sympy:
            return obj_with_sym
        return Expr(obj_with_sym, **self.assumptions)

    def block_diagonalize_fock(self, return_sympy: bool = False):
        """
        Block diagonalize the Fock matrix, i.e. if the object is part of an
        off-diagonal fock matrix block, it is set to 0.
        Parameters
        ----------
        return_sympy : bool, optional
            If this is set no Expr object will be returned but the raw
            unwrapped object.
        """
        if self.name == tensor_names.fock and self.space in ['ov', 'vo']:
            bl_diag = 0
        else:
            bl_diag = self.sympy
        if return_sympy:
            return bl_diag
        return Expr(bl_diag, **self.assumptions)

    def diagonalize_fock(self, target: tuple[Index] = None,
                         return_sympy: bool = False):
        sub = {}
        if self.name == tensor_names.fock:  # self contains a fock element
            # off diagonal fock matrix block -> return 0
            if self.space in ['ov', 'vo']:
                diag = 0
            else:  # diagonal fock block
                if target is None:
                    target = self.term.target
                idx = self.idx  # 0 is preferred, 1 is killable
                if len(idx) != 2:
                    raise RuntimeError(f"found fock matrix element {self} that"
                                       " does not hold exactly 2 indices.")
                # don't touch:
                #  - diagonal fock elements (for not loosing
                #    a contracted index by accident)
                #  - fock elements with both indices being target indices
                #    (for not loosing a target index in the term)
                if idx[0] is idx[1] or all(s in target for s in idx):
                    diag = self.sympy
                else:  # we can diagonalize the fock matrix element safely
                    # killable is contracted -> kill
                    if idx[1] not in target:
                        sub[idx[1]] = idx[0]
                        preferred = idx[0]
                    # preferred is contracted (and not killable) -> kill
                    elif idx[0] not in target:
                        sub[idx[0]] = idx[1]
                        preferred = idx[1]
                    # construct a orbital energy e with the preferred idx
                    diag = Pow(
                        NonSymmetricTensor(tensor_names.orb_energy,
                                           (preferred,)),
                        self.exponent
                    )
        else:  # no fock matrix element
            diag = self.sympy

        if return_sympy:
            return diag, sub
        else:
            assumptions = self.assumptions
            assumptions['target_idx'] = target
            return Expr(diag, **assumptions), sub

    def rename_tensor(self, current: str, new: str,
                      return_sympy: bool = False):
        """Renames a tensor object."""
        base, exponent = self.base_and_exponent
        if isinstance(base, SymbolicTensor) and base.name == current:
            if isinstance(base, AntiSymmetricTensor):
                args = (new, base.upper, base.lower, base.bra_ket_sym)
            elif isinstance(base, NonSymmetricTensor):
                args = (new, base.indices)
            else:
                raise TypeError(f"Unknown tensor type {type(base)}.")
            base = base.__class__(*args)
            new_obj = Pow(base, exponent)
        else:  # delta / create / annihilate / pref
            new_obj = self.sympy
        if return_sympy:
            return new_obj
        else:
            return Expr(new_obj, **self.assumptions)

    def expand_antisym_eri(self, return_sympy: bool = True):
        """
        Expands the antisymmetric ERI using chemists notation
        <pq||rs> = (pr|qs) - (ps|qr).
        ERI's in chemists notation are by default denoted as 'v'.
        Currently this only works for real orbitals, i.e.,
        for symmetric ERI's <pq||rs> = <rs||pq>.
        """
        expanded_coulomb = False
        if self.name == tensor_names.eri:
            # ensure that the eri is Symmetric. Otherwise we would introduce
            # additional unwanted symmetry in the result
            if self.bra_ket_sym != 1:
                raise NotImplementedError("Can only expand antisymmetric ERI "
                                          "in a real orbital basis.")
            p, q, r, s = self.idx  # <pq||rs>
            res = S.Zero
            if p.spin == r.spin and q.spin == s.spin:
                res += SymmetricTensor(tensor_names.coulomb, (p, r), (q, s), 1)
                expanded_coulomb = True
            if p.spin == s.spin and q.spin == r.spin:
                res -= SymmetricTensor(tensor_names.coulomb, (p, s), (q, r), 1)
                expanded_coulomb = True
            res = Pow(res, self.exponent)
        else:  # nothing to do
            res = self.sympy

        if return_sympy:
            return res
        assumptions = self.assumptions
        # add the coulomb integral to sym_tensors if necessary
        if expanded_coulomb:
            assumptions['sym_tensors'] = (
                assumptions['sym_tensors'] + (tensor_names.coulomb,)
            )
        return Expr(res, **assumptions)

    @property
    def base_and_exponent(self) -> tuple:
        """Return base and exponent of the object."""
        base = self.sympy
        if isinstance(base, Pow):
            return base.args
        else:
            return base, 1

    @property
    def base(self):
        """Returns the base (base^exponent) of the object."""
        base = self.sympy
        if isinstance(base, Pow):
            return base.args[0]
        else:
            return base

    @property
    def exponent(self):
        """Returns the exponent of the object."""
        return self.sympy.args[1] if isinstance(self.sympy, Pow) else 1

    @property
    def type_as_str(self) -> str:
        """Returns a string that describes the type of the object."""
        if self.sympy.is_number:
            return "prefactor"
        obj = self.base
        if isinstance(obj, Amplitude):
            return "amplitude"
        elif isinstance(obj, SymmetricTensor):
            return "symtensor"
        elif isinstance(obj, AntiSymmetricTensor):
            return "antisymtensor"
        elif isinstance(obj, NonSymmetricTensor):
            return "nonsymtensor"
        elif isinstance(obj, KroneckerDelta):
            return "delta"
        elif isinstance(obj, F):
            return "annihilate"
        elif isinstance(obj, Fd):
            return "create"
        elif isinstance(obj, Symbol):
            return "symbol"
        else:
            raise TypeError(f"Unknown object {self} of type {type(obj)}.")

    @property
    def name(self) -> str | None:
        """Extract the name of tensor objects."""
        if isinstance(self.base, SymbolicTensor):
            return self.base.name

    @property
    def is_t_amplitude(self) -> bool:
        """Whether the object is a ground state t-amplitude."""
        name = self.name
        return False if name is None else is_t_amplitude(name)

    @property
    def is_gs_density(self) -> bool:
        """Check whether the object is a ground state density tensor."""
        name = self.name
        return False if name is None else is_gs_density(name)

    @cached_property
    def order(self):
        """Returns the perturbation theoretical order of the obj."""
        from .intermediates import Intermediates

        if isinstance(self.base, SymbolicTensor):
            if (name := self.name) == tensor_names.eri:  # eri
                return 1
            elif is_t_amplitude(name):
                _, ext = split_t_amplitude_name(name)
                return int(ext.replace('c', ''))
            # all intermediates
            itmd_cls = Intermediates().available.get(self.longname(True), None)
            if itmd_cls is not None:
                return itmd_cls.order
        return 0

    def longname(self, use_default_names: bool = False):
        """
        Returns a more exhaustive name of the object. Used for intermediates
        and transformation to code.

        Parameters
        ----------
        use_default_names: bool, optional
            If set, the default names are used to generate the longname.
            This is necessary to e.g., map a tensor name to an intermediate
            name, since they are defined using the default names.
            (default: False)
        """
        name = None
        base = self.base
        if isinstance(base, SymbolicTensor):
            name = base.name
            # t-amplitudes
            if is_t_amplitude(name):
                if len(base.upper) != len(base.lower):
                    raise RuntimeError("Number of upper and lower indices not "
                                       f"equal for t-amplitude {self}.")
                base_name, ext = split_t_amplitude_name(name)
                if use_default_names:
                    base_name = tensor_names.defaults().get("gs_amplitude")
                    assert base_name is not None
                name = f"{base_name}{len(base.upper)}_{ext}"
            elif is_adc_amplitude(name):  # adc amplitudes
                # need to determine the excitation space as int
                space = self.space
                n_o, n_v = space.count('o'), space.count('v')
                if n_o == n_v:  # pp-ADC
                    n = n_o  # p-h -> 1 // 2p-2h -> 2 etc.
                else:  # ip-/ea-/dip-/dea-ADC
                    n = min([n_o, n_v]) + 1  # h -> 1 / 2h -> 1 / p-2h -> 2...
                lr = "l" if name == tensor_names.left_adc_amplitude else 'r'
                name = f"u{lr}{n}"
            elif is_gs_density(name):  # mp densities
                if len(base.upper) != len(base.lower):
                    raise RuntimeError("Number of upper and lower indices not "
                                       f"equal for mp density {self}.")
                base_name, ext = split_gs_density_name(name)
                if use_default_names:
                    base_name = tensor_names.defaults().get("gs_density")
                    assert base_name is not None
                name = f"{base_name}0_{ext}_{self.space}"
            elif name.startswith('t2eri'):  # t2eri
                name = f"t2eri_{name[5:]}"
            elif name == 't2sq':
                pass
            else:  # arbitrary other tensor
                name += f"_{self.space}"
        elif isinstance(base, KroneckerDelta):  # deltas -> d_oo / d_vv
            name = f"d_{self.space}"
        return name

    @property
    def bra_ket_sym(self) -> int | None:
        """Returns the bra-ket-symmetry of the object."""
        # nonsym_tensors are assumed to not have any symmetry atm!.
        if isinstance(self.base, AntiSymmetricTensor):
            return self.base.bra_ket_sym

    def symmetry(self, only_contracted: bool = False,
                 only_target: bool = False) -> dict[tuple, int]:
        """Determines the symmetry of the obj."""
        if only_contracted and only_target:
            raise Inputerror("only_contracted and only_target can not be set "
                             "simultaneously.")
        if self.sympy.is_number or isinstance(self.sympy, NonSymmetricTensor):
            return {}  # can not find any symmetry in both cases
        # obtain the desired indices
        if only_contracted:
            indices = self.term.contracted
        elif only_target:
            indices = self.term.target
        else:
            indices = self.idx
        assumptions = self.assumptions
        assumptions['target_idx'] = indices
        # create a term obj and use the appropriate indices as target_idx
        new_expr = Expr(self.sympy, **assumptions)
        return new_expr.terms[0].symmetry(only_target=True)

    @cached_property
    def idx(self) -> tuple[Index]:
        """Return the indices of the object."""

        if self.sympy.is_number:  # prefactor
            return tuple()

        obj = self.base
        # Antisym-, Sym-, Nonsymtensor, Amplitude, Kroneckerdelta
        if isinstance(obj, (SymbolicTensor, KroneckerDelta)):
            return obj.idx
        elif isinstance(obj, FermionicOperator):  # F and Fd
            return obj.args
        elif isinstance(obj, Symbol):  # a symbol without indices
            return tuple()
        else:
            raise TypeError("Can not determine the indices for an obj of type"
                            f"{type(obj)}: {self}.")

    @property
    def space(self) -> str:
        """Returns the index space (tensor block) of the object."""
        return "".join(s.space[0] for s in self.idx)

    @property
    def spin(self) -> str:
        """Returns the spin block of the current object."""
        return "".join(s.spin if s.spin else "n" for s in self.idx)

    @cached_member
    def crude_pos(self, include_target_idx: bool = True,
                  include_exponent: bool = True) -> dict[Index, list]:
        """Returns the 'crude' position of the indices in the object.
           (e.g. only if they are located in bra/ket, not the exact position)
           """
        if not self.idx:  # just a number (prefactor or symbol)
            return {}

        if include_target_idx:
            target = self.term.target

        ret = {}
        description = self.description(include_exponent=include_exponent,
                                       include_target_idx=include_target_idx)
        obj = self.base
        # antisym-, symtensor and amplitude
        if isinstance(obj, AntiSymmetricTensor):
            tensor = self.base
            for uplo, idx_tpl in \
                    {'u': tensor.upper, 'l': tensor.lower}.items():
                for s in idx_tpl:
                    # space (upper/lower) in which the tensor occurs
                    if tensor.bra_ket_sym is S.Zero:
                        pos = f"{description}-{uplo}"
                    else:
                        pos = description
                    # space (occ/virt) of neighbour indices
                    neighbours = [i for i in idx_tpl if i is not s]
                    if neighbours:
                        neighbour_data = ''.join(
                            i.space[0] + i.spin for i in neighbours
                        )
                        pos += f"-{neighbour_data}"
                    # names of neighbour target indices
                    if include_target_idx:
                        neighbour_target = [i.name for i in neighbours
                                            if i in target]
                        if neighbour_target:
                            pos += f"-{''.join(neighbour_target)}"
                    if s not in ret:
                        ret[s] = []
                    ret[s].append(pos)
        elif isinstance(obj, NonSymmetricTensor):  # nonsymtensor
            # target idx position is already in the description
            idx = self.idx
            for i, s in enumerate(idx):
                if s not in ret:
                    ret[s] = []
                ret[s].append(f"{description}_{i}")
        # delta, create, annihilate
        elif isinstance(obj, (KroneckerDelta, F, Fd)):
            for s in self.idx:
                if s not in ret:
                    ret[s] = []
                ret[s].append(description)
        return ret

    def expand_intermediates(self, target: tuple = None,
                             return_sympy: bool = False):
        """Expand the object if it is a known intermediate."""
        from .intermediates import Intermediates

        # intermediates only defined for tensors
        if not isinstance(self.base, SymbolicTensor):
            if return_sympy:
                return self.sympy
            else:
                return Expr(self.sympy, **self.assumptions)

        if target is None:
            target = self.term.target

        itmd = Intermediates()
        itmd = itmd.available.get(self.longname(True), None)
        if itmd is None:
            expanded = self.sympy
        else:
            exp = self.exponent
            idx = self.idx
            expanded = 1
            for _ in range(abs(exp)):
                expanded *= itmd.expand_itmd(indices=idx, return_sympy=True)
            if exp < 0:
                expanded = Pow(expanded, -1)

        if return_sympy:
            return expanded
        else:
            assumptions = self.assumptions
            assumptions['target_idx'] = target
            return Expr(expanded, **assumptions)

    @cached_member
    def description(self, include_exponent: bool = True,
                    include_target_idx: bool = True) -> str:
        """A string that describes the object."""

        descr = self.type_as_str
        if descr in ['prefactor', 'symbol']:
            return descr

        if include_target_idx:
            target = self.term.target

        if descr in ['antisymtensor', 'amplitude', 'symtensor']:
            base, exponent = self.base_and_exponent
            # - space separated in upper and lower part
            upper, lower = base.upper, base.lower
            data_u = "".join(s.space[0] + s.spin for s in upper)
            data_l = "".join(s.space[0] + s.spin for s in lower)
            descr += f"-{base.name}-{data_u}-{data_l}"
            # names of target indices, also separated in upper and lower part
            # indices in upper and lower have been sorted upon tensor creation!
            if include_target_idx:
                target_u = "".join(s.name for s in upper if s in target)
                target_l = "".join(s.name for s in lower if s in target)
                if target_l or target_u:  # we have at least 1 target idx
                    if base.bra_ket_sym is S.Zero:  # no bra ket symmetry
                        if not target_u:
                            target_u = 'none'
                        if not target_l:
                            target_l = 'none'
                        descr += f"-{target_u}-{target_l}"
                    else:  # bra ket sym or antisym
                        # target indices in both spaces
                        if target_u and target_l:
                            descr += (
                                f"-{'-'.join(sorted([target_u, target_l]))}"
                            )
                        else:  # only in 1 space at least 1 target idx
                            descr += f"-{target_u + target_l}"
            if include_exponent:  # add exponent to description
                descr += f"-{exponent}"
        elif descr == 'nonsymtensor':
            data = "".join(s.space[0] + s.spin for s in self.idx)
            descr += f"-{self.name}-{data}"
            if include_target_idx:
                target_str = "".join(s.name + str(i) for i, s in
                                     enumerate(self.idx) if s in target)
                if target_str:
                    descr += f"-{target_str}"
            if include_exponent:
                descr += f"-{self.exponent}"
        elif descr in ['delta', 'annihilate', 'create']:
            data = "".join(s.space[0] + s.spin for s in self.idx)
            descr += f"-{data}"
            if include_target_idx and \
                    (target_str := "".join(s.name for s in self.idx
                                           if s in target)):
                descr += f"-{target_str}"
            if include_exponent:
                descr += f"-{self.exponent}"
        return descr

    @property
    def allowed_spin_blocks(self) -> tuple[str] | None:
        """Returns the valid spin blocks of the object."""
        from .intermediates import Intermediates
        from itertools import product

        # prefactor or symbol have no indices -> no allowed spin blocks
        if not self.idx:
            return None

        obj = self.base
        # antisym-, sym-, nonsymtensor and amplitude
        if isinstance(obj, SymbolicTensor):
            name = obj.name
            if name == tensor_names.eri:  # hardcode the ERI spin blocks
                return ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
            # t-amplitudes: all spin conserving spin blocks are allowed, i.e.,
            # all blocks with the same amount of alpha and beta indices
            # in upper and lower
            elif is_t_amplitude(name):
                idx = obj.idx
                if len(idx) % 2:
                    raise ValueError("Expected t-amplitude to have the same "
                                     f"of upper and lower indices: {self}.")
                n = len(idx)//2
                return tuple(sorted(
                    ["".join(block) for block in product("ab", repeat=len(idx))
                     if block[:n].count("a") == block[n:].count("a")]
                ))
            elif name == tensor_names.coulomb:  # ERI in chemist notation
                return ("aaaa", "aabb", "bbaa", "bbbb")
        elif isinstance(obj, KroneckerDelta):  # delta
            # spins have to be equal
            return ("aa", "bb")
        elif isinstance(obj, FermionicOperator):  # create / annihilate
            # both spins allowed!
            return ("a", "b")
        # the known allowed spin blocks of eri, t-amplitudes and deltas
        # may be used to generate the spin blocks of other intermediates
        itmd = Intermediates().available.get(self.longname(True), None)
        if itmd is None:
            logger.warning(
                f"Could not determine valid spin blocks for {self}."
            )
            return None
        else:
            return itmd.allowed_spin_blocks

    def use_explicit_denominators(self, return_sympy: bool = False):
        """
        Switch to an explicit representation of orbital energy denominators by
        replacing all symbolic denominators by their explicit counter part,
        i.e., D^{ij}_{ab} -> (e_i + e_j - e_a - e_b)^{-1}.+
        """
        if self.name == tensor_names.sym_orb_denom:
            tensor, exponent = self.base_and_exponent
            # upper indices are added, lower indices subtracted
            explicit_denom = 0
            for s in tensor.upper:
                explicit_denom += NonSymmetricTensor(
                    tensor_names.orb_energy, (s,)
                )
            for s in tensor.lower:
                explicit_denom -= NonSymmetricTensor(
                    tensor_names.orb_energy, (s,)
                )
            explicit_denom = Pow(explicit_denom, -exponent)
        else:
            explicit_denom = self.sympy
        if return_sympy:
            return explicit_denom
        assumptions = self.assumptions
        # remove the symbolic denom from the assumptions if necessary
        if tensor_names.sym_orb_denom in self.antisym_tensors:
            assumptions["antisym_tensors"] = tuple(
                n for n in assumptions["antisym_tensors"]
                if n != tensor_names.sym_orb_denom
            )
        return Expr(explicit_denom, **assumptions)

    @property
    def contains_only_orb_energies(self):
        """Whether the term only contains orbital energies."""
        # all orb energies should be nonsym_tensors actually
        return self.name == tensor_names.orb_energy and len(self.idx) == 1

    def to_latex_str(self, only_pull_out_pref: bool = False,
                     spin_as_overbar: bool = False) -> str:
        """Returns a latex string for the object."""

        def format_indices(indices: tuple[Index]) -> str:
            if indices is None:
                return None
            if spin_as_overbar:
                spins = [s.spin for s in indices]
                if any(spins) and not all(spins):
                    raise ValueError("All indices have to have a spin "
                                     "assigned in order to differentiate "
                                     "indices without spin from indices with "
                                     f"alpha spin: {self}")
                return "".join(
                    f"\\overline{{{i.name}}}" if s == "b" else i.name
                    for i, s in zip(indices, spins)
                )
            else:
                return "".join(latex(i) for i in indices)

        if only_pull_out_pref:  # use sympy latex print
            return self.__str__()

        name = self.name
        obj, exp = self.base_and_exponent
        if isinstance(obj, SymbolicTensor):
            special_tensors = {
                tensor_names.eri: (  # antisym ERI physicist
                    lambda up, lo: f"\\langle {up}\\vert\\vert {lo}\\rangle"
                ),
                tensor_names.fock: (  # fock matrix
                    lambda up, lo: f"{tensor_names.fock}_{{{up}{lo}}}"
                ),
                # coulomb integral chemist notation
                tensor_names.coulomb: lambda up, lo: f"({up}\\vert {lo})",
                # orbital energy
                tensor_names.orb_energy: lambda _, lo: f"\\varepsilon_{{{lo}}}"
            }
            # convert the indices to string
            if isinstance(obj, AntiSymmetricTensor):
                upper = format_indices(obj.upper)
                lower = format_indices(obj.lower)
            elif isinstance(obj, NonSymmetricTensor):
                upper, lower = None, format_indices(obj.indices)
            else:
                raise TypeError(f"Unknown tensor object {obj} of type "
                                f"{type(obj)}")

            if name in special_tensors:
                tex_str = special_tensors[name](upper, lower)
            else:
                order_str = None
                if is_t_amplitude(name):  # mp t-amplitudes
                    base_name, ext = split_t_amplitude_name(name)
                    if "c" in ext:
                        order_str = f"({ext.replace('c', '')})\\ast"
                    else:
                        order_str = f"({ext})"
                    order_str = f"}}^{{{order_str}}}"
                    name = f"{{{base_name}"
                elif is_gs_density(name):  # mp densities
                    _, ext = split_gs_density_name(name)
                    order_str = f"}}^{{({ext})}}"
                    name = "{\\rho"

                tex_str = name
                if upper is not None:
                    tex_str += f"^{{{upper}}}"
                tex_str += f"_{{{lower}}}"

                # append pt order for amplitude and mp densities
                if order_str is not None:
                    tex_str += order_str
        elif isinstance(obj, KroneckerDelta):
            tex_str = f"\\delta_{{{format_indices(obj.idx)}}}"
        elif isinstance(obj, F):  # annihilate
            tex_str = f"a_{{{format_indices(obj.args)}}}"
        elif isinstance(obj, Fd):  # create
            tex_str = f"a^\\dagger_{{{format_indices(obj.args)}}}"
        else:
            return self.__str__()

        if exp != 1:
            # special case for ERI and coulomb
            if name in [tensor_names.eri, tensor_names.coulomb]:
                tex_str += f"^{{{exp}}}"
            else:
                tex_str = f"\\bigl({tex_str}\\bigr)^{{{exp}}}"
        return tex_str


class NormalOrdered(Obj):
    """
    Container for a normal ordered operator string.

    Parameters
    ----------
    t : Term
        The 'Term' instance the operator string is contained in.
    pos : int
        The index of the NO object in the term.
    """
    def __new__(cls, t, pos=None, **assumptions):
        if isinstance(t, Term):
            if not isinstance(pos, int):
                raise Inputerror('Position needs to be provided as int.')
            o = t.sympy if len(t) == 1 else t.args[pos]
            if isinstance(o, NO):
                return object.__new__(cls)
            else:
                raise RuntimeError('Trying to use normal_ordered container'
                                   f'for a non NO object: {o}.')
        else:
            return Expr(t, **assumptions)

    def __len__(self) -> int:
        # a NO obj can only contain a Mul object.
        return len(self.extract_no.args)

    @property
    def args(self) -> tuple:
        return self.extract_no.args

    @cached_property
    def objects(self) -> tuple['Obj']:
        return tuple(Obj(self, i) for i in range(len(self.extract_no.args)))

    @property
    def extract_no(self):
        return self.sympy.args[0]

    @property
    def exponent(self):
        # actually sympy should throw an error if a NO object contains a Pow
        # obj or anything else than a*b*c
        exp = set(o.exponent for o in self.objects)
        if len(exp) == 1:
            return exp.pop()
        else:
            raise NotImplementedError(
                'Exponent only implemented for NO objects, where all '
                f'operators share the same exponent. {self}'
            )

    @property
    def type_as_str(self):
        return 'NormalOrdered'

    @cached_property
    def idx(self) -> tuple[Index]:
        """
        Indices of the normal ordered operator string. Indices that appear
        multiple times will be listed multiple times.
        """
        objects = self.objects
        exp = self.exponent
        ret = tuple(s for o in objects for s in o.idx for _ in range(exp))
        if len(objects) != len(ret):
            raise NotImplementedError('Expected a NO object only to contain'
                                      "second quantized operators with an "
                                      f"exponent of 1. {self}")
        return ret

    @cached_member
    def crude_pos(self, include_target_idx: bool = True,
                  include_exponent: bool = True) -> dict:

        descr = self.description(include_exponent, include_target_idx)
        ret = {}
        for o in self.objects:
            o_descr = o.description(include_exponent=include_exponent,
                                    include_target_idx=include_target_idx)
            for s in o.idx:
                if s not in ret:
                    ret[s] = []
                ret[s].append(f"{descr}_{o_descr}")
        return ret

    @cached_member
    def description(self, include_exponent: bool = True,
                    include_target_idx: bool = True) -> str:

        # exponent has to be 1 for all contained operators
        target = self.term.target
        obj_contribs = []
        for o in self.objects:
            # add either index space or target idx name
            idx = o.idx
            if include_target_idx and idx[0] in target:
                op_str = idx[0].name
            else:
                op_str = idx[0].space[0] + idx[0].spin
            # add a plus for creation operators
            base = o.base
            if isinstance(base, Fd):
                op_str += '+'
            elif not isinstance(base, F):  # has to be annihilate here
                raise TypeError("Unexpected content for NormalOrdered "
                                f"container: {o}, {type(o)}.")
            obj_contribs.append(op_str)
        return f"{self.type_as_str}-{'-'.join(sorted(obj_contribs))}"

    @property
    def allowed_spin_blocks(self) -> tuple[str]:
        from itertools import product
        allowed_blocks = [o.allowed_spin_blocks for o in self.objects]
        return tuple("".join(b) for b in product(*allowed_blocks))

    def to_latex_str(self, only_pull_out_pref: bool = False,
                     spin_as_overbar: bool = False) -> str:
        # no prefs possible in NO
        return " ".join([o.to_latex_str(only_pull_out_pref, spin_as_overbar)
                        for o in self.objects])


class Polynom(Obj):
    """
    Container for a polynom (a+b+c)^x.

    Parameters
    ----------
    t : Term
        The 'Term' instance the polynom is a part of.
    pos : int
        The index of the polynom in the term.
    """
    def __new__(cls, t, pos=None, **assumptions):
        if isinstance(t, Term):
            if not isinstance(pos, int):
                raise Inputerror("Position needs to be provided as int.")
            o = t.sympy if len(t) == 1 else t.args[pos]
            if isinstance(o.args[0], Add) or isinstance(o, Add):
                return object.__new__(cls)
            else:
                raise RuntimeError("Trying to use polynom container for a non"
                                   f"polynom object {o}.")
        else:
            return Expr(t, **assumptions)

    def __len__(self) -> int:
        # has to at least contain 2 terms: a+b
        return len(self.base.args)

    @property
    def args(self):
        return self.base.args

    @cached_property
    def terms(self) -> tuple[Term]:
        # overwriting args allows to pass self to the term instances
        return tuple(Term(self, i) for i in range(len(self)))

    @property
    def type_as_str(self):
        return 'polynom'

    @cached_property
    def idx(self):
        """
        Returns all indices that occur in the polynom. Indices that occur
        multiple times will be listed multiple times.
        """
        idx = [s for t in self.terms for s in t.idx]
        return tuple(sorted(idx, key=sort_idx_canonical))

    def make_real(self, return_sympy: bool = False):
        """
        Represent the polynom in a real orbital basis.
        - names of complex conjugate t-amplitudes, for instance t1cc -> t1
        - adds bra-ket-symmetry to the fock matrix and the ERI.

        Parameters
        ----------
        return_sympy : bool, optional
            If this is set no Expr object will be returned but the raw
            unwrapped object.
        """
        real = Add(*[t.make_real(return_sympy=True) for t in self.terms])
        real = Pow(real, self.exponent)
        if return_sympy:
            return real
        assumptions = self.assumptions
        assumptions['real'] = True
        return Expr(real, **assumptions)

    def _apply_tensor_braket_sym(self, return_sympy: bool = False) -> Expr:
        with_sym = Add(*[t._apply_tensor_braket_sym(return_sympy=True)
                         for t in self.terms])
        with_sym = Pow(with_sym, self.exponent)
        if return_sympy:
            return with_sym
        return Expr(with_sym, **self.assumptions)

    def block_diagonalize_fock(self, return_sympy: bool = False):
        """
        Block diagonalize the fock matrix in the polynom by removing terms
        that contain elements of off-diagonal blocks.
        """
        bl_diag = Add(*(t.block_diagonalize_fock(return_sympy=True)
                        for t in self.terms))
        bl_diag = Pow(bl_diag, self.exponent)
        if return_sympy:
            return bl_diag
        return Expr(bl_diag, **self.assumptions)

    def diagonalize_fock(self, target=None):
        raise NotImplementedError("Fock matrix diagonalization not implemented"
                                  f" for polynoms: {self}")

    def rename_tensor(self, current: str, new: str,
                      return_sympy: bool = False):
        """Rename a tensor from current to new."""
        renamed = Add(*(term.rename_tensor(current, new, return_sympy=True)
                        for term in self.terms))
        renamed = Pow(renamed, self.exponent)
        if return_sympy:
            return renamed
        else:
            return Expr(renamed, **self.assumptions)

    def expand_antisym_eri(self, return_sympy: bool = False):
        """
        Expands the antisymmetric ERI using chemists notation
        <pq||rs> = (pr|qs) - (ps|qr).
        ERI's in chemists notation are by default denoted as 'v'.
        Currently this only works for real orbitals, i.e.,
        for symmetric ERI's <pq||rs> = <rs||pq>.
        """
        expanded = Add(*(term.expand_antisym_eri(return_sympy=True)
                         for term in self.terms))
        expanded = Pow(expanded, self.exponent)
        if return_sympy:
            return expanded
        assumptions = self.assumptions
        # add the coulomb tensor to sym_tensors if necessary
        if Symbol(tensor_names.coulomb) in expanded.atoms(Symbol):
            assumptions['sym_tensors'] = (
                assumptions['sym_tensors'] + (tensor_names.coulomb,)
            )
        return Expr(expanded, **assumptions)

    @property
    def order(self):
        raise NotImplementedError("Order not implemented for polynoms: "
                                  f"{self} in {self.term}")

    def crude_pos(self, *args, **kwargs):
        raise NotImplementedError("crude_pos for determining index positions "
                                  f"not implemented for polynoms: {self} in "
                                  f"{self.term}")

    def expand_intermediates(self, target: tuple = None,
                             return_sympy: bool = False):
        """Expands all known intermediates in the polynom."""
        if target is None:
            target = self.term.target

        expanded = Add(*[t.expand_intermediates(target, return_sympy=True)
                         for t in self.terms])
        expanded = Pow(expanded, self.exponent)
        if return_sympy:
            return expanded
        else:
            assumptions = self.assumptions
            assumptions['target_idx'] = target
            return Expr(expanded, **assumptions)

    def use_explicit_denominators(self, return_sympy: bool = False):
        """
        Switch to an explicit representation of orbital energy denominators by
        replacing all symbolic denominators by their explicit counter part,
        i.e., D^{ij}_{ab} -> (e_i + e_j - e_a - e_b)^{-1}.
        """
        explicit_denom = Add(*[t.use_explicit_denominators(return_sympy=True)
                               for t in self.terms])
        explicit_denom = Pow(explicit_denom, self.exponent)
        if return_sympy:
            return explicit_denom
        assumptions = self.assumptions
        if tensor_names.sym_orb_denom in self.antisym_tensors:
            assumptions["antisym_tensors"] = tuple(
                n for n in assumptions["antisym_tensors"]
                if n != tensor_names.sym_orb_denom
            )
        return Expr(explicit_denom, **assumptions)

    def description(self, *args, **kwargs):
        raise NotImplementedError("description not implemented for polynoms:",
                                  f"{self} in {self.term}")

    @property
    def allowed_spin_blocks(self) -> None:
        # allowed spin blocks not available for Polynoms
        return None

    @property
    def contains_only_orb_energies(self):
        """Whether the poylnom only contains orbital energy tensors."""
        return all(term.contains_only_orb_energies for term in self.terms)

    def to_latex_str(self, only_pull_out_pref: bool = False,
                     spin_as_overbar: bool = False):
        """Returns a latex string for the polynom."""
        tex_str = " ".join(
            [term.to_latex_str(only_pull_out_pref, spin_as_overbar)
             for term in self.terms]
        )
        tex_str = f"\\bigl({tex_str}\\bigr)"
        if self.exponent != 1:
            tex_str += f"^{{{self.exponent}}}"
        return tex_str
