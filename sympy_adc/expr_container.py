from .indices import index_space
from .misc import Inputerror
from .sympy_objects import NonSymmetricTensor, AntiSymmetricTensor
from sympy import latex, Add, Mul, Pow, sympify, S, Dummy, Basic
from sympy.physics.secondquant import NO, F, Fd, KroneckerDelta


class container:
    """Base class for all container classes."""

    def __add__(self, other):
        if isinstance(other, container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        return expr(self.sympy + other, **self.assumptions)

    def __iadd__(self, other):
        return self.__add__(other)

    def __radd__(self, other):
        # other: some sympy stuff or some number
        return expr(other + self.sympy, **self.assumptions)

    def __sub__(self, other):
        if isinstance(other, container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        return expr(self.sympy - other, **self.assumptions)

    def __isub__(self, other):
        return self.__sub__(other)

    def __rsub__(self, other):
        # other: some sympy stuff or some number
        return expr(other - self.sympy, **self.assumptions)

    def __mul__(self, other):
        if isinstance(other, container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        return expr(self.sympy * other, **self.assumptions)

    def __imul__(self, other):
        return self.__mul__(other)

    def __rmul__(self, other):
        # other: some sympy stuff or some number
        return expr(other * self.sympy, **self.assumptions)

    def __truediv__(self, other):
        if isinstance(other, container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        return expr(self.sympy / other, **self.assumptions)

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __rtruediv__(self, other):
        # other: some sympy stuff or some number
        return expr(other / self.sympy, **self.assumptions)

    def __eq__(self, other):
        return isinstance(other, container) \
                and self.assumptions == other.assumptions and \
                self.sympy == other.sympy


class expr(container):
    def __init__(self, e, real: bool = False, sym_tensors: list[str] = None,
                 antisym_tensors: list[str] = None,
                 target_idx: list[Dummy] = None):
        if isinstance(e, container):
            e = e.sympy
        self.__expr = sympify(e)
        self.__real: bool = False
        self.__sym_tensors: set = (set() if sym_tensors is None
                                   else set(sym_tensors))
        self.__antisym_tensors: set = (set() if antisym_tensors is None
                                       else set(antisym_tensors))
        self.__target_idx: None | tuple[Dummy] = None
        if target_idx is not None:
            self.set_target_idx(target_idx)
        if real:  # also calls _apply_tensor_braket_sym()
            self.make_real()
        # only apply bra_ket sym if 'f' and 'V' have been set manually
        # otherwise make_real applied the bra_ket sym already
        if (sym_tensors or antisym_tensors) and \
                ('f' in sym_tensors and 'V' in sym_tensors):
            self._apply_tensor_braket_sym()

    def __str__(self):
        return latex(self.sympy)

    def __len__(self):
        # 0 has length 1
        if self.type == 'expr':
            return len(self.args)
        else:
            return 1

    def __getattr__(self, attr):
        return getattr(self.__expr, attr)

    @property
    def assumptions(self) -> dict:
        return {'real': self.real,
                'sym_tensors': self.sym_tensors,
                'antisym_tensors': self.antisym_tensors,
                'target_idx': self.provided_target_idx}

    @property
    def real(self) -> bool:
        return self.__real

    @property
    def sym_tensors(self) -> tuple:
        return tuple(sorted(self.__sym_tensors))

    @property
    def antisym_tensors(self) -> tuple:
        return tuple(sorted(self.__antisym_tensors))

    @property
    def provided_target_idx(self) -> None | tuple[Dummy]:
        return self.__target_idx

    @property
    def sympy(self):
        return self.__expr

    @property
    def type(self) -> str:
        if isinstance(self.sympy, Add):
            return 'expr'
        elif isinstance(self.sympy, Mul):
            return 'term'
        else:
            return 'obj'

    @property
    def terms(self):
        return [term(self, i) for i in range(len(self))]

    def set_sym_tensors(self, sym_tensors: list[str]) -> None:
        if not all(isinstance(t, str) for t in sym_tensors):
            raise Inputerror("Symmetric tensors need to be provided as str.")
        sym_tensors = set(sym_tensors)
        if self.real:
            sym_tensors.update(['f', 'V'])
        if sym_tensors != self.__sym_tensors:
            self.__sym_tensors = sym_tensors
            self._apply_tensor_braket_sym()

    def set_antisym_tensors(self, antisym_tensors: list[str]) -> None:
        if not all(isinstance(t, str) for t in antisym_tensors):
            raise Inputerror("Tensors with antisymmetric bra ket symemtry need"
                             "to be provided as string.")
        antisym_tensors = set(antisym_tensors)
        self.__antisym_tensors = antisym_tensors
        self._apply_tensor_braket_sym()

    def set_target_idx(self, target_idx: None | list[str | Dummy]) -> None:
        from .indices import get_symbols
        if target_idx is None:
            self.__target_idx = None
        else:
            target_idx = set(get_symbols(target_idx))
            self.__target_idx = tuple(
                sorted(target_idx, key=lambda s:
                       (int(s.name[1:]) if s.name[1:] else 0, s.name[0]))
            )

    def make_real(self):
        """Makes the expression real by removing all 'c' in tensor names.
           This only renames the tensor, but their might be more to simplify
           by swapping bra/ket.
           """
        # need to have the option return_sympy at lower levels, because
        # this function may be called upon instantiation

        if self.__real:
            return self

        self.__real = True
        sym_tensors = self.__sym_tensors
        if 'f' not in sym_tensors or 'V' not in sym_tensors:
            self.__sym_tensors.update(['f', 'V'])
            self._apply_tensor_braket_sym()
        if self.sympy.is_number:
            return self
        self.__expr = Add(*[t.make_real(return_sympy=True)
                            for t in self.terms])
        return self

    def _apply_tensor_braket_sym(self):
        if self.sympy.is_number:
            return self
        expr_with_sym = Add(*[t._apply_tensor_braket_sym(return_sympy=True)
                              for t in self.terms])
        self.__expr = expr_with_sym
        return self

    def block_diagonalize_fock(self):
        """Block diagonalize the Fock matrix, i.e. all terms that contain off
           diagonal Fock matrix blocks (f_ov/f_vo) are set to 0."""
        self.expand()
        self.__expr = Add(*[t.block_diagonalize_fock().sympy
                            for t in self.terms])
        return self

    def diagonalize_fock(self):
        """Represent the expression in the canonical orbital basis, i.e. the
           Fock matrix is a diagonal matrix. Because it is not possible to
           determine the target indices in the resulting expression according
           to the Einstein sum convention, the current target indices will
           be set manually in the result."""
        # expand to get rid of polynoms as much as possible
        self.expand()
        diag = 0
        for term in self.terms:
            diag += term.diagonalize_fock()
        self.__expr = diag.sympy
        self.__target_idx = diag.provided_target_idx
        return self

    def rename_tensor(self, current, new):
        if not isinstance(current, str) or not isinstance(new, str):
            raise Inputerror("Old and new tensor name need to be provided as "
                             "strings.")
        renamed = 0
        for t in self.terms:
            renamed += t.rename_tensor(current, new)
        self.__expr = renamed.sympy
        return self

    def expand(self):
        self.__expr = self.sympy.expand()
        return self

    def subs(self, *args, **kwargs):
        self.__expr = self.sympy.subs(*args, **kwargs)
        return self

    def doit(self, *args, **kwargs):
        self.__expr = self.sympy.doit(*args, **kwargs)
        return self

    def substitute_contracted(self):
        """Tries to substitute all contracted indices with pretty indices, i.e.
           i, j, k instad of i3, n4, o42 etc."""
        from .indices import indices
        self.__expr = indices().substitute(self).sympy
        return self

    def factor(self, num=None):
        from sympy import factor, nsimplify
        if num is None:
            self.__expr = factor(self.sympy)
            return self
        num = nsimplify(num, rational=True)
        factored = map(lambda t: Mul(nsimplify(Pow(num, -1), rational=True),
                       t.sympy), self.terms)
        self.__expr = Mul(num, Add(*factored), evaluate=False)
        return self

    def permute(self, *perms):
        from .indices import get_symbols

        if not perms:
            return self
        if any(len(perm) != 2 for perm in perms):
            raise Inputerror("All permutations need to be of length 2. Got: "
                             f"{perms}")

        sub = {}
        for perm in perms:
            p, q = get_symbols(perm)
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
        sub = {old: new for old, new in sub.items() if old is not new}
        return self.subs(sub, simultaneous=True)

    def expand_intermediates(self):
        """Insert all defined intermediates in the expression."""
        # TODO: only expand specific intermediates
        expanded = 0
        for t in self.terms:
            expanded += t.expand_intermediates()
        self.__expr = expanded.sympy
        return self

    @property
    def idx(self):
        """Returns all indices that occur in the expression. Indices that occur
           multiple times will be listed multiple times."""
        idx = [s for t in self.terms for s in t.idx]
        return tuple(sorted(
            idx, key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    def copy(self):
        return expr(self.sympy, **self.assumptions)

    def print_latex(self, terms_per_line=None, only_pull_out_pref=False):
        """Returns a Latex string of the canonical form of the expr.
           The output may be adjusted to be compatible with the Latex align
           environment, where the parameter terms_per_line defines the number
           of terms that should be printed per line.
           """
        tex_terms = [term.print_latex(only_pull_out_pref)
                     for term in self.terms]
        # remove '+' in the first term
        if tex_terms[0][0] == '+':
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
        if isinstance(other, container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        elif isinstance(other, Basic):
            other = expr(other, **self.assumptions).sympy
        self.__expr = self.sympy + other
        return self

    def __isub__(self, other):
        if isinstance(other, container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        elif isinstance(other, Basic):
            other = expr(other, **self.assumptions).sympy
        self.__expr = self.sympy - other
        return self

    def __imul__(self, other):
        if isinstance(other, container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        elif isinstance(other, Basic):
            other = expr(other, **self.assumptions).sympy
        self.__expr = self.sympy * other
        return self

    def __itruediv__(self, other):
        if isinstance(other, container):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal. Got: "
                                f"{self.assumptions} and {other.assumptions}")
            other = other.sympy
        elif isinstance(other, Basic):
            other = expr(other, **self.assumptions).sympy
        self.__expr = self.sympy / other
        return self


class term(container):
    def __new__(cls, e, pos=None, real=False, sym_tensors=None,
                antisym_tensors=None, target_idx=None):
        if isinstance(e, (expr, polynom)):
            if not isinstance(pos, int):
                raise Inputerror('Position needs to be provided as int.')
            return super().__new__(cls)
        else:
            return expr(e, real=real, sym_tensors=sym_tensors,
                        antisym_tensors=antisym_tensors, target_idx=target_idx)

    def __init__(self, e, pos=None, real=False, sym_tensors=None,
                 antisym_tensors=None, target_idx=None):
        self.__expr: expr = e
        self.__pos: int = pos

    def __str__(self):
        return latex(self.term)

    def __len__(self):
        return len(self.args) if self.type == 'term' else 1

    def __getattr__(self, attr):
        return getattr(self.term, attr)

    @property
    def expr(self) -> expr:
        return self.__expr

    @property
    def term(self):
        if self.expr.type in ['expr', 'polynom']:
            return self.__expr.args[self.__pos]
        else:
            return self.__expr.sympy

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
        return self.__pos

    @property
    def sympy(self):
        return self.term

    @property
    def type(self) -> str:
        return 'term' if isinstance(self.term, Mul) else 'obj'

    @property
    def objects(self) -> list:
        return [obj(self, i) for i in range(len(self))]

    @property
    def tensors(self) -> list:
        """Returns all tensor objects in the term."""
        return [o for o in self.objects if 'tensor' in o.type]

    @property
    def deltas(self):
        """Returns all delta objects of the term."""
        return [o for o in self.objects if o.type == 'delta']

    @property
    def polynoms(self):
        return [o for o in self.objects if o.type == 'polynom']

    @property
    def target_idx_objects(self):
        """Returns all objects that hold at least 1 of the target indices of
           the term."""
        target = self.target
        ret = []
        for o in self.objects:
            idx = o.idx
            if any(s in idx for s in target):
                ret.append(o)
        return ret

    @property
    def order(self):
        """Returns the perturbation theoretical order of the term."""
        return sum(t.order for t in self.tensors)

    @property
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
        """Makes the expression real by removing all 'c' in the names
           of the t-amplitudes."""
        real_term = Mul(*[o.make_real(return_sympy=True)
                          for o in self.objects])
        if return_sympy:
            return real_term
        assumptions = self.assumptions
        assumptions['real'] = True
        return expr(real_term, **assumptions)

    def _apply_tensor_braket_sym(self, return_sympy=False):
        term_with_sym = Mul(*[o._apply_tensor_braket_sym(return_sympy=True)
                              for o in self.objects])
        if return_sympy:
            return term_with_sym
        return expr(term_with_sym, **self.assumptions)

    def block_diagonalize_fock(self):
        """Block diagonalize the Fock matrix, i.e. if the term contains a off
           diagonal Fock matrix block (f_ov/f_vo) it is set to 0."""
        bl_diag = Mul(*[o.block_diagonalize_fock().sympy
                        for o in self.objects])
        return expr(bl_diag, **self.assumptions)

    def diagonalize_fock(self, target=None):
        """Transform the term in the canonical basis without loosing any
           information. It might not be possible to determine the
           target indices in the result according to the einstein sum
           convention."""
        if target is None:
            target = self.target
        sub = {}
        diag = 1
        for o in self.objects:
            diag_obj, sub_obj = o.diagonalize_fock(target)
            diag *= diag_obj.sympy
            if any(k in sub and sub[k] != v for k, v in sub_obj.items()):
                raise NotImplementedError("Did not implement the case of "
                                          "multiple fock matrix elements with "
                                          f"intersecting indices: {self}")
            sub.update(sub_obj)
        # if term is part of a polynom -> return the sub dict and perform the
        # substitution in the polynoms parent term object.
        assumptions = self.assumptions
        assumptions['target_idx'] = target
        if isinstance(self.expr, polynom):
            return expr(diag, **assumptions), sub
        # provide the target indices to the returned expression, because
        # the current target indices might be lost due to the diagonalization
        return expr(diag.subs(sub, simultaneous=True), **assumptions)

    def rename_tensor(self, current, new):
        """Rename tensors in a terms. Returns a new expr instance."""
        renamed = 1
        for o in self.objects:
            renamed *= o.rename_tensor(current, new)
        return renamed

    def expand(self):
        return expr(self.term.expand(), **self.assumptions)

    def factor(self):
        from sympy import factor
        return expr(factor(self.sympy), **self.assumptions)

    def subs(self, *args, **kwargs):
        return expr(self.term.subs(*args, **kwargs), **self.assumptions)

    def doit(self, *args, **kwargs):
        return expr(self.sympy.doit(*args, **kwargs), **self.assumptions)

    def permute(self, *perms):
        """Applies the provided permutations to the term one after another,
           starting with the first one.
           Permutations need to be provided as e.g. tuples (a,b), (i,j), ...
           Indices may be provided as sympy Dummy symbols or strings."""
        from .indices import get_symbols

        if not perms:  # no perms to apply -> return self
            return expr(self.sympy, **self.assumptions)
        if any(len(perm) != 2 for perm in perms):
            raise Inputerror("All permutations need to be of length 2. Got: "
                             f"{perms}")

        sub = {}
        for perm in perms:
            p, q = get_symbols(perm)
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
        sub = {old: new for old, new in sub.items() if old is not new}
        return self.subs(sub, simultaneous=True)

    def expand_intermediates(self, target=None):
        target = self.target if target is None else target
        expanded = Mul(*[o.expand_intermediates(target).sympy
                         for o in self.objects])
        assumptions = self.assumptions
        assumptions['target_idx'] = target
        return expr(expanded, **assumptions)

    def symmetry(self, only_contracted: bool = False,
                 only_target: bool = False) -> dict[tuple, int]:
        """Determines the symmetry of the term with respect to index
           permutations, also taking bra_ket symmetry into account.
           By default all indices of the term are considered.
           However, by setting either only_contracted or only_target the
           indices may be restricted to the respective subspace."""
        from itertools import combinations, permutations, chain, product
        from math import factorial
        from collections import defaultdict
        from .indices import split_idx_string

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
                    yield tuple(perm for perms in perm_tpl for perm in perms)

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

        # split in occ and virt indices (only generate P_oo, P_vv and P_gg)
        sorted_idx = defaultdict(list)
        for s in indices:
            sorted_idx[index_space(s.name)[0]].append(s)

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
            pairs = list(combinations(idx_list, 2))
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
        symmetry: dict[tuple, int] = {}
        for perms in get_perms(*space_perms):
            permuted = self.permute(*perms)
            if self.sympy + permuted.sympy is S.Zero:
                symmetry[perms] = -1
            elif self.sympy - permuted.sympy is S.Zero:
                symmetry[perms] = +1
        return symmetry

    @property
    def symmetrize(self):
        """Symmetrise the term by applying all found symmetries to the term
           that only involve contracted indices and adding up the normalized
           result."""
        from sympy import Rational
        symmetry = self.symmetry(only_contracted=True)
        res = self.sympy
        for perm, factor in symmetry.items():
            res += self.permute(*perm).sympy * factor
        # renormalize the term
        res *= Rational(1, len(symmetry) + 1)
        return expr(res.expand(), **self.assumptions)

    @property
    def contracted(self):
        """Returns all contracted indices of the term. If no target indices
           have been provided to the parent expression, the Einstein sum
           convention will be applied."""

        if (target := self.provided_target_idx) is not None:
            return tuple(sorted(
                [s for s in self.__idx_counter.keys() if s not in target],
                key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
            ))
        return tuple(sorted(
            [s for s, n in self.__idx_counter.items() if n],
            key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    @property
    def target(self):
        """Returns all target indices of the term. If no target indices
           have been provided to the parent expression, the Einstein sum
           convention will be applied."""

        if (target := self.provided_target_idx) is not None:
            return target
        return tuple(sorted(
            [s for s, n in self.__idx_counter.items() if not n],
            key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    @property
    def idx(self):
        """Returns all indices that occur in the term. Indices that occur
           multiple times will be listed multiple times."""
        idx = [s for s, n in self.__idx_counter.items() for _ in range(n + 1)]
        return tuple(sorted(
            idx, key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    @property
    def __idx_counter(self):
        idx = {}
        for o in self.objects:
            n = abs(o.exponent)  # abs value for denominators
            for s in o.idx:
                if s in idx:
                    idx[s] += n
                else:  # start counting at 0
                    idx[s] = n - 1
        return idx

    @property
    def prefactor(self):
        """Returns the prefactor of the term."""
        from sympy import nsimplify
        pref = [o.sympy for o in self.objects if o.type == 'prefactor']
        if not pref:
            return sympify(1)
        return nsimplify(Mul(*pref), rational=True)

    @property
    def sign(self):
        """Returns the sign of the term."""
        return "minus" if self.prefactor < 0 else "plus"

    def pattern(self, target=None):
        """Returns the pattern of the indices in the term."""

        if target is None:
            target = self.target
        objects = self.objects
        positions = [o.crude_pos(target=target) for o in objects]
        coupl = self.coupling(target=target, positions=positions)
        pattern = {}
        for i, pos in enumerate(positions):
            c = f"_{'_'.join(sorted(coupl[i]))}" if i in coupl else None
            for s, ps in pos.items():
                ov = index_space(s.name)[0]
                if ov not in pattern:
                    pattern[ov] = {}
                if s not in pattern[ov]:
                    pattern[ov][s] = []
                pattern[ov][s].extend([p + c if c else p for p in ps])
        # sort pattern to allow for direct comparison
        for ov, idx_pat in pattern.items():
            for s, pat in idx_pat.items():
                pattern[ov][s] = sorted(pat)
        return pattern

    def coupling(self, target=None, positions=None,
                 target_idx_string=True, include_exponent=True):
        """Returns the coupling between the objects in the term, where two
           objects are coupled when they share common indices. Only the
           coupling of non unique objects is returned, i.e., the coupling
           of e.g. a t2_1 amplitude is only returned if there is another one in
           the same term."""
        from collections import Counter, defaultdict
        # 1) collect all the couplings (e.g. if a index s occurs at two tensors
        #    t and V: the crude_pos of s at t will be extended by the crude_pos
        #    of s at V. And vice versa for V.)
        if target is None:
            target = self.target
        objects = self.objects
        descriptions = [o.description(include_exponent) for o in objects]
        descr_counter = Counter(descriptions)
        if positions is None:
            positions = [o.crude_pos(target=target,
                                     target_idx_string=target_idx_string,
                                     include_exponent=include_exponent)
                         for i, o in enumerate(objects)]
        coupling = defaultdict(list)
        for i, descr in enumerate(descriptions):
            # if the tensor is unique in the term -> no coupling necessary
            if descr_counter[descr] < 2:
                continue
            idx_pos = positions[i]
            for other_i, other_idx_pos in enumerate(positions):
                if i == other_i:
                    continue
                matches = [idx for idx in idx_pos if idx in other_idx_pos]
                if not matches:
                    continue
                coupling[i].extend(
                    [p for s in matches for p in other_idx_pos[s]]
                )
        # at some point it might be necessary to also add a counter to the
        # coupling if also the coupling is identical. However, so far I found
        # no example where this is necessary.
        return dict(coupling)

    def split_orb_energy(self):
        """Splits the term in a orbital energy fraction and a remainder, e.g.
           (e_i + e_j) / (e_i + e_j - e_a - e_b) * (tensor1 * tensor2). To
           this end all polynoms that only contain e tensors are collected to
           form the numerator and denominator, while the rest of the term
           is collected in the remainder."""

        assumptions = self.assumptions
        assumptions['target_idx'] = self.target
        ret = {"num": expr(1, **assumptions),
               'denom': expr(1, **assumptions),
               'remainder': expr(1, **assumptions)}
        for o in self.objects:
            if o.contains_only_orb_energies:
                key = "denom" if o.exponent < 0 else "num"
            elif o.type == 'prefactor':
                key = "num"
            else:
                key = 'remainder'
            exponent = o.exponent
            if key == 'denom':
                exponent *= -1
            ret[key] *= expr(Pow(o.extract_pow, exponent), **assumptions)
        return ret

    @property
    def contains_only_orb_energies(self):
        return all(o.contains_only_orb_energies for o in self.objects
                   if not o.type == 'prefactor')

    def print_latex(self, only_pull_out_pref=False):
        """Returns a Latex string of the canonical form of the term."""
        # - sign and prefactor
        pref = self.prefactor
        tex_str = "+ " if pref >= 0 else "- "
        if pref not in [+1, -1]:
            tex_str += f"{latex(abs(pref))} "

        # - latex strings for the remaining objects
        tex_str += " ".join(
            [o.print_latex(only_pull_out_pref) for o in self.objects
             if o.type != 'prefactor']
        )
        return tex_str

    def optimized_contractions(self, target_indices: str = None,
                               max_tensor_dim: int = None):
        """Determine the contraction scheme of the term with the lowest
           computational scaling. The target indices can be provided as
           string, e.g., 'ijab' for a Doubles amplitude or 'iajb' for the
           p-h/p-h matrix block. If no target indices are provided the
           target indices will be sorted canonical."""
        from .generate_code import scaling, contraction_data, mem_scaling
        from .indices import get_symbols
        from collections import Counter
        from itertools import combinations

        def sort_canonical(idx):
            # duplicate of antisym tensor sort function. Hash was omitted
            return (index_space(idx.name)[0],
                    int(idx.name[1:]) if idx.name[1:] else 0,
                    idx.name[0])

        def extract_data(o):
            if isinstance(o, obj):
                return o.idx, o.pretty_name
            elif isinstance(o, contraction_data):
                return o.target, 'contraction'
            else:
                raise TypeError(f"Can not extract idx and name from {o}")

        def contraction(ob: obj | contraction_data, i: int | tuple[int],
                        other_ob: obj | contraction_data,
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

            # check that the result of the contraction does not violate
            # the max_tensor_dim
            if max_tensor_dim is not None and len(target) > max_tensor_dim:
                return None

            # determine the scaling of the contraction
            target_sp = Counter([index_space(s.name)[0] for s in target])
            contracted_sp = \
                Counter([index_space(s.name)[0] for s in contracted])

            occ = target_sp['o'] + contracted_sp['o']
            virt = target_sp['v'] + contracted_sp['v']
            general = target_sp['g'] + contracted_sp['g']
            total = occ + virt + general
            # mem = (total, general, virt, occ)
            mem = mem_scaling(len(target), target_sp['g'], target_sp['v'],
                              target_sp['o'])
            scal = scaling(total, general, virt, occ, mem)

            # sort contracted and target indices canonical and store as tuple
            contracted = tuple(sorted(contracted, key=sort_canonical))
            target = tuple(sorted(target, key=sort_canonical))

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
        n = 0
        for o in self.objects:
            # nonsym_tensor / antisym_tensor / delta
            if 'tensor' in (o_type := o.type) or o_type == 'delta':
                if (exp := o.exponent) < 0:
                    raise NotImplementedError("Contractions for divisions not "
                                              f"implemented: {self}.")
                for i in range(n, exp+n):
                    relevant_objects[i] = o
                    n += 1
            elif o_type == 'prefactor':  # prefactor
                continue
            else:  # polynom / create / annihilate / NormalOrdered
                raise NotImplementedError("Contractions not implemented for "
                                          "polynoms, creation and annihilation"
                                          f" operators: {self}.")

        # use the canonical target indices of the term
        if target_indices is None:
            target_indices = sorted(self.target, key=sort_canonical)
            canonical_target = tuple(target_indices)
        else:  # or transform the provided target indices to sympy symbols
            target_indices = get_symbols(target_indices)
            canonical_target = tuple(sorted(target_indices,
                                            key=sort_canonical))

        if len(relevant_objects) == 0:
            return []
        elif len(relevant_objects) == 1:  # only a single tensor
            i, o = next(iter(relevant_objects.items()))
            indices = o.idx
            target_sp = Counter(index_space(s.name)[0] for s in target_indices)
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
            return [contraction_data((i,), (indices,), (o.pretty_name,),
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
            sum_scaling = sum(contr.scaling.total for contr in variant)
            scalings = []
            for contr in variant:
                scal = list(contr.scaling)
                scal.insert(1, sum_scaling)
                scalings.append(scal)
            max_scalings.append(max(scalings))
        if not max_scalings:
            raise RuntimeError("Could not find a valid contraction scheme for "
                               f"{self} while restricting the maximum tensor "
                               f"dimension to {max_tensor_dim}.")
        variant, _ = min(
            zip(contraction_variants, max_scalings), key=lambda tpl: tpl[1]
        )
        return variant


class obj(container):
    def __new__(cls, t, pos=None, real=False, sym_tensors=None,
                antisym_tensors=None, target_idx=None):
        types = {
            NO: lambda o: 'no',
            Pow: lambda o: 'polynom' if isinstance(o.args[0], Add) else 'obj',
            # (a + b)^1 is a Add object that is part of a Mul object
            Add: lambda o: 'polynom'
        }
        if isinstance(t, (term, normal_ordered)):
            if not isinstance(pos, int):
                raise Inputerror('Position needs to be provided as int.')
            o = t.sympy if len(t) == 1 else t.args[pos]
            obj_type = types.get(type(o), lambda x: 'obj')(o)
            if obj_type == 'obj':
                return super().__new__(cls)
            elif obj_type == 'no':
                return normal_ordered(t, pos=pos)
            else:
                return polynom(t, pos=pos)
        else:
            return expr(t, real=real, sym_tensors=sym_tensors,
                        antisym_tensors=antisym_tensors, target_idx=target_idx)

    def __init__(self, t, pos=None, real=False, sym_tensors=None,
                 antisym_tensors=None, target_idx=None):
        self.__expr: expr = t.expr
        self.__term: term = t
        self.__pos: int = pos

    def __str__(self):
        return latex(self.sympy)

    def __getattr__(self, attr):
        return getattr(self.obj, attr)

    @property
    def expr(self) -> expr:
        return self.__expr

    @property
    def term(self) -> term:
        return self.__term

    @property
    def obj(self):
        return self.term.sympy if len(self.term) == 1 \
            else self.term.args[self.__pos]

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
        return self.obj

    def make_real(self, return_sympy: bool = False):
        """Removes all 'c' in the names of t-amplitudes."""

        if self.type == 'antisym_tensor':
            old = self.name
            new = old.replace('c', '')
            if old == new:
                real_obj = self.sympy
            else:
                tensor = self.extract_pow
                real_obj = Pow(
                    AntiSymmetricTensor(new, tensor.upper, tensor.lower,
                                        tensor.bra_ket_sym), self.exponent
                )
        else:
            real_obj = self.sympy
        if return_sympy:
            return real_obj
        assumptions = self.assumptions
        assumptions['real'] = True
        return expr(real_obj, **assumptions)

    def _apply_tensor_braket_sym(self, return_sympy: bool = False):
        if self.type == 'antisym_tensor':
            tensor = self.extract_pow
            bra_ket_sym = None
            if (name := self.name) in self.sym_tensors and \
                    tensor.bra_ket_sym is not S.One:
                bra_ket_sym = 1
            elif name in self.antisym_tensors and \
                    tensor.bra_ket_sym is not S.NegativeOne:
                bra_ket_sym = -1
            if bra_ket_sym is None:
                obj_with_sym = self.sympy
            else:
                obj_with_sym = Pow(tensor.add_bra_ket_sym(bra_ket_sym),
                                   self.exponent)
        else:
            obj_with_sym = self.sympy
        if return_sympy:
            return obj_with_sym
        return expr(obj_with_sym, **self.assumptions)

    def block_diagonalize_fock(self):
        if self.type == 'antisym_tensor' and self.name == 'f' and \
                self.space in ['ov', 'vo']:
            bl_diag = 0
        else:
            bl_diag = self.sympy
        return expr(bl_diag, **self.assumptions)

    def diagonalize_fock(self, target=None):
        sub = {}
        if self.type == 'antisym_tensor' and self.name == 'f':
            # block diagonalize -> target indices don't change
            if self.space in ['ov', 'vo']:
                return expr(0, **self.assumptions), sub
            if target is None:
                target = self.term.target
            # 0 is preferred, 1 is killable
            idx = self.idx
            if len(idx) != 2:
                raise RuntimeError(f"found fock matrix element {self} that "
                                   "does not hold exactly 2 indices.")
            # don't touch:
            #  - diagonal fock elements (for not loosing
            #    a contracted index by accident)
            #  - fock elements with both indices being target indices
            #    (for not loosing a target index in the term)
            if idx[0] is idx[1] or all(s in target for s in idx):
                return expr(self.sympy, **self.assumptions), sub
            # killable is contracted -> kill
            if idx[1] not in target:
                sub[idx[1]] = idx[0]
                preferred = idx[0]
            # preferred is contracted (and not killable) -> kill
            elif idx[0] not in target:
                sub[idx[0]] = idx[1]
                preferred = idx[1]
            diag = Pow(NonSymmetricTensor('e', (preferred,)), self.exponent)
            assumptions = self.assumptions
            assumptions['target_idx'] = target
            return expr(diag, **assumptions), sub
        return expr(self.sympy, **self.assumptions), sub

    def rename_tensor(self, current: str, new: str):
        """Renames a tensor object."""
        if 'tensor' in (type := self.type) and self.name == current:
            base = self.extract_pow
            if type == 'antisym_tensor':
                base = AntiSymmetricTensor(
                        new, base.upper, base.lower, base.bra_ket_sym
                    )
            elif type == 'nonsym_tensor':
                base = NonSymmetricTensor(new, base.indices)
            new_obj = Pow(base, self.exponent)
        else:
            new_obj = self.sympy
        return expr(new_obj, **self.assumptions)

    def expand(self):
        return expr(self.sympy.expand(), **self.assumptions)

    def subs(self, *args, **kwargs):
        return expr(self.obj.subs(*args, **kwargs), **self.assumptions)

    def doit(self, *args, **kwargs):
        return expr(self.sympy.doit(*args, **kwargs), **self.assumptions)

    @property
    def exponent(self):
        return self.obj.args[1] if isinstance(self.obj, Pow) else 1

    @property
    def extract_pow(self):
        return self.obj if self.exponent == 1 else self.obj.args[0]

    @property
    def type(self):
        types = {
            AntiSymmetricTensor: 'antisym_tensor',
            KroneckerDelta: 'delta',
            F: 'annihilate',
            Fd: 'create',
            NonSymmetricTensor: 'nonsym_tensor',
        }
        try:
            return types[type(self.extract_pow)]
        except KeyError:
            if self.is_number:
                return 'prefactor'
            raise RuntimeError(f"Unknown object: {self.obj} of type "
                               f"{type(self.obj)}.")

    @property
    def name(self):
        """Return the name of tensor objects."""
        if 'tensor' in self.type:
            return self.extract_pow.symbol.name

    @property
    def is_amplitude(self):
        name = self.name
        # ADC amplitude or t-amplitude
        return name in ['X', 'Y'] or \
            (name[0] == 't' and name[1:].replace('c', '').isnumeric())

    @property
    def order(self):
        """Returns the perturbation theoretical order of the obj."""
        from .intermediates import intermediates

        itmd = intermediates()
        if 'tensor' in self.type:
            if (name := self.name) == 'V':  # eri
                return 1
            # t-amplitudes
            elif name[0] == 't' and name[1:].replace('c', '').isnumeric():
                return int(name[1:].replace('c', ''))
            # all intermediates
            itmd_cls = itmd.available.get(self.pretty_name)
            if itmd_cls is not None:
                return itmd_cls.order
        return 0

    @property
    def pretty_name(self):
        name = None
        if 'tensor' in (o_type := self.type):
            name = self.name
            # t-amplitudes
            if name[0] == 't' and name[1:].replace('c', '').isnumeric():
                if len(self.extract_pow.upper) != len(self.extract_pow.lower):
                    raise RuntimeError("Number of upper and lower indices not "
                                       f"equal for t-amplitude {self}.")
                name = f"t{len(self.extract_pow.upper)}_{name[1:]}"
            elif name in ['X', 'Y']:  # adc amplitudes
                # need to determine the excitation space as int
                space = self.space
                n_o, n_v = space.count('o'), space.count('v')
                if n_o == n_v:  # pp-ADC
                    n = n_o  # p-h -> 1 // 2p-2h -> 2 etc.
                else:  # ip-/ea-/dip-/dea-ADC
                    n = min([n_o, n_v]) + 1  # h -> 1 / 2h -> 1 / p-2h -> 2...
                lr = "l" if name == "X" else 'r'
                name = f"u{lr}{n}"
            elif name[0] == 'p' and name[1:].isnumeric():  # mp densities
                if len(self.extract_pow.upper) != len(self.extract_pow.lower):
                    raise RuntimeError("Number of upper and lower indices not "
                                       f"equal for mp density {self}.")
                name = f"p0_{name[1:]}_{self.space}"
            elif name.startswith('t2eri'):  # t2eri
                name = f"t2eri_{name[5:]}"
            # t2sq -> just return name
        elif o_type == 'delta':  # deltas -> d_oo / d_vv
            name = f"d_{self.space}"
        return name

    @property
    def bra_ket_sym(self):
        # nonsym_tensors are assumed to not have any symmetry atm!.
        if self.type == 'antisym_tensor':
            return self.extract_pow.bra_ket_sym

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
        new_expr = expr(self.sympy, **assumptions)
        sym = new_expr.terms[0].symmetry(only_target=True)
        return sym

    @property
    def idx(self) -> tuple:
        """Return the indices of the canonical ordered object."""

        get_idx = {
            'antisym_tensor': lambda o: (
                o.lower + o.upper if self.is_amplitude else o.upper + o.lower
            ),
            'delta': lambda o: o.args,
            'annihilate': lambda o: o.args,
            'create': lambda o: o.args,
            'nonsym_tensor': lambda o: o.indices
        }
        try:
            return get_idx[self.type](self.extract_pow)
        except KeyError:
            if self.type == 'prefactor':
                return tuple()
            else:
                raise KeyError(f'Unknown obj type {self.type} for obj {self}')

    @property
    def space(self) -> str:
        """Returns the canonical space of tensors and other objects."""
        return "".join([index_space(s.name)[0] for s in self.idx])

    def crude_pos(self, target=None, target_idx_string: bool = True,
                  include_exponent: bool = True) -> dict:
        """Returns the 'crude' position of the indices in the object.
           (e.g. only if they are located in bra/ket, not the exact position)
           """
        if target_idx_string and target is None:
            target = self.term.target
        ret = {}
        description = self.description(include_exponent)
        o_type = self.type
        if o_type == 'antisym_tensor':
            uplo_idx = {'upper': self.extract_pow.upper,
                        'lower': self.extract_pow.lower}
            # extract all target indices if the full position, including
            # the names of target indices is desired.
            if target_idx_string:
                uplo_target = {bk: [s for s in idx_tpl if s in target]
                               for bk, idx_tpl in uplo_idx.items()}
            for uplo, idx_tpl in uplo_idx.items():
                other_uplo = 'upper' if uplo == 'lower' else 'lower'
                for s in idx_tpl:
                    if s not in ret:
                        ret[s] = []
                    pos = f"{description}_{uplo[0]}" \
                        if self.bra_ket_sym is S.Zero else description
                    # also attatch the space of the neighbours
                    # s can only occur once in the same space
                    neighbour_sp = [index_space(i.name)[0]
                                    for i in idx_tpl if i is not s]
                    if neighbour_sp:
                        pos += f"_{''.join(neighbour_sp)}"
                    # also need to include the explicit name of target indices
                    if target_idx_string:
                        # - target indices that are in the same braket
                        same_target = [i.name for i in uplo_target[uplo]
                                       if i is not s]
                        if same_target:
                            pos += f"_{''.join(same_target)}"
                        # - target indices that are in the other space/braket
                        other_target = [i.name for i in
                                        uplo_target[other_uplo]]
                        if other_target:
                            pos += f"_other-{''.join(other_target)}"
                    ret[s].append(pos)
        elif o_type == 'nonsym_tensor':
            # something like nonsym_tensor_name_space_expo_pos_targetname+pos
            idx = self.idx
            if target_idx_string:
                # should always be sorted! (insertion order of dicts)
                target_pos = {i: s.name for i, s in enumerate(idx)
                              if s in target}
            for i, s in enumerate(idx):
                if s not in ret:
                    ret[s] = []
                pos = f"{description}_{i}"
                if target_idx_string:
                    other_target = ["-".join([name, str(j)])
                                    for j, name in target_pos.items()
                                    if j != i]
                    if other_target:
                        pos += f"_{''.join((name for name in other_target))}"
                ret[s].append(pos)
        elif o_type == 'delta':
            idx = self.idx
            if target_idx_string:
                target = [s for s in idx if s in target]
            for s in idx:
                if s not in ret:
                    ret[s] = []
                pos = description
                # also add the name of target indices on the same delta
                if target_idx_string:
                    other_target = [i.name for i in target if i is not s]
                    if other_target:
                        pos += f"_{''.join(other_target)}"
                ret[s].append(pos)
        elif o_type in ['annihilate', 'create']:
            for s in self.idx:
                if s not in ret:
                    ret[s] = []
                ret[s].append(description)
        # for prefactor a empty dict is returned
        return ret

    def expand_intermediates(self, target=None):
        from .intermediates import intermediates
        # only tensors atm
        if 'tensor' not in self.type:
            return expr(self.sympy, **self.assumptions)
        if target is None:
            target = self.term.target
        itmd = intermediates()
        itmd = itmd.available.get(self.pretty_name, None)
        if itmd is None:
            expanded = self.sympy
        else:
            expanded = Pow(itmd.expand_itmd(indices=self.idx).sympy,
                           self.exponent)
        assumptions = self.assumptions
        assumptions['target_idx'] = target
        return expr(expanded, **assumptions)

    def description(self, include_exponent: bool = True) -> str:
        """A string that describes the object."""
        descr = self.type
        if descr == 'antisym_tensor':
            # try to define the description by including the space
            # mainly relevant for ERI
            tensor = self.extract_pow
            space_u = "".join(index_space(s.name)[0] for s in tensor.upper)
            space_l = "".join(index_space(s.name)[0] for s in tensor.lower)
            descr += f"_{self.name}_{space_u}_{space_l}"
            if include_exponent:
                descr += f"_{self.exponent}"
        elif descr == 'nonsym_tensor':
            descr += f"_{self.name}_{self.space}"
            if include_exponent:
                descr += f"_{self.exponent}"
        elif include_exponent and descr in ['delta', 'annihilate', 'create']:
            descr += f"_{self.exponent}"
        # prefactor
        return descr

    @property
    def contains_only_orb_energies(self):
        # all orb energies should be nonsym_tensors actually
        return 'tensor' in self.type and self.name == 'e'

    def print_latex(self, only_pull_out_pref: bool = False) -> str:
        """Returns a Latex string of the canonical form of the object."""

        def tensor_string(upper=None, lower=None):
            name = self.name
            if name == 'V':  # ERI
                tex_string = f"\\langle {upper}\\vert\\vert {lower}\\rangle"
            # t-amplitude
            elif name[0] == 't' and name[1:].replace('c', '').isnumeric():
                if 'c' in name[1:]:
                    order = name[1:].replace('c', '')
                    tex_string = "{t^{" + upper + "}_{" + lower + "}}^{(" + \
                        order + ")\\ast}"
                else:
                    order = name[1:]
                    tex_string = "{t^{" + upper + "}_{" + lower + "}}^{(" + \
                        order + ")}"
            else:  # arbitrary other tensor and ADC-amplitudes
                if name == 'e':  # orbital energies as epsilon
                    name = "\\varepsilon"
                tex_string = name
                if upper is not None:
                    tex_string += "^{" + upper + "}"
                if lower is not None:
                    tex_string += "_{" + lower + "}"
            if (exp := self.exponent) != 1:
                if name == 'V':  # ERI
                    tex_string += "{" + exp + "}"
                else:
                    tex_string = f"\\bigl({tex_string}\\bigr)^{{{exp}}}"
            return tex_string

        if only_pull_out_pref or 'tensor' not in (o_type := self.type):
            return self.__str__()

        # Only For Tensors!
        tensor = self.extract_pow
        if o_type == 'antisym_tensor':  # t/ADC-amplitudes etc.
            kwargs = {'upper': "".join(s.name for s in tensor.upper),
                      'lower': "".join(s.name for s in tensor.lower)}
        elif o_type == 'nonsym_tensor':  # orb energy + some special itmds
            kwargs = {'lower': "".join(s.name for s in tensor.indices)}
        return tensor_string(**kwargs)


class normal_ordered(obj):
    """Container for a normal ordered operator string."""
    def __new__(cls, t, pos=None, real=False, sym_tensors=None,
                antisym_tensors=None, target_idx=None):
        if isinstance(t, term):
            if not isinstance(pos, int):
                raise Inputerror('Position needs to be provided as int.')
            o = t.sympy if len(t) == 1 else t.args[pos]
            if isinstance(o, NO):
                return object.__new__(cls)
            else:
                raise RuntimeError('Trying to use normal_ordered container'
                                   f'for a non NO object: {o}.')
        else:
            return expr(t, real=real, sym_tensors=sym_tensors,
                        antisym_tensors=antisym_tensors, target_idx=target_idx)

    def __len__(self) -> int:
        # a NO obj can only contain a Mul object.
        return len(self.extract_no.args)

    @property
    def args(self) -> tuple:
        return self.extract_no.args

    @property
    def objects(self):
        return [obj(self, i) for i in range(len(self.extract_no.args))]

    @property
    def extract_no(self):
        return self.obj.args[0]

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
    def type(self):
        return 'NormalOrdered'

    @property
    def idx(self):
        objects = self.objects
        exp = self.exponent
        ret = tuple(s for o in objects for s in o.idx for _ in range(exp))
        if len(objects) != len(ret):
            raise NotImplementedError('Expected a NO object only to contain'
                                      "second quantized operators with an "
                                      f"exponent of 1. {self}")
        return ret

    def crude_pos(self, target=None, target_idx_string: bool = True,
                  include_exponent: bool = True):
        objects = self.objects
        if target_idx_string:
            if target is None:
                target = self.term.target
            target_in_no = [s for s in self.idx if s in target]

        descr = self.description(include_exponent)
        ret = {}
        for o in objects:
            o_descr = o.description(include_exponent)
            for s in o.idx:
                if s not in ret:
                    ret[s] = []
                pos = f"{descr}_{o_descr}"
                if target_idx_string:
                    other_target = "".join([i.name for i in target_in_no
                                            if i is not s])
                    if other_target:
                        pos += f"_{other_target}"
                ret[s].append(pos)
        return ret

    def description(self, include_exponent=True):
        descr = f"{self.type}_"
        exponent = self.exponent if include_exponent else 1
        for o in self.objects:
            if (type := o.type) == 'create':
                sp = f"{index_space(o.idx[0].name)[0]}+"
            elif type == 'annihilate':
                sp = index_space(o.idx[0].name)[0]
            else:
                raise TypeError("Unexpected content for NormalOrdered "
                                f"container: {o}, {type(o)}.")
            descr += "".join((sp for _ in range(exponent)))
        return descr

    def print_latex(self, only_pull_out_pref=False):
        # no prefs possible in NO
        return " ".join([o.print_latex(only_pull_out_pref)
                        for o in self.objects])


class polynom(obj):
    """Container for a polynom (a+b+c)^x."""
    def __new__(cls, t, pos=None, real=False, sym_tensors=None,
                antisym_tensors=None, target_idx=None):
        if isinstance(t, term):
            if not isinstance(pos, int):
                raise Inputerror("Position needs to be provided as int.")
            o = t.sympy if len(t) == 1 else t.args[pos]
            if isinstance(o.args[0], Add) or isinstance(o, Add):
                return object.__new__(cls)
            else:
                raise RuntimeError("Trying to use polynom container for a non"
                                   f"polynom object {o}.")
        else:
            return expr(t, real=real, sym_tensors=sym_tensors,
                        antisym_tensors=antisym_tensors, target_idx=target_idx)

    def __len__(self) -> int:
        # has to at least contain 2 terms: a+b
        return len(self.extract_pow.args)

    @property
    def args(self):
        return self.extract_pow.args

    @property
    def terms(self) -> list[term]:
        # overwriting args allows to pass self to the term instances
        return [term(self, i) for i in range(len(self))]

    @property
    def type(self):
        return 'polynom'

    @property
    def idx(self):
        """Returns all indices that occur in the polynom. Indices that occur
           multiple times will be listed multiple times."""
        idx = [s for t in self.terms for s in t.idx]
        return tuple(sorted(
            idx, key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    def make_real(self, return_sympy: bool = False) -> expr:
        real = Add(*[t.make_real(return_sympy=True) for t in self.terms])
        real = Pow(real, self.exponent)
        if return_sympy:
            return real
        assumptions = self.assumptions
        assumptions['real'] = True
        return expr(real, **assumptions)

    def _apply_tensor_braket_sym(self, return_sympy: bool = False) -> expr:
        with_sym = Add(*[t._apply_tensor_braket_sym(return_sympy=True)
                         for t in self.terms])
        with_sym = Pow(with_sym, self.exponent)
        if return_sympy:
            return with_sym
        return expr(with_sym, **self.assumptions)

    def block_diagonalize_fock(self):
        bl_diag = Add(*[t.block_diagonalize_fock().sympy for t in self.terms])
        return expr(Pow(bl_diag, self.exponent), **self.assumptions)

    def diagonalize_fock(self, target=None):
        raise NotImplementedError("Fock matrix diagonalization not implemented"
                                  " for polynoms")

    def rename_tensor(self, current: str, new: str) -> expr:
        renamed = 0
        for term in self.terms:
            renamed += term.rename_tensor(current, new).sympy
        return expr(Pow(renamed, self.exponent), **self.assumptions)

    @property
    def order(self):
        raise NotImplementedError("Order not implemented for polynoms.")

    def crude_pos(self, target=None, target_idx_string: bool = True,
                  include_exponent: bool = True) -> dict:
        raise NotImplementedError("crude_pos for determining index positions "
                                  "not implemented for polynoms.")

    def expand_intermediates(self, target=None) -> expr:
        if target is None:
            target = self.term.target
        expanded = Add(*[t.expand_intermediates(target).sympy
                         for t in self.terms])
        assumptions = self.assumptions
        assumptions['target_idx'] = target
        return expr(Pow(expanded, self.exponent), **assumptions)

    def description(self, include_exponent=True):
        raise NotImplementedError("description not implemented for polynoms.")

    @property
    def contains_only_orb_energies(self):
        return all(term.contains_only_orb_energies for term in self.terms)

    def print_latex(self, only_pull_out_pref=False):
        tex_str = " ".join(
            [term.print_latex(only_pull_out_pref) for term in self.terms]
        )
        tex_str = f"\\bigl({tex_str}\\bigr)"
        if self.exponent != 1:
            tex_str += f"^{{{self.exponent}}}"
        return tex_str
