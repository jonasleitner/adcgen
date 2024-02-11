from .indices import (get_lowest_avail_indices, get_symbols,
                      order_substitutions, idx_sort_key, Index)
from .misc import Inputerror, cached_property, cached_member
from .sympy_objects import (
    NonSymmetricTensor, AntiSymmetricTensor, KroneckerDelta
)
from sympy import latex, Add, Mul, Pow, sympify, S, Basic, nsimplify
from sympy.physics.secondquant import NO, F, Fd


class Container:
    """Base class for all container classes."""

    def expand(self):
        return Expr(self.sympy.expand(), **self.assumptions)

    def doit(self, *args, **kwargs):
        return Expr(self.sympy.doit(*args, **kwargs), **self.assumptions)

    def subs(self, *args, **kwargs):
        return Expr(self.sympy.subs(*args, **kwargs), **self.assumptions)

    def permute(self, *perms):
        """Applies the provided permutations in the specified order
           to the content of the container."""
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
                self._sym_tensors.update(['f', 'V'])
            self._apply_tensor_braket_sym()
        # then check if we are real (make real will adjust the tensor
        # symmetry too - if necessary)
        if real:
            self.make_real()

    def __str__(self):
        return latex(self.sympy)

    def __len__(self):
        # 0 has length 1
        if self.type == 'expr':
            return len(self.args)
        else:
            return 1

    def __getattr__(self, attr):
        return getattr(self._expr, attr)

    @property
    def assumptions(self) -> dict:
        return {'real': self.real,
                'sym_tensors': self.sym_tensors,
                'antisym_tensors': self.antisym_tensors,
                'target_idx': self.provided_target_idx}

    @property
    def real(self) -> bool:
        return self._real

    @property
    def sym_tensors(self) -> tuple:
        return tuple(sorted(self._sym_tensors))

    @property
    def antisym_tensors(self) -> tuple:
        return tuple(sorted(self._antisym_tensors))

    @property
    def provided_target_idx(self) -> None | tuple[Index]:
        return self._target_idx

    @property
    def sympy(self):
        return self._expr

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
        return tuple(Term(self, i) for i in range(len(self)))

    def set_sym_tensors(self, sym_tensors: list[str]) -> None:
        if not all(isinstance(t, str) for t in sym_tensors):
            raise Inputerror("Symmetric tensors need to be provided as str.")
        sym_tensors = set(sym_tensors)
        if self.real:
            sym_tensors.update(['f', 'V'])
        if sym_tensors != self._sym_tensors:
            self._sym_tensors = sym_tensors
            self._apply_tensor_braket_sym()

    def set_antisym_tensors(self, antisym_tensors: list[str]) -> None:
        if not all(isinstance(t, str) for t in antisym_tensors):
            raise Inputerror("Tensors with antisymmetric bra ket symemtry need"
                             "to be provided as string.")
        antisym_tensors = set(antisym_tensors)
        if antisym_tensors != self._antisym_tensors:
            self._antisym_tensors = antisym_tensors
            self._apply_tensor_braket_sym()

    def set_target_idx(self, target_idx: None | list[str | Index]) -> None:
        if target_idx is None:
            self._target_idx = None
        else:
            target_idx = set(get_symbols(target_idx))
            self._target_idx = tuple(sorted(target_idx, key=idx_sort_key))

    def make_real(self):
        """Makes the expression real by removing all 'c' in tensor names.
           This only renames the tensor, but their might be more to simplify
           by swapping bra/ket.
           """
        # need to have the option return_sympy at lower levels, because
        # this function may be called upon instantiation

        if self._real:
            return self

        self._real = True
        sym_tensors = self._sym_tensors
        if 'f' not in sym_tensors or 'V' not in sym_tensors:
            self._sym_tensors.update(['f', 'V'])
            self._apply_tensor_braket_sym()
        if self.sympy.is_number:
            return self
        self._expr = Add(*[t.make_real(return_sympy=True)
                           for t in self.terms])
        return self

    def _apply_tensor_braket_sym(self):
        if self.sympy.is_number:
            return self
        expr_with_sym = Add(*[t._apply_tensor_braket_sym(return_sympy=True)
                              for t in self.terms])
        self._expr = expr_with_sym
        return self

    def block_diagonalize_fock(self):
        """Block diagonalize the Fock matrix, i.e. all terms that contain off
           diagonal Fock matrix blocks (f_ov/f_vo) are set to 0."""
        self.expand()
        self._expr = Add(*[t.block_diagonalize_fock(return_sympy=True)
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
        self._expr = diag.sympy
        self._target_idx = diag.provided_target_idx
        return self

    def rename_tensor(self, current, new):
        if not isinstance(current, str) or not isinstance(new, str):
            raise Inputerror("Old and new tensor name need to be provided as "
                             "strings.")
        renamed = 0
        for t in self.terms:
            renamed += t.rename_tensor(current, new, return_sympy=True)
        self._expr = renamed
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
        """Insert all defined intermediates in the expression."""
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
        """Returns all indices that occur in the expression. Indices that occur
           multiple times will be listed multiple times."""
        idx = [s for t in self.terms for s in t.idx]
        return tuple(sorted(
            idx, key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    def copy(self):
        return Expr(self.sympy, **self.assumptions)

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
        return len(self.args) if self.type == 'term' else 1

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
        # can't cache directly, because getattr is overwritten
        if self._sympy is not None:
            return self._sympy

        if self.expr.type in ['expr', 'polynom']:
            sympy = self._expr.args[self._pos]
        else:
            sympy = self._expr.sympy
        self._sympy = sympy
        return self._sympy

    @property
    def type(self) -> str:
        return 'term' if isinstance(self.sympy, Mul) else 'obj'

    @cached_property
    def objects(self) -> tuple['Obj']:
        return tuple(Obj(self, i) for i in range(len(self)))

    @property
    def tensors(self) -> tuple['Obj']:
        """Returns all tensor objects in the term."""
        return tuple(o for o in self.objects if 'tensor' in o.type)

    @property
    def deltas(self) -> tuple['Obj']:
        """Returns all delta objects of the term."""
        return tuple(o for o in self.objects if o.type == 'delta')

    @property
    def polynoms(self) -> tuple['Polynom']:
        return tuple(o for o in self.objects if o.type == 'polynom')

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
        """Makes the expression real by removing all 'c' in the names
           of the t-amplitudes."""
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
        """Block diagonalize the Fock matrix, i.e. if the term contains a off
           diagonal Fock matrix block (f_ov/f_vo) it is set to 0."""
        bl_diag = Mul(*[o.block_diagonalize_fock(return_sympy=True)
                        for o in self.objects])
        if return_sympy:
            return bl_diag
        return Expr(bl_diag, **self.assumptions)

    def diagonalize_fock(self, target: tuple[Index] = None,
                         return_sympy: bool = False):
        """Transform the term in the canonical basis without loosing any
           information. It might not be possible to determine the
           target indices in the result according to the einstein sum
           convention."""
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
            if (key := (s.space, s.spin)) not in contracted:
                contracted[key] = []
            contracted[key].append(s)
        used = {}
        for s in set(self.target):
            if (key := (s.space, s.spin)) not in used:
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
        """Rename tensors in a terms. Returns a new expr instance."""
        renamed = Mul(*(o.rename_tensor(current, new, return_sympy=True)
                        for o in self.objects))
        if return_sympy:
            return renamed
        else:
            return Expr(renamed, **self.assumptions)

    def factor(self):
        from sympy import factor
        return Expr(factor(self.sympy), **self.assumptions)

    def expand_intermediates(self, target: tuple = None,
                             return_sympy: bool = False):
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
        """Determines the symmetry of the term with respect to index
           permutations, also taking bra_ket symmetry into account.
           By default all indices of the term are considered.
           However, by setting either only_contracted or only_target the
           indices may be restricted to the respective subspace."""
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

        # split in occ and virt indices (only generate P_oo, P_vv and P_gg)
        sorted_idx = {}
        for s in indices:
            if (sp := s.space[0]) not in sorted_idx:
                sorted_idx[sp] = []
            sorted_idx[sp].append(s)

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
        return Expr(res.expand(), **self.assumptions)

    @property
    def contracted(self) -> tuple[Index]:
        """Returns all contracted indices of the term. If no target indices
           have been provided to the parent expression, the Einstein sum
           convention will be applied."""

        # target indices have been provided -> no need to count indices
        if (target := self.provided_target_idx) is not None:
            return tuple(s for s, _ in self._idx_counter if s not in target)
        else:  # count indices to determine target and contracted indices
            return tuple(s for s, n in self._idx_counter if n)

    @property
    def target(self) -> tuple[Index]:
        """Returns all target indices of the term. If no target indices
           have been provided to the parent expression, the Einstein sum
           convention will be applied."""
        # dont cache target and contracted to allow them to react to
        # modifications of the assumptions

        if (target := self.provided_target_idx) is not None:
            return target
        else:
            return tuple(s for s, n in self._idx_counter if not n)

    @cached_property
    def idx(self) -> tuple[Index]:
        """Returns all indices that occur in the term. Indices that occur
           multiple times will be listed multiple times."""
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
        return tuple(sorted(idx.items(), key=lambda tpl: idx_sort_key(tpl[0])))

    @cached_property
    def prefactor(self):
        """Returns the prefactor of the term."""
        return nsimplify(
            Mul(*(o.sympy for o in self.objects if o.type == 'prefactor')),
            rational=True
        )

    @property
    def sign(self):
        """Returns the sign of the term."""
        return "minus" if self.prefactor < 0 else "plus"

    @cached_property
    def pattern(self) -> dict:
        """Determins the pattern of the indices in the term."""

        coupl = self.coupling()
        pattern = {}
        for i, o in enumerate(self.objects):
            positions = o.crude_pos()
            c = f"_{'_'.join(sorted(coupl[i]))}" if i in coupl else None
            for s, pos in positions.items():
                key = s.space[0] + s.spin
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
        """Returns the coupling between the objects in the term, where two
           objects are coupled when they share common indices. Only the
           coupling of non unique objects is returned, i.e., the coupling
           of e.g. a t2_1 amplitude is only returned if there is another one in
           the same term."""
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

    def split_orb_energy(self) -> dict:
        """Splits the term in a orbital energy fraction and a remainder, e.g.
           (e_i + e_j) / (e_i + e_j - e_a - e_b) * (tensor1 * tensor2). To
           this end all polynoms that only contain e tensors are collected to
           form the numerator and denominator, while the rest of the term
           is collected in the remainder."""

        assumptions = self.assumptions
        assumptions['target_idx'] = self.target
        ret = {"num": Expr(1, **assumptions),
               'denom': Expr(1, **assumptions),
               'remainder': Expr(1, **assumptions)}
        for o in self.objects:
            if o.contains_only_orb_energies:
                key = "denom" if o.exponent < 0 else "num"
            elif o.type == 'prefactor':
                key = "num"
            else:
                key = 'remainder'
            ret[key] *= Pow(o.extract_pow, abs(o.exponent))
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
        from collections import Counter
        from itertools import combinations

        def sort_canonical(idx):
            # duplicate of antisym tensor sort function. Hash was omitted
            return (idx.space[0],
                    int(idx.name[1:]) if idx.name[1:] else 0,
                    idx.name[0])

        def extract_data(o):
            if isinstance(o, Obj):
                return o.idx, o.pretty_name
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
        idx_occurences = {}
        n = 0
        for o in self.objects:
            # nonsym_tensor / antisym_tensor / delta
            if 'tensor' in (o_type := o.type) or o_type == 'delta':
                if (exp := o.exponent) < 0:
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
            elif o_type == 'prefactor':  # prefactor
                continue
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
            target_indices = sorted(self.target, key=sort_canonical)
            canonical_target = tuple(target_indices)
        else:  # or transform the provided target indices to sympy symbols
            target_indices = tuple(get_symbols(target_indices))
            canonical_target = tuple(sorted(target_indices,
                                            key=sort_canonical))

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
        else:
            return self.term.sympy if len(self.term) == 1 \
                else self.term.args[self._pos]

    def make_real(self, return_sympy: bool = False):
        """Removes all 'c' in the names of t-amplitudes."""

        if 'tensor' in self.type and self.is_amplitude:
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
        return Expr(real_obj, **assumptions)

    def _apply_tensor_braket_sym(self, return_sympy: bool = False):
        if self.type == 'antisymtensor':
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
        return Expr(obj_with_sym, **self.assumptions)

    def block_diagonalize_fock(self, return_sympy: bool = False):
        if self.name == 'f' and self.space in ['ov', 'vo']:
            bl_diag = 0
        else:
            bl_diag = self.sympy
        if return_sympy:
            return bl_diag
        return Expr(bl_diag, **self.assumptions)

    def diagonalize_fock(self, target: tuple[Index] = None,
                         return_sympy: bool = False):
        sub = {}
        if self.name == 'f':  # self contains a fock element
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
                    diag = Pow(NonSymmetricTensor('e', (preferred,)),
                               self.exponent)
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
        if 'tensor' in (o_type := self.type) and self.name == current:
            base = self.extract_pow
            if o_type == 'antisymtensor':
                base = AntiSymmetricTensor(
                        new, base.upper, base.lower, base.bra_ket_sym
                    )
            elif o_type == 'nonsymtensor':
                base = NonSymmetricTensor(new, base.indices)
            new_obj = Pow(base, self.exponent)
        else:
            new_obj = self.sympy
        if return_sympy:
            return new_obj
        else:
            return Expr(new_obj, **self.assumptions)

    @property
    def exponent(self):
        return self.sympy.args[1] if isinstance(self.sympy, Pow) else 1

    @property
    def extract_pow(self):
        return self.sympy.args[0] if isinstance(self.sympy, Pow) \
            else self.sympy

    @cached_property
    def type(self) -> str:
        types = {
            AntiSymmetricTensor: 'antisymtensor',
            KroneckerDelta: 'delta',
            F: 'annihilate',
            Fd: 'create',
            NonSymmetricTensor: 'nonsymtensor',
        }
        try:
            return types[type(self.extract_pow)]
        except KeyError:
            if self.is_number:
                return 'prefactor'
            raise NotImplementedError(f"Unknown object: {self.sympy} of type "
                                      f"{type(self.sympy)}.")

    @property
    def name(self) -> str:
        """Extract the name of tensor objects."""
        if 'tensor' in self.type:
            return self.extract_pow.symbol.name

    @property
    def is_amplitude(self):
        name = self.name
        # ADC amplitude or t-amplitude
        return name in ['X', 'Y'] or \
            (name[0] == 't' and name[1:].replace('c', '').isnumeric())

    @cached_property
    def order(self):
        """Returns the perturbation theoretical order of the obj."""
        from .intermediates import Intermediates

        if 'tensor' in self.type:
            if (name := self.name) == 'V':  # eri
                return 1
            # t-amplitudes
            elif name[0] == 't' and name[1:].replace('c', '').isnumeric():
                return int(name[1:].replace('c', ''))
            # all intermediates
            itmd_cls = Intermediates().available.get(self.pretty_name, None)
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
            elif name == 't2sq':
                pass
            else:  # arbitrary other tensor
                name += f"_{self.space}"
        elif o_type == 'delta':  # deltas -> d_oo / d_vv
            name = f"d_{self.space}"
        return name

    @property
    def bra_ket_sym(self):
        # nonsym_tensors are assumed to not have any symmetry atm!.
        if self.type == 'antisymtensor':
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
        new_expr = Expr(self.sympy, **assumptions)
        return new_expr.terms[0].symmetry(only_target=True)

    @cached_property
    def idx(self) -> tuple:
        """Return the indices of the canonical ordered object."""

        get_idx = {
            'antisymtensor': lambda o: (
                o.lower + o.upper if self.is_amplitude else o.upper + o.lower
            ),
            'delta': lambda o: o.args,
            'annihilate': lambda o: o.args,
            'create': lambda o: o.args,
            'nonsymtensor': lambda o: o.indices
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

        if include_target_idx:
            target = self.term.target

        ret = {}
        description = self.description(include_exponent=include_exponent,
                                       include_target_idx=include_target_idx)
        o_type = self.type
        if o_type == 'antisymtensor':
            tensor = self.extract_pow
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
        elif o_type == 'nonsymtensor':
            # target idx position is already in the description
            idx = self.idx
            for i, s in enumerate(idx):
                if s not in ret:
                    ret[s] = []
                ret[s].append(f"{description}_{i}")
        elif o_type in ['delta', 'annihilate', 'create']:
            for s in self.idx:
                if s not in ret:
                    ret[s] = []
                ret[s].append(description)
        # for prefactor a empty dict is returned
        return ret

    def expand_intermediates(self, target: tuple = None,
                             return_sympy: bool = False):
        from .intermediates import Intermediates

        # intermediates only defined for tensors
        if 'tensor' not in self.type:
            if return_sympy:
                return self.sympy
            else:
                return Expr(self.sympy, **self.assumptions)

        if target is None:
            target = self.term.target

        itmd = Intermediates()
        itmd = itmd.available.get(self.pretty_name, None)
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

        descr = self.type
        if descr == 'prefactor':
            return descr

        if include_target_idx:
            target = self.term.target

        if descr == 'antisymtensor':
            # - space separated in upper and lower part
            tensor = self.extract_pow
            upper, lower = tensor.upper, tensor.lower
            data_u = "".join(s.space[0] + s.spin for s in upper)
            data_l = "".join(s.space[0] + s.spin for s in lower)
            descr += f"-{self.name}-{data_u}-{data_l}"
            # names of target indices, also separated in upper and lower part
            # indices in upper and lower have been sorted upon tensor creation!
            if include_target_idx:
                target_u = "".join(s.name for s in upper if s in target)
                target_l = "".join(s.name for s in lower if s in target)
                if target_l or target_u:  # we have at least 1 target idx
                    if tensor.bra_ket_sym is S.Zero:  # no bra ket symmetry
                        if not target_u:
                            target_u = 'none'
                        if not target_l:
                            target_l = 'none'
                        descr += f"-{target_u}-{target_l}"
                    else:  # bra ket sym or antisym
                        # target indices in both spaces
                        if target_u and target_l:
                            descr += f"-{'-'.join(sorted([target_u, target_l]))}"  # noqa E501
                        else:  # only in 1 space at least 1 target idx
                            descr += f"-{target_u + target_l}"
            if include_exponent:  # add exponent to description
                descr += f"-{self.exponent}"
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
            descr += f"-{self.space}"
            if include_target_idx and \
                    (target_str := "".join(s.name for s in self.idx
                                           if s in target)):
                descr += f"-{target_str}"
            if include_exponent:
                descr += f"-{self.exponent}"
        return descr

    @property
    def allowed_spin_blocks(self) -> tuple[str]:
        """Returns the valid spin blocks of tensors."""
        from .intermediates import Intermediates
        from itertools import product

        if "tensor" not in self.type:
            raise NotImplementedError("Only implemented for tensors.")

        name = self.name
        if name == "V":  # hardcode the ERI spin blocks
            return ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
        # t-amplitudes: all spin conserving spin blocks are allowed, i.e.,
        # all blocks with the same amount of alpha and beta indices
        # in upper and lower
        elif name[0] == 't' and name[1:].replace('c', '').isnumeric():
            if len(self.idx) % 2:
                raise ValueError("Expected t-amplitude to have the same "
                                 f"of upper and lower indices: {self}.")
            n = len(self.idx)//2
            return tuple(
                sorted(["".join(block)
                        for block in product("ab", repeat=len(self.idx))
                        if block[:n].count("a") == block[n:].count("a")])
            )
        # the known spin blocks of eri and t-amplitudes may be used to
        # generate the spin blocks of other intermediates
        itmd = Intermediates().available.get(self.pretty_name, None)
        if itmd is None:
            raise NotImplementedError("Can not determine spin blocks for "
                                      f"{self}. Not available as intermediate")
        return itmd.allowed_spin_blocks

    @property
    def contains_only_orb_energies(self):
        # all orb energies should be nonsym_tensors actually
        return self.name == 'e' and len(self.idx) == 1

    def print_latex(self, only_pull_out_pref: bool = False) -> str:
        """Returns a Latex string of the canonical form of the object."""

        def tensor_string(upper=None, lower=None):
            name = self.name
            if name == 'V':  # ERI
                tex_string = f"\\langle {upper}\\vert\\vert {lower}\\rangle"
            elif name == 'f':  # fock matrix: only lower indices
                tex_string = "f_{" + upper + lower + "}"
            else:  # arbitrary other tensor and amplitudes
                order_str = ""
                # t-amplitudes
                if name[0] == 't' and name[1:].replace('c', '').isnumeric():
                    if 'c' in name[1:]:
                        order = name[1:].replace('c', '')
                    else:
                        order = name[1:]
                    name = "{t"
                    order_str = "}^{(" + order + ")}"
                elif name == 'e':  # orbital energies as epsilon
                    name = "\\varepsilon"
                elif name[0] == 'p' and name[1:].isnumeric():  # mp densities
                    order = name[1:]
                    name = "{\\rho"
                    order_str = "}^{(" + order + ")}"

                tex_string = name
                if upper is not None:
                    tex_string += "^{" + upper + "}"
                if lower is not None:
                    tex_string += "_{" + lower + "}"
                tex_string += order_str

            if (exp := self.exponent) != 1:
                if name == 'V':  # ERI
                    tex_string += "^{" + str(exp) + "}"
                else:
                    tex_string = f"\\bigl({tex_string}\\bigr)^{{{exp}}}"
            return tex_string

        if only_pull_out_pref or 'tensor' not in (o_type := self.type):
            return self.__str__()

        # Only For Tensors!
        tensor = self.extract_pow
        if o_type == 'antisymtensor':  # t/ADC-amplitudes etc.
            kwargs = {'upper': "".join(s.name for s in tensor.upper),
                      'lower': "".join(s.name for s in tensor.lower)}
        elif o_type == 'nonsymtensor':  # orb energy + some special itmds
            kwargs = {'lower': "".join(s.name for s in tensor.indices)}
        return tensor_string(**kwargs)


class NormalOrdered(Obj):
    """Container for a normal ordered operator string."""
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
    def type(self):
        return 'NormalOrdered'

    @cached_property
    def idx(self):
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
            if (type := o.type) == 'create':
                op_str += '+'
            elif type == 'annihilate':
                pass
            else:
                raise TypeError("Unexpected content for NormalOrdered "
                                f"container: {o}, {type(o)}.")
            obj_contribs.append(op_str)
        return f"{self.type}-{'-'.join(sorted(obj_contribs))}"

    def print_latex(self, only_pull_out_pref=False):
        # no prefs possible in NO
        return " ".join([o.print_latex(only_pull_out_pref)
                        for o in self.objects])


class Polynom(Obj):
    """Container for a polynom (a+b+c)^x."""
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
        return len(self.extract_pow.args)

    @property
    def args(self):
        return self.extract_pow.args

    @cached_property
    def terms(self) -> list[Term]:
        # overwriting args allows to pass self to the term instances
        return tuple(Term(self, i) for i in range(len(self)))

    @property
    def type(self):
        return 'polynom'

    @cached_property
    def idx(self):
        """Returns all indices that occur in the polynom. Indices that occur
           multiple times will be listed multiple times."""
        idx = [s for t in self.terms for s in t.idx]
        return tuple(sorted(
            idx, key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    def make_real(self, return_sympy: bool = False) -> Expr:
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
        renamed = Add(*(term.rename_tensor(current, new, return_sympy=True)
                        for term in self.terms))
        renamed = Pow(renamed, self.exponent)
        if return_sympy:
            return renamed
        else:
            return Expr(renamed, **self.assumptions)

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
        if target is None:
            target = self.term.target

        expanded = Add(*[t.expand_intermediates(target, return_sympy=True)
                         for t in self.terms])
        if return_sympy:
            return expanded
        else:
            assumptions = self.assumptions
            assumptions['target_idx'] = target
            return Expr(Pow(expanded, self.exponent), **assumptions)

    def description(self, *args, **kwargs):
        raise NotImplementedError("description not implemented for polynoms:",
                                  f"{self} in {self.term}")

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
