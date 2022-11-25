from indices import index_space
from misc import Inputerror
from sympy_objects import NonSymmetricTensor
from sympy import latex, Add, Mul, Pow, sympify, S
from sympy.physics.secondquant import (
    AntiSymmetricTensor, NO, F, Fd, KroneckerDelta
)


class expr:
    def __init__(self, e, real=False, sym_tensors=None, target_idx=None):
        if isinstance(e, (obj, term, expr, normal_ordered, polynom)):
            e = e.sympy
        self.__expr = sympify(e)
        self.__assumptions = {
            'real': real,
            'sym_tensors': set() if sym_tensors is None else set(sym_tensors),
            'target_idx': None
        }
        if target_idx is not None:
            self.set_target_idx(target_idx)
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
        return getattr(self.__expr, attr)

    @property
    def assumptions(self) -> dict:
        return self.__assumptions

    @property
    def real(self) -> bool:
        return self.__assumptions['real']

    @property
    def sym_tensors(self) -> set:
        return self.__assumptions['sym_tensors']

    @property
    def provided_target_idx(self) -> None | tuple:
        return self.__assumptions['target_idx']

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

    def set_sym_tensors(self, sym_tensors):
        if not all(isinstance(t, str) for t in sym_tensors):
            raise Inputerror("Symmetric tensors need to be provided as str.")
        self.__assumptions['sym_tensors'] = set(sym_tensors)
        if self.real:
            self.__assumptions['sym_tensors'].update(['f', 'V'])

    def set_target_idx(self, target_idx):
        from indices import get_symbols
        if target_idx is None:
            self.__assumptions['target_idx'] = None
        else:
            target_idx = get_symbols(target_idx)
            self.__assumptions['target_idx'] = tuple(
                sorted(set(target_idx), key=lambda s:
                       (int(s.name[1:]) if s.name[1:] else 0, s.name[0]))
            )

    def make_real(self):
        """Makes the expression real by removing all 'c' in tensor names.
           This only renames the tensor, but their might be more to simplify
           by swapping bra/ket.
           """
        # need to have the option return_sympy at lower levels, because
        # this function may be called upon instantiation

        self.__assumptions['real'] = True
        self.__assumptions['sym_tensors'].update(['f', 'V'])
        self.__expr = Add(*[t.make_real(return_sympy=True)
                            for t in self.terms])
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
        diag = compatible_int(0)
        for term in self.terms:
            diag += term.diagonalize_fock()
        self.__expr = diag.sympy
        self.__assumptions['target_idx'] = diag.provided_target_idx
        return self

    def rename_tensor(self, current, new):
        if not isinstance(current, str) or not isinstance(new, str):
            raise Inputerror("Old and new tensor name need to be provided as "
                             "strings.")
        renamed = compatible_int(0)
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

    def substitute_contracted(self):
        """Tries to substitute all contracted indices with pretty indices, i.e.
           i, j, k instad of i3, n4, o42 etc."""
        from indices import indices
        self.__expr = indices().substitute(self).sympy
        return self

    def factor(self, num=None):
        from sympy import factor, nsimplify
        if num is None:
            self.__expr = factor(self.sympy)
            return self
        num = sympify(num, rational=True)
        factored = map(lambda t: Mul(nsimplify(Pow(num, -1), rational=True),
                       t.sympy), self.terms)
        self.__expr = Mul(num, Add(*factored), evaluate=False)
        return self

    def permute(self, *perms):
        from indices import get_symbols

        for perm in perms:
            if len(perm) != 2:
                raise Inputerror(f"Permutation {perm} needs to be of length 2")
            p, q = get_symbols(perm)
            sub = {p: q, q: p}
            self.subs(sub, simultaneous=True)
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
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        self.__expr = self.sympy + other
        return self

    def __add__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy + other, **self.assumptions)

    def __isub__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        self.__expr = self.sympy - other
        return self

    def __sub__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy - other, **self.assumptions)

    def __imul__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        self.__expr = self.sympy * other
        return self

    def __mul__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy * other, **self.assumptions)

    def __itruediv__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        self.__expr = self.sympy / other
        return self

    def __truediv__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy / other, **self.assumptions)

    def __eq__(self, other):
        return isinstance(other, (obj, term, expr, normal_ordered, polynom)) \
                and self.assumptions == other.assumptions and \
                self.sympy == other.sympy


class term:
    def __new__(cls, e, pos=None, real=False, sym_tensors=[], target_idx=None):
        if isinstance(e, (expr, polynom)):
            if not isinstance(pos, int):
                raise Inputerror('Position needs to be provided as int.')
            return super().__new__(cls)
        else:
            return expr(e, real=real, sym_tensors=sym_tensors,
                        target_idx=target_idx)

    def __init__(self, e, pos=None, real=False, sym_tensors=[],
                 target_idx=None):
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
    def sym_tensors(self) -> set:
        return self.expr.sym_tensors

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
        """Returns all objects that contain at least 1 of the target indices of
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

    def make_real(self, return_sympy=False):
        """Makes the expression real by removing all 'c' in tensor names."""
        real_term = Mul(*[o.make_real(return_sympy=True)
                          for o in self.objects])
        if return_sympy:
            return real_term
        return expr(real_term, real=True, sym_tensors=self.sym_tensors,
                    target_idx=self.provided_target_idx)

    def block_diagonalize_fock(self):
        """Block diagonalize the Fock matrix, i.e. if the term contains a off
           diagonal Fock matrix block (f_ov/f_vo) it is set to 0."""
        bl_diag = Mul(*[o.block_diagonalize_fock().sympy
                        for o in self.objects])
        return expr(bl_diag, **self.assumptions)

    def diagonalize_fock(self, target=None):
        """Transform the term in the canonical basis without loosing any
           information. It might not be possible to determine the
           target inices in the result according to the einstein sum
           convention."""
        if target is None:
            target = self.target
        sub = {}
        diag = 1
        for o in self.objects:
            diag_obj, sub_obj = o.diagonalize_fock(target)
            diag *= diag_obj.sympy
            sub.update(sub_obj)
        # if term is part of a polynom -> return the sub dict and perform the
        # substitution in the polynoms parent term object.
        if isinstance(self.expr, polynom):
            return expr(diag, real=self.real, sym_tensors=self.sym_tensors,
                        target_idx=target), sub
        # provide the target indices to the returned expression, because
        # the current target indices might be lost due to the diagonalization
        return expr(diag.subs(sub, simultaneous=True), real=self.real,
                    sym_tensors=self.sym_tensors, target_idx=target)

    def rename_tensor(self, current, new):
        """Rename tensors in a terms. Returns a new expr instance."""
        renamed = compatible_int(1)
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

    def permute(self, *perms):
        """Applies the provided permutations to the term one after another,
           starting with the first one.
           Permutations need to be provided as e.g. tuples (a,b), (i,j), ...
           Indices may be provided as sympy Dummy symbols or strings."""
        from indices import get_symbols
        temp = self
        for perm in perms:
            if len(perm) != 2:
                raise Inputerror(f"Permutation {perm} needs to be of length 2")
            p, q = get_symbols(perm)
            sub = {p: q, q: p}
            temp = temp.subs(sub, simultaneous=True)
        return temp if isinstance(temp, expr) else \
            expr(temp, **self.assumptions)

    def symmetry(self, only_contracted=False, only_target=False):
        """Return a dict with all Symmetric and Antisymmetric Permutations of
           the term. If only contracted is set to True, only permutations of
           indices that are contracted in the term are considered. If
           only_target is set, only permutations of target indices are taken
           into account."""
        from itertools import combinations, permutations, chain
        from simplify import make_real
        from math import factorial
        from indices import split_idx_string

        def permute_str(string, *perms):
            string = split_idx_string(string)
            for perm in perms:
                p, q = [s.name for s in perm]
                idx1 = string.index(p)
                idx2 = string.index(q)
                string[idx1] = q
                string[idx2] = p
            return "".join(string)

        if only_contracted and only_target:
            raise Inputerror("Can not set only_contracted and only_target "
                             "simultaneously.")
        if only_contracted:
            indices = self.contracted
        elif only_target:
            indices = self.target
        else:
            indices = self.idx

        is_zero = {
            False: lambda x: x.sympy is S.Zero,
            True: lambda x: make_real(x, *self.sym_tensors).sympy is S.Zero
        }
        is_zero = is_zero[self.real]

        # sort indices according to their space. Use list for nice ordering
        idx = {'o': [], 'v': []}
        for s in indices:
            # in self.idx contracted indices are listed multiple times
            ov = index_space(s.name)[0]
            if s not in idx[ov]:
                idx[ov].append(s)
        # list of permutations and their symmetry, +-1
        sym = {'o': {}, 'v': {}}
        for ov, ov_idx in idx.items():
            max_n_perms = factorial(len(ov_idx))
            # represent the current indices as a string. All perms
            # will also be applied to the string, in order to
            # check whether the current perm is redundant
            idx_string = "".join([s.name for s in ov_idx])
            permuted_strings = [idx_string]
            # create all index pairs that may be permuted
            pairs = list(combinations(ov_idx, 2))
            # this creates more permutations than necessary
            # e.g. the cyclic permutation ijk -> kij can be defined
            # as P_jk P_ij // P_ij P_ik // P_ik P_jk
            combs = chain.from_iterable(
                [permutations(pairs, r) for r in range(1, len(ov_idx))]
            )
            for perms in combs:
                # we only need to find n! permutations (including
                # the identity)
                if len(permuted_strings) == max_n_perms:
                    break
                # apply the perm to the string and check whether
                # the resulting permuted string has already been
                # created by another permutation
                perm_string = permute_str(idx_string, *perms)
                if perm_string in permuted_strings:
                    continue
                permuted_strings.append(perm_string)
                sub = self.permute(*perms)
                if is_zero(self+sub):
                    sym[ov][perms] = -1
                elif is_zero(self-sub):
                    sym[ov][perms] = +1
        # multiply the occ and virt permutations
        symmetry = sym['o'] | sym['v']
        for o_perms, o_sym in sym['o'].items():
            for v_perms, v_sym in sym['v'].items():
                symmetry[o_perms + v_perms] = o_sym * v_sym
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
            res += factor * self.permute(*perm).sympy
        # renormalize the term
        res *= Rational(1, len(symmetry) + 1)
        return expr(res.expand(), **self.assumptions)

    @property
    def contracted(self):
        """Returns all contracted indices of the term. If no target indices
           have been provided to the parent expression, the Einstein sum
           convention will be applied."""

        if self.provided_target_idx is not None:
            return tuple(sorted(
                [s for s in self.__idx_counter.keys()
                 if s not in self.provided_target_idx], key=lambda s:
                (int(s.name[1:]) if s.name[1:] else 0, s.name)
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

        if self.provided_target_idx is not None:
            return self.provided_target_idx
        return tuple(sorted(
            [s for s, n in self.__idx_counter.items() if not n],
            key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    @property
    def idx(self):
        """Returns all indices that occur in the term. Indices that occur multiple
           times will be listed multiple times."""
        counter = self.__idx_counter
        idx = [s for s in counter.keys() for _ in range(counter[s] + 1)]
        return tuple(sorted(
            idx, key=lambda s: (int(s.name[1:]) if s.name[1:] else 0, s.name)
        ))

    @property
    def __idx_counter(self):
        idx = {}
        for o in self.objects:
            # abs value for denominators
            n = abs(o.exponent)
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
        """Returns the sign of the term. The sign might change according to
           sign_change when all objects are sorted in the canonical space.
           """
        return "minus" if self.prefactor < 0 else "plus"

    @property
    def sign_change(self):
        """Returns True if the sign of the term changes upon resorting all
           tensors in the term in their canonical form."""
        sign_change = False
        for o in self.objects:
            sign_change = not sign_change if o.sign_change else sign_change
        return sign_change

    @property
    def pattern(self):
        """Returns the pattern of the indices in the term."""
        target = self.target
        objects = self.objects
        positions = [o.crude_pos(target=target) for o in objects]
        coupl = self.coupling(target=target, positions=positions)
        ret = {'o': {}, 'v': {}}
        for i, pos in enumerate(positions):
            c = f"_{'_'.join(sorted(coupl[i]))}" if i in coupl else None
            for s, ps in pos.items():
                ov = index_space(s.name)[0]
                if s not in ret[ov]:
                    ret[ov][s] = []
                ret[ov][s].extend([p + c if c else p for p in ps])
        return ret

    def coupling(self, target=None, target_idx_string=True, positions=None,
                 include_exponent=True, descriptions=None):
        """Returns the coupling between the objects in the term, where two
           objects are coupled when they share common indices. Only the
           coupling of non unique objects is returned, i.e., the coupling
           of e.g. a t2_1 amplitude is only returned if there is another one in
           the same term."""
        from collections import Counter
        # 1) collect all the couplings (e.g. if a index s occurs at two tensors
        #    t and V: the crude_pos of s at t will be extended by the crude_pos
        #    of s at V. And vice versa for V.)
        if target is None:
            target = self.target
        objects = self.objects
        if descriptions is None:
            descriptions = [o.description(include_exponent) for o in objects]
        descr_counter = Counter(descriptions)
        if positions is None:
            positions = [
                o.crude_pos(target=target, target_idx_string=target_idx_string,
                            include_exponent=include_exponent,
                            description=descriptions[i])
                for i, o in enumerate(objects)
            ]
        coupling = {}
        for i, descr in enumerate(descriptions):
            # if the tensor is unique in the term -> no coupling necessary
            if descr_counter[descr] == 1:
                continue
            if i not in coupling:
                coupling[i] = []
            idx_pos = positions[i]
            for other_i, other_idx_pos in enumerate(positions):
                if i == other_i:
                    continue
                matches = [idx for idx in idx_pos if idx in other_idx_pos]
                coupling[i].extend(
                    [p for s in matches for p in other_idx_pos[s]]
                )
        # at some point it might be necessary to also add a counter to the
        # coupling if also the coupling is identical. However, so far I found
        # no example where this is necessary.
        return coupling

    def split_orb_energy(self):
        """Splits the term in a orbital energy fraction and a remainder, e.g.
           (e_i + e_j) / (e_i + e_j - e_a - e_b) * (tensor1 * tensor2). To
           this end all polynoms that only contain e tensors are collected to
           form the numerator and denominator, while the rest of the term
           is collected in the remainder."""

        ret = {"num": expr(1, self.real, self.sym_tensors, self.target),
               'denom': expr(1, self.real, self.sym_tensors, self.target),
               'remainder': expr(1, self.real, self.sym_tensors, self.target)}
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
            ret[key] *= expr(Pow(o.extract_pow, exponent), self.real,
                             self.sym_tensors, self.target)
        return ret

    @property
    def contains_only_orb_energies(self):
        return all(o.contains_only_orb_energies for o in self.objects
                   if not o.type == 'prefactor')

    def print_latex(self, only_pull_out_pref=False):
        """Returns a Latex string of the canonical form of the term."""
        # - sign and prefactor
        pref = self.prefactor * -1 if self.sign_change else self.prefactor
        tex_str = "+ " if pref >= 0 else "- "
        if pref not in [+1, -1]:
            tex_str += f"{latex(abs(pref))} "

        # - latex strings for the remaining objects
        tex_str += " ".join(
            [o.print_latex(only_pull_out_pref) for o in self.objects
             if o.type != 'prefactor']
        )
        return tex_str

    def __iadd__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy + other, **self.assumptions)

    def __add__(self, other):
        return self.__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy - other, **self.assumptions)

    def __sub__(self, other):
        return self.__isub__(other)

    def __imul__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy * other, **self.assumptions)

    def __mul__(self, other):
        return self.__imul__(other)

    def __itruediv__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy / other, **self.assumptions)

    def __truediv__(self, other):
        return self.__itruediv__(other)

    def __eq__(self, other):
        return isinstance(other, (obj, term, expr, normal_ordered, polynom)) \
                and self.assumptions == other.assumptions and \
                self.sympy == other.sympy


class obj:
    def __new__(cls, t, pos=None, real=False, sym_tensors=[], target_idx=None):
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
                        target_idx=target_idx)

    def __init__(self, t, pos=None, real=False, sym_tensors=[],
                 target_idx=None):
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
    def sym_tensors(self) -> set:
        return self.expr.sym_tensors

    @property
    def provided_target_idx(self) -> None | tuple:
        return self.expr.provided_target_idx

    @property
    def sympy(self):
        return self.obj

    def make_real(self, return_sympy=False):
        """Removes all 'c' in tensor names."""
        if self.type == 'antisym_tensor':
            old = self.name
            new = old.replace('c', '')
            real_obj = (Pow(AntiSymmetricTensor(new, self.extract_pow.upper,
                        self.extract_pow.lower), self.exponent)
                        if old != new else self.sympy)
        else:
            real_obj = self.sympy
        if return_sympy:
            return real_obj
        return expr(real_obj, True, self.sym_tensors, self.provided_target_idx)

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
            idx = sorted(
                    self.idx, key=lambda s:
                    (int(s.name[1:]) if s.name[1:] else 0, s.name[0])
            )
            if len(idx) != 2:
                raise RuntimeError(f"found fock matrix element {self} that "
                                   "does not hold exactly 2 indices.")
            # don't touch:
            #  - diagonal fock elements (for not loosing
            #    a contracted index by accident)
            #  - fock elements with both indices being target indices
            #    (for not loosing a target index in the term)
            if idx[0] == idx[1] or all(s in target for s in idx):
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
            return expr(diag, self.real, self.sym_tensors, target), sub
        return expr(self.sympy, **self.assumptions), sub

    def rename_tensor(self, current, new):
        """Renames a tensor object."""
        if 'tensor' in (type := self.type) and self.name == current:
            if type == 'antisym_tensor':
                base = AntiSymmetricTensor(
                        new, self.extract_pow.upper, self.extract_pow.lower
                    )
            elif type == 'nonsym_tensor':
                base = NonSymmetricTensor(new, self.extract_pow.indices)
            new_obj = Pow(base, self.exponent)
        else:
            new_obj = self.sympy
        return expr(new_obj, **self.assumptions)

    def expand(self):
        return expr(self.sympy.expand(), **self.assumptions)

    def subs(self, *args, **kwargs):
        return expr(self.obj.subs(*args, **kwargs), **self.assumptions)

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
        if (name := self.name) in ['X', 'Y']:
            return True  # ADC amplitudes
        elif name[0] == 't' and name[1:].isnumeric():
            return True  # t-amplitudes
        return False

    @property
    def pretty_name(self):
        name = None
        if 'tensor' in self.type:
            name = self.name
            if name[0] == 't' and name[1:].isnumeric():  # t-amplitudes
                if len(self.extract_pow.upper) != len(self.extract_pow.lower):
                    raise RuntimeError("Number of upper and lower indices not "
                                       f"equal for t-amplitude {self}.")
                name = f"t{len(self.extract_pow.upper)}_{name[1:]}"
        return name

    @property
    def symmetric(self):
        # nonsym_tensors are assumed to not have any symmetry atm!.
        return (self.type == 'antisym_tensor' and self.name in self.sym_tensors
                and len(self.extract_pow.upper) == len(self.extract_pow.lower))

    @property
    def idx(self):
        """Return the indices of the canonical ordered object."""
        get_idx = {
            'antisym_tensor': lambda _: self.canonical_idx_space[0],
            # delta indices are sorted automatically
            'delta': lambda o: (o.preferred_index, o.killable_index),
            'annihilate': lambda o: (o.args[0], ),
            'create': lambda o: (o.args[0], ),
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
    def space(self):
        """Returns the canonical space of tensors and other objects."""
        return "".join([index_space(s.name)[0] for s in self.idx])

    @property
    def canonical_idx_space(self):
        can_uplo = self.__canonical_uplo
        # amplitudes
        if self.is_amplitude:
            can_order = ('lower', 'upper')
        elif self.symmetric:
            uplo_space = {}
            uplo_idx = {}
            for uplo, idx in can_uplo.items():
                uplo_space[uplo] = (
                    "".join([index_space(s.name)[0] for s in idx])
                )
                # the 0'th index should be the lowest of the space
                uplo_idx[uplo] = idx[0].name
            can_order = sorted(
                uplo_space, key=lambda uplo:
                (uplo_space[uplo].count('v'),
                 int(uplo_idx[uplo][1:]) if uplo_idx[uplo][1:] else 0,
                 uplo_idx[uplo][0])
            )
            can_order = tuple(can_order)
        else:
            can_order = ('upper', 'lower')
        sp_len = {}
        for uplo in can_order:
            sp_len[uplo] = len(getattr(self.extract_pow, uplo))
        can_idx = tuple(s for uplo in can_order for s in can_uplo[uplo])
        return can_idx, can_order, sp_len

    @property
    def __canonical_uplo(self):
        """Sorts the indices of the upper and lower space in canonical order"""
        can_uplo = {}
        o = self.extract_pow
        for uplo in ['upper', 'lower']:
            can_uplo[uplo] = sorted(
                getattr(o, uplo), key=lambda s:
                (index_space(s.name)[0], int(s.name[1:]) if s.name[1:] else 0,
                 s.name[0])
            )
        return can_uplo

    def crude_pos(self, target=None, target_idx_string=True,
                  include_exponent=True, description=None):
        """Returns the 'crude' position of the indices in the object.
           (e.g. only if they are located in bra/ket, not the exact position)
           """
        if target_idx_string and target is None:
            target = self.term.target
        ret = {}
        if description is None:
            description = self.description(include_exponent)
        type = self.type
        if type == 'antisym_tensor':
            can_bk = self.__canonical_uplo
            # extract all target indices if the full position including
            # the names of target indices is desired.
            if target_idx_string:
                target_bk = {bk: [s for s in idx_tpl if s in target]
                             for bk, idx_tpl in can_bk.items()}
            for bk, idx_tpl in can_bk.items():
                other_bk = "upper" if bk == 'lower' else 'lower'
                for s in idx_tpl:
                    if s not in ret:
                        ret[s] = []
                    pos = f"{description}_ul" if self.symmetric else \
                        f"{description}_{bk[0]}"
                    # also attatch the space of the neighbours
                    neighbour_sp = [index_space(i.name)[0]
                                    for i in idx_tpl if i != s]
                    if neighbour_sp:
                        pos += f"_{''.join(neighbour_sp)}"
                    # also need to include the explicit name of target indices
                    if target_idx_string:
                        # - target indices that are in the same braket
                        same_target = [i.name for i in target_bk[bk] if i != s]
                        if same_target:
                            pos += f"_{''.join(same_target)}"
                        # - target indices that are in the other space/braket
                        other_target = [i.name for i in target_bk[other_bk]]
                        if other_target:
                            pos += f"_other-{''.join(other_target)}"
                    ret[s].append(pos)
        elif type == 'nonsym_tensor':
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
                        pos += f"_{'_'.join((name for name in other_target))}"
                ret[s].append(pos)
        elif type == 'delta':
            idx = self.idx
            if target_idx_string:
                target = [s for s in idx if s in target]
            for s in idx:
                if s not in ret:
                    ret[s] = []
                pos = description
                # also add the name of target indices on the same delta
                if target_idx_string:
                    other_target = [i.name for i in target if i != s]
                    if other_target:
                        pos += f"_{''.join(other_target)}"
                ret[s].append(pos)
        elif type in ['annihilate', 'create']:
            for s in self.idx:
                if s not in ret:
                    ret[s] = []
                ret[s].append(description)
        # for prefactor a empty dict is returned
        return ret

    @property
    def sign_change(self):
        """Returns True if the sign of the tensor object changes upon
           reordering the indices in canonical order."""
        if self.type != 'antisym_tensor':
            return False
        can_bk = self.__canonical_uplo
        tensor = Pow(AntiSymmetricTensor('42', can_bk['upper'],
                     can_bk['lower']), self.exponent)
        return isinstance(tensor, Mul)

    def description(self, include_exponent=True):
        """A string that describes the object."""
        descr = self.type
        if descr == 'antisym_tensor':
            # try to define the description by including the space
            # mainly relevant for ERI
            idx, can_order, sp_len = self.canonical_idx_space
            first_sp = [
                index_space(s.name)[0] for s in idx[:sp_len[can_order[0]]]
            ]
            second_sp = [
                index_space(s.name)[0] for s in idx[sp_len[can_order[0]]:]
            ]
            descr += f"_{self.name}_{''.join(first_sp)}_{''.join(second_sp)}"
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
        return 'tensor' in self.type and self.name == 'e'

    def print_latex(self, only_pull_out_pref=False):
        """Returns a Latex string of the canonical form of the object."""

        def tensor_string(idx):
            name = self.name
            # ERI; idx = [bra, ket]
            if name == 'V':
                tex_string = f"\\langle {idx[0]}\\vert\\vert {idx[1]}\\rangle"
            # t-amplitude; idx = [lower, upper]
            elif name[0] == 't' and name[1:].isnumeric():
                order = name[1:].replace('c', '')
                tex_string = (
                    "{" + name[0] + "^{" + idx[1] + "}_{" + idx[0] + "}}^{(" +
                    order + ")\\ast}" if 'c' in name[1:] else
                    "{" + name[0] + "^{" + idx[1] + "}_{" + idx[0] + "}}^{(" +
                    order + ")}"
                )
            # ADC-amplitude; idx = [lower, upper]
            elif name[0] in ['X', 'Y']:
                tex_string = name + "^{" + idx[1] + "}_{" + idx[0] + "}"
            # arbitrary other tensor; idx = [upper, lower]
            else:
                if name == 'e':  # orbital energies as epsilon
                    name = "\\varepsilon"
                tex_string = name + "^{" + idx[0] + "}_{" + idx[1] + "}"
            if self.exponent != 1:
                tex_string = (
                    f"{tex_string}^{{{self.exponent}}}" if name == 'V' else
                    f"\\bigl({tex_string}\\bigr)^{{{self.exponent}}}"
                )
            return tex_string

        if only_pull_out_pref or 'tensor' not in (type := self.type):
            return self.__str__()

        # Only For Tensors!
        if type == 'antisym_tensor':
            # idx are in canonical order!
            idx, sp_order, sp_len = self.canonical_idx_space
            # - split the indices in upper and lower spaces
            n = sp_len[sp_order[0]]
            idx = ("".join([s.name for s in idx[:n]]),
                   "".join([s.name for s in idx[n:]]))
        elif type == 'nonsym_tensor':  # always upper, lower
            idx = ("", "".join(s.name for s in self.idx))
        return tensor_string(idx)

    def __iadd__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy + other, **self.assumptions)

    def __add__(self, other):
        return self.__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy - other, **self.assumptions)

    def __sub__(self, other):
        return self.__isub__(other)

    def __imul__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy * other, **self.assumptions)

    def __mul__(self, other):
        return self.__imul__(other)

    def __itruediv__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            if self.assumptions != other.assumptions:
                raise TypeError("Assumptions need to be equal.")
            other = other.sympy
        return expr(self.sympy / other, **self.assumptions)

    def __truediv__(self, other):
        return self.__itruediv__(other)

    def __eq__(self, other):
        return isinstance(other, (obj, term, expr, normal_ordered, polynom)) \
                and self.assumptions == other.assumptions and \
                self.sympy == other.sympy


class normal_ordered(obj):
    """Container for a normal ordered operator string."""
    def __new__(cls, t, pos=None, real=False, sym_tensors=[], target_idx=None):
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
                        target_idx=target_idx)

    def __init__(self, t, pos=None, real=False, sym_tensors=[],
                 target_idx=None):
        super().__init__(t, pos)

    def __len__(self):
        # a NO obj can only contain a Mul object.
        return len(self.extract_no.args)

    @property
    def args(self):
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
                                      f"second quantized operators. {self}")
        return ret

    def crude_pos(self, target=None):
        if target is None:
            target = self.term.target
        descr = self.description
        ret = {}
        for o in self.objects:
            for s, pos in o.crude_pos(target).items():
                if s not in ret:
                    ret[s] = []
                ret[s].extend([f"{descr}_" + p for p in pos])
        return ret

    @property
    def description(self):
        # normal_ordered_#create_#annihilate
        objects = self.objects
        n_create = len([o for o in objects if o.type == 'create'])
        n_annihilate = len([o for o in objects if o.type == 'annihilate'])
        return f"{self.type}_{str(n_annihilate)}_{str(n_create)}"

    def print_latex(self, only_pull_out_pref=False):
        # no prefs possible in NO
        return " ".join([o.print_latex(only_pull_out_pref)
                        for o in self.objects])


class polynom(obj):
    """Container for a polynom (a+b+c)^x."""
    def __new__(cls, t, pos=None, real=False, sym_tensors=[], target_idx=None):
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
                        target_idx=target_idx)

    def __init__(self, t, pos=None, real=False, sym_tensors=[],
                 target_idx=None):
        super().__init__(t, pos)

    def __len__(self):
        # has to at least contain 2 terms: a+b
        return len(self.extract_pow.args)

    @property
    def args(self):
        return self.extract_pow.args

    @property
    def terms(self):
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

    def crude_pos(self, target=None):
        # I think this does not rly work correctly
        if target is None:
            target = self.term.target
        ret = {}
        descr = self.description
        for term in self.terms:
            sign = term.sign
            for o in term.objects:
                for s, pos in o.crude_pos(target).items():
                    if s not in ret:
                        ret[s] = []
                    ret[s].extend([f"{descr}_{sign[0]}_" + p for p in pos])
        return ret

    @property
    def description(self):
        content = []
        for term in self.terms:
            # differentiate prefactors
            descr = "".join(sorted(
                [o.description for o in term.objects if o.sympy not in [1, -1]]
            ))
            content.append(f"_{term.sign[0]}_{descr}")
        content = sorted(content)
        return "".join([f"{self.type}_{len(self)}", *content,
                       f"_{str(self.exponent)}"])

    def make_real(self, return_sympy=False):
        real = Add(*[t.make_real(return_sympy=True) for t in self.terms])
        real = Pow(real, self.exponent)
        if return_sympy:
            return real
        return expr(real, True, self.sym_tensors, self.provided_target_idx)

    def block_diagonalize_fock(self):
        bl_diag = Add(*[t.block_diagonalize_fock().sympy for t in self.terms])
        return expr(Pow(bl_diag, self.exponent), **self.assumptions)

    def diagonalize_fock(self, target=None):
        # block diagonalize first. There is not much to think about here and
        # the number of terms may be reduced
        bl_diag = self.block_diagonalize_fock()
        # block diagonalized polynom may be no longer a polynom
        # exponent 1 may also give just a tensor/delta/...
        if not isinstance(bl_diag.sympy, Pow) or \
                isinstance(bl_diag.sympy, Pow) and \
                not isinstance(bl_diag.sympy.args[0], Add):
            if target is not None:
                bl_diag.set_target_idx(target)
            return bl_diag.diagonalize_fock()
        # we are still having a (now block diagonal) polynom
        if len(bl_diag) != 1 or len(bl_diag.terms[0]) != 1:
            raise RuntimeError("Obtained an unexpected block diagonalized "
                               f"result {bl_diag}, starting with the polynom "
                               f"{self}.")
        bl_diag = bl_diag.terms[0].objects[0]
        if target is None:
            target = self.term.target
        # 1) old and new are not allowed to repeat in any other term within
        #    the same polynom
        # 2) old and new are not allowed to repeat in any other polynom that is
        #    also contained in the parent term, e.g. (a+b)/(c+d) * X
        terms = bl_diag.terms
        diag = 0
        sub = {}
        # idx of all terms in the polynom [(i,),]
        idx = [t.idx for t in terms]
        # idx of other polynoms are biased with off diagonal fock indices.
        if [pol for pol in self.term.polynoms if pol != self]:
            raise NotImplementedError("Fock diagonalization not implemented "
                                      "for multiple polynoms in a term: ",
                                      self)
        for i, term in enumerate(terms):
            # copy, delete self from idx and flatten
            other_idx = idx[:]
            del other_idx[i]
            other_idx = set([s for stpl in other_idx for s in stpl])
            # diagonalize the term and obtain the sub dictionary
            diag_term, sub_term = term.diagonalize_fock(target)
            # if any index that is involved in the substitution (old or new)
            # occurs in another term of the same polynom
            # -> dont diagonalize the fock matrix
            if any(s in other_idx for s in
                    [sym for kv in sub_term.items() for sym in kv]):
                diag_term = term
                sub_term = {}
            diag += diag_term.sympy
            sub.update(sub_term)
        return expr(Pow(diag, self.exponent), self.real, self.sym_tensors,
                    target), sub

    def rename_tensor(self, current, new):
        renamed = 0
        for term in self.terms:
            renamed += term.rename_tensor(current, new).sympy
        return expr(Pow(renamed, self.exponent), **self.assumptions)

    @property
    def symmetric(self):
        # what about the symmetry of a polynom (e.g. the symmetry of the
        # orbital energy denominator)
        raise NotImplementedError()

    @property
    def sign_change(self):
        # whether it is possible to pull a minus out, if ordering canonical?
        return all(t.sign_change for t in self.terms)

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


class compatible_int(int):
    def __init__(self, num):
        self.num = int(num)

    def __iadd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            return expr(self.num + other.sympy, **other.assumptions)
        elif isinstance(other, compatible_int):
            return compatible_int(self.num + other)
        else:
            return other + self.num

    def __isub__(self, other):
        return self.__sub__(other)

    def __sub__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            return expr(self.num - other.sympy, **other.assumptions)
        elif isinstance(other, compatible_int):
            return compatible_int(self.num - other)
        else:
            return self.num - other

    def __imul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            return expr(self.num * other.sympy, **other.assumptions)
        elif isinstance(other, compatible_int):
            return compatible_int(self.num * other)
        else:
            return self.num * other

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __truediv__(self, other):
        if isinstance(other, (obj, term, expr, normal_ordered, polynom)):
            return expr(self.num / other.sympy, **other.assumptions)
        else:
            return self.num / other
