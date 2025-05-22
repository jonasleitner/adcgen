from collections.abc import Iterable
from collections import Counter
from functools import cached_property
from typing import Any, TYPE_CHECKING, Sequence

from sympy import Add, Expr, Mul, Pow, S, factor, latex, nsimplify
from sympy.physics.secondquant import NO

from ..indices import (
    Index, Indices, get_lowest_avail_indices, get_symbols, order_substitutions,
    sort_idx_canonical
)
from ..misc import Inputerror, cached_member
from ..sympy_objects import NonSymmetricTensor
from .container import Container
from .normal_ordered_container import NormalOrderedContainer
from .polynom_container import PolynomContainer
from .object_container import ObjectContainer

# imports only required for type checking (avoid circular imports)
if TYPE_CHECKING:
    from ..symmetry import Permutation
    from .expr_container import ExprContainer


class TermContainer(Container):
    """
    Wrapper for a single term of the form a * b * c.

    Parameters
    ----------
    inner:
        The algebraic term to wrap, e.g., a sympy.Mul object
    target_idx: Iterable[Index] | None, optional
        Target indices of the expression. By default the Einstein sum
        convention will be used to identify target and contracted indices,
        which is not always sufficient.
    """

    def __init__(self, inner: Expr | Container | Any,
                 target_idx: Iterable[Index] | None = None) -> None:
        super().__init__(inner=inner, target_idx=target_idx)
        # we can not wrap an Add object: should be wrapped by ExprContainer
        # But everything else should be fine (Mul or single objects)
        assert not isinstance(self._inner, Add)

    def __len__(self) -> int:
        if isinstance(self.inner, Mul):
            return len(self.inner.args)
        else:
            return 1

    @cached_property
    def objects(self) -> tuple[ObjectContainer, ...]:
        """
        Returns all objects the term contains, e.g. tensors.
        """
        def dispatch(obj, kwargs) -> ObjectContainer:
            if isinstance(obj, NO):
                return NormalOrderedContainer(inner=obj, **kwargs)
            elif (isinstance(obj, Pow) and isinstance(obj.args[0], Add)) or \
                    isinstance(obj, Add):
                return PolynomContainer(inner=obj, **kwargs)
            else:
                return ObjectContainer(inner=obj, **kwargs)

        kwargs = self.assumptions
        if isinstance(self.inner, Mul):
            return tuple(
                dispatch(obj, kwargs)
                for obj in self.inner.args
            )
        else:
            return (dispatch(self.inner, kwargs),)

    ###############################################
    # methods that compute additional information #
    ###############################################
    @cached_property
    def order(self) -> int:
        return sum(
            obj.order for obj in self.objects
            if not isinstance(obj, PolynomContainer)
        )

    @cached_property
    def prefactor(self) -> Expr:
        """Returns the (numeric) prefactor of the term."""
        return nsimplify(
            Mul(*(o.inner for o in self.objects if o.inner.is_number)),
            rational=True
        )

    @property
    def sign(self) -> str:
        """Returns the sign of the term."""
        return "minus" if self.prefactor < S.Zero else "plus"

    @property
    def contracted(self) -> tuple[Index, ...]:
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
    def target(self) -> tuple[Index, ...]:
        """
        Returns all target indices of the term. If no target indices have been
        provided to the parent expression, the Einstein sum convention will
        be applied.
        """
        if (target := self.provided_target_idx) is not None:
            return target
        else:
            return tuple(s for s, n in self._idx_counter if not n)

    @cached_property
    def idx(self) -> tuple[Index, ...]:
        """
        Returns all indices that occur in the term. Indices that occur multiple
        times will be listed multiple times.
        """
        return tuple(s for s, n in self._idx_counter for _ in range(n + 1))

    @cached_property
    def _idx_counter(self) -> tuple[tuple[Index, int], ...]:
        idx: dict[Index, int] = {}
        for o in self.objects:
            if o.inner.is_number:
                continue
            n = abs(o.exponent)  # abs value for denominators
            assert n.is_Integer
            n = int(n)
            for s in o.idx:
                if s in idx:
                    idx[s] += n
                else:  # start counting at 0
                    idx[s] = n - 1
        return tuple(sorted(
            idx.items(), key=lambda itms: sort_idx_canonical(itms[0])
        ))

    @cached_member
    def pattern(self, include_target_idx: bool = True,
                include_exponent: bool = True
                ) -> dict[tuple[str, str], dict[Index, list[str]]]:
        """
        Determins the pattern of the indices in the term. This is a (kind of)
        readable string hash for each index that is based upon the positions
        the index appears and the coupling of the objects.

        Parameters
        ----------
        include_target_idx: bool, optional
            If set, the explicit names of target indices are included to make
            the pattern more precise. Should be set if the target indices
            are not allowed to be renamed. (default: True)
        include_exponent: bool, optional
            If set, the exponents of the objects are included in the pattern
            (default: True)
        """

        target_idx = self.target if include_target_idx else None
        coupl = self.coupling(
            include_target_idx=include_target_idx,
            include_exponent=include_exponent
        )
        pattern: dict[tuple[str, str], dict[Index, list[str]]] = {}
        for i, o in enumerate(self.objects):
            positions = o.crude_pos(target_idx=target_idx,
                                    include_exponent=include_exponent)
            c = f"_{'_'.join(sorted(coupl[i]))}" if i in coupl else None
            for s, pos in positions.items():
                key = s.space_and_spin
                if key not in pattern:
                    pattern[key] = {}
                if s not in pattern[key]:
                    pattern[key][s] = []
                if c is None:
                    pattern[key][s].extend(p for p in pos)
                else:
                    pattern[key][s].extend(p + c for p in pos)
        # sort pattern to allow for direct comparison
        for ov, idx_pat in pattern.items():
            for s, pat in idx_pat.items():
                pattern[ov][s] = sorted(pat)
        return pattern

    @cached_member
    def coupling(self, include_target_idx: bool = True,
                 include_exponent: bool = True) -> dict[int, list[str]]:
        """
        Returns the coupling between the objects in the term, where two objects
        are coupled when they share common indices. Only the coupling of non
        unique objects is returned, i.e., the coupling of e.g. a t2_1 amplitude
        is only returned if there is another one in the same term.
        """
        # - collect all the couplings (e.g. if a index s occurs at two tensors
        #   t and V: the crude_pos of s at t will be extended by the crude_pos
        #   of s at V. And vice versa for V.)
        objects = self.objects
        target_idx = self.target if include_target_idx else None
        descriptions = [
            o.description(include_exponent=include_exponent,
                          target_idx=target_idx)
            for o in objects
        ]
        descr_counter = Counter(descriptions)
        positions = [
            o.crude_pos(include_exponent=include_exponent,
                        target_idx=target_idx)
            for o in objects
        ]
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

    @cached_member
    def symmetry(self, only_contracted: bool = False,
                 only_target: bool = False
                 ) -> "dict[tuple[Permutation, ...], int]":
        """
        Determines the symmetry of the term with respect to index permutations.
        By default all indices of the term are considered. However, by setting
        either only_contracted or only_target the indices may be restricted to
        the respective subset of indices.
        """
        from itertools import combinations, permutations, chain, product
        from math import factorial
        from ..indices import split_idx_string
        from ..symmetry import Permutation, PermutationProduct

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
                    yield PermutationProduct(*chain.from_iterable(perm_tpl))

        if only_contracted and only_target:
            raise Inputerror("Can not set only_contracted and only_target "
                             "simultaneously.")
        if self.inner.is_number or isinstance(self.inner, NonSymmetricTensor):
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
        original_term = self.inner
        for perms in get_perms(*space_perms):
            permuted = self.permute(*perms).inner
            if Add(original_term, permuted) is S.Zero:
                symmetry[perms] = -1
            elif Add(original_term, -permuted) is S.Zero:
                symmetry[perms] = +1
        return symmetry

    @property
    def contains_only_orb_energies(self) -> bool:
        """Whether the term only contains orbital energies."""
        return all(
            o.contains_only_orb_energies for o in self.objects
            if not o.inner.is_number
        )

    ###############################
    # method that modify the term #
    ###############################
    def _apply_tensor_braket_sym(
            self, braket_sym_tensors: Sequence[str] = tuple(),
            braket_antisym_tensors: Sequence[str] = tuple(),
            wrap_result: bool = True) -> "ExprContainer | Expr":
        """
        Applies the tensor bra-ket symmetry defined in braket_sym_tensors and
        braket_antisym_tensors to all tensors in the term.
        If wrap_result is set,
        the new term will be wrapped by :py:class:`ExprContainer`.
        """
        from .expr_container import ExprContainer

        term = S.One
        for obj in self.objects:
            term *= obj._apply_tensor_braket_sym(
                braket_sym_tensors=braket_sym_tensors,
                braket_antisym_tensors=braket_antisym_tensors,
                wrap_result=False
            )
        assert isinstance(term, Expr)
        if wrap_result:
            term = ExprContainer(inner=term, **self.assumptions)
        return term

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

        res = S.One
        for obj in self.objects:
            res *= obj._rename_complex_tensors(wrap_result=False)
        if wrap_result:
            res = ExprContainer(inner=res, **self.assumptions)
        return res

    def block_diagonalize_fock(self, wrap_result: bool = True
                               ) -> "ExprContainer | Expr":
        """
        Block diagonalize the Fock matrix, i.e. if the term contains a off
        diagonal Fock matrix block (f_ov/f_vo) it is set to 0.

        Parameters
        ----------
        wrap_result: bool, optional
            If this is set the result will be wrapped with an
            :py:class:`ExprContainer`.
        """
        bl_diag = S.One
        for obj in self.objects:
            bl_diag *= obj.block_diagonalize_fock(wrap_result=False)

        if wrap_result:
            bl_diag = ExprContainer(bl_diag, **self.assumptions)
        return bl_diag

    def diagonalize_fock(self, target: Sequence[Index] | None = None,
                         wrap_result: bool = True,
                         apply_substitutions: bool = True
                         ) -> "ExprContainer | Expr | tuple[ExprContainer | Expr, dict[Index, Index]]":  # noqa E501
        """
        Represent the term in the canonical orbital basis, where the
        Fock matrix is diagonal. Because it is not possible to
        determine the target indices in the resulting term according
        to the Einstein sum convention, the current target indices will
        be set manually in the resulting term.

        Parameters
        ----------
        target: Sequence[Index] | None
            The target indices of a potential parent term.
        wrap_result: bool, optional
            If this is set the result will be wrapped with an
            :py:class:`ExprContainer`.
        apply_substitutions: bool, optional
            If set the index substitutions will be applied to the result.
            Otherwhise the substitutions will be returned in addition to the
            expression (without applying them).
            In both cases fock matrix elements will be replaced by orbital
            energie elements, e.g., f_ij will be replaced by e_i.
        """
        from .expr_container import ExprContainer

        if target is None:
            target = self.target

        sub: dict[Index, Index] = {}
        diag = S.One
        for o in self.objects:
            diag_obj, sub_obj = o.diagonalize_fock(target, wrap_result=False)
            diag *= diag_obj
            if any(k in sub and sub[k] != v for k, v in sub_obj.items()):
                raise NotImplementedError("Did not implement the case of "
                                          "multiple fock matrix elements with "
                                          f"intersecting indices: {self}")
            sub.update(sub_obj)

        if wrap_result:
            kwargs = self.assumptions
            kwargs["target_idx"] = target
            diag = ExprContainer(diag, **kwargs)
        if apply_substitutions:
            return diag.subs(order_substitutions(sub))
        else:
            return diag, sub

    def substitute_contracted(self, wrap_result: bool = True,
                              apply_substitutions: bool = True
                              ) -> "ExprContainer | Expr | list[tuple[Index, Index]]":  # noqa E501
        """
        Replace the contracted indices in the term with the lowest available
        (non-target) indices. This is done for each space and spin
        independently, i.e.,
        i_{\\alpha} j_{\\beta} -> i_{\\alpha} i_{\\beta}
        assuming both indices are contracted indices and
        i_{\\alpha} i_{\\beta} are not used as target indices.

        Parameters
        ----------
        wrap_result: bool, optional
            If set the result will be wrapped in an
            :py:class:`ExprContainer`. (default: True)
        apply_substitutions: bool, optional
            If set the substitutions will be applied to the
            term and the new expression is returned. Otherwise,
            the index substitutions will be returned without
            applying them to the expression. (default: True)
        """
        from .expr_container import ExprContainer

        # - determine the target and contracted indices
        #   and split them according to their space
        #   Don't use atoms to obtain the contracted indices! Atoms is a set
        #   and therefore not sorted -> will produce a random result.
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

        # - generate new indices the contracted will be replaced with
        #   and build a substitution dictionary
        #   Don't filter out indices that will not change!
        sub = {}
        for (space, spin), idx_list in contracted.items():
            new_idx = get_lowest_avail_indices(
                len(idx_list), used.get((space, spin), []), space
            )
            if spin:
                new_idx = get_symbols(new_idx, spin * len(idx_list))
            else:
                new_idx = get_symbols(new_idx)
            sub.update({o: n for o, n in zip(idx_list, new_idx)})
        # - apply substitutions while ensuring the substitutions are
        #   performed in the correct order
        sub = order_substitutions(sub)

        if not apply_substitutions:  # only build and return the sub_list
            return sub

        substituted = self.inner.subs(sub)
        assert isinstance(substituted, Expr)
        # ensure that the substitutions are valid
        if substituted is S.Zero and self.inner is not S.Zero:
            raise ValueError(f"Invalid substitutions {sub} for {self}.")

        if wrap_result:
            substituted = ExprContainer(substituted, **self.assumptions)
        return substituted

    def substitute_with_generic(self, wrap_result: bool = True
                                ) -> "ExprContainer | Expr":
        """
        Replace the contracted indices in the term with new, unused generic
        indices.
        """
        from .expr_container import ExprContainer
        # sort the contracted indices according to their space and spin
        contracted: dict[tuple[str, str], list[Index]] = {}
        for idx in self.contracted:
            if (key := idx.space_and_spin) not in contracted:
                contracted[key] = []
            contracted[key].append(idx)
        # generate new generic indices
        kwargs = {f"{space}_{spin}" if spin else space: len(indices)
                  for (space, spin), indices in contracted.items()}
        generic = Indices().get_generic_indices(**kwargs)
        # build the subs dict
        subs: dict[Index, Index] = {}
        for key, old_indices in contracted.items():
            new_indices = generic[key]
            subs.update({
                idx: new_idx for idx, new_idx in zip(old_indices, new_indices)
            })
        # substitute the indices
        substituted = self.inner.subs(order_substitutions(subs))
        assert isinstance(substituted, Expr)
        # ensure that the substitutions are valid
        if substituted is S.Zero and self.inner is not S.Zero:
            raise ValueError(f"Invalid substitutions {subs} for {self}.")

        if wrap_result:
            substituted = ExprContainer(substituted, **self.assumptions)
        return substituted

    def rename_tensor(self, current: str, new: str, wrap_result: bool = True
                      ) -> "ExprContainer | Expr":
        """
        Rename tensors in a terms.

        Parameters
        ----------
        wrap_result: bool, optional
            If this is set the result will be wrapped with an
            :py:class:`ExprContainer`. (default: True)
        """
        from .expr_container import ExprContainer

        renamed = S.One
        for obj in self.objects:
            renamed *= obj.rename_tensor(
                current=current, new=new, wrap_result=False
            )

        if wrap_result:
            renamed = ExprContainer(renamed, **self.assumptions)
        return renamed

    def expand_antisym_eri(self, wrap_result: bool = True
                           ) -> "ExprContainer | Expr":
        """
        Expands the antisymmetric ERI using chemists notation
        <pq||rs> = (pr|qs) - (ps|qr).
        ERI's in chemists notation are by default denoted as 'v'.
        Currently this only works for real orbitals, i.e., for
        symmetric ERI's <pq||rs> = <rs||pq>.
        """
        from .expr_container import ExprContainer

        expanded = S.One
        for obj in self.objects:
            expanded *= obj.expand_antisym_eri(wrap_result=False)

        if wrap_result:
            expanded = ExprContainer(expanded, **self.assumptions)
        return expanded

    def expand_intermediates(self, target: Sequence[Index] | None = None,
                             wrap_result: bool = True,
                             fully_expand: bool = True,
                             braket_sym_tensors: Sequence[str] = tuple(),
                             braket_antisym_tensors: Sequence[str] = tuple()
                             ) -> "ExprContainer | Expr":
        """
        Expands all known intermediates in the term.

        Parameters
        ----------
        target: tuple[Index] | None, optional
            The target indices of the term. Determined automatically if not
            given. Since it might not be possible to determine the
            target indices in the resulting expression (e.g. after
            expanding MP t-amplitudes) the target indices will be
            set in the expression.
        wrap_result: bool, optional
            If set the result is wrapped in an
            :py:class:`ExprContainer`. (default: True)
        fully_expand: bool, optional
            True (default): The intermediates are recursively expanded
              into orbital energies and ERI (if possible)
            False: The intermediates are only expanded once, e.g., n'th
              order MP t-amplitudes are expressed by means of (n-1)'th order
              MP t-amplitudes and ERI.
        braket_sym_tensors: Sequence[str], optional
            Add bra-ket-symmetry to the given tensors of the expanded
            expression (after expansion of the intermediates).
        braket_antisym_tensors: Sequence[str], optional
            Add bra-ket-antisymmetry to the given tensors of the expanded
            expression (after expansion of the intermediates).
        """
        from .expr_container import ExprContainer

        if target is None:
            target = self.target

        expanded = S.One
        for obj in self.objects:
            expanded *= obj.expand_intermediates(
                target, wrap_result=False, fully_expand=fully_expand,
                braket_sym_tensors=braket_sym_tensors,
                braket_antisym_tensors=braket_antisym_tensors
            )

        if wrap_result:
            assumptions = self.assumptions
            assumptions["target_idx"] = target
            expanded = ExprContainer(expanded, **assumptions)
        return expanded

    def factor(self) -> "ExprContainer":
        """
        Tries to factor the term.
        """
        from .expr_container import ExprContainer

        return ExprContainer(
            inner=factor(self.inner), **self.assumptions
        )

    def use_explicit_denominators(self, wrap_result: bool = True
                                  ) -> "ExprContainer | Expr":
        """
        Switch to an explicit representation of orbital energy denominators by
        replacing all symbolic denominators by their explicit counter part,
        i.e., D^{ij}_{ab} -> (e_i + e_j - e_a - e_b)^{-1}.
        """
        from .expr_container import ExprContainer

        explicit_denom = S.One
        for obj in self.objects:
            explicit_denom *= obj.use_explicit_denominators(wrap_result=False)

        if wrap_result:
            explicit_denom = ExprContainer(explicit_denom, **self.assumptions)
        return explicit_denom

    def split_orb_energy(self) -> "dict[str, ExprContainer]":
        """
        Splits the term in a orbital energy fraction and a remainder, e.g.
        (e_i + e_j) / (e_i + e_j - e_a - e_b) * (tensor1 * tensor2).
        To this end all polynoms that only contain orbital energy tensors
        ('e' by default) are collected to form the numerator and denominator,
        while the rest of the term is collected in the remainder.
        Prefactors are collected in the numerator.
        """
        from .expr_container import ExprContainer

        assumptions = self.assumptions
        assumptions["target_idx"] = self.target
        ret = {"num": ExprContainer(1, **assumptions),
               "denom": ExprContainer(1, **assumptions),
               "remainder": ExprContainer(1, **assumptions)}
        for o in self.objects:
            base, exponent = o.base_and_exponent
            if o.inner.is_number:
                key = "num"
            elif o.contains_only_orb_energies:
                key = "denom" if exponent < S.Zero else "num"
            else:
                key = "remainder"
            ret[key] *= Pow(base, abs(exponent))
        return ret

    def use_symbolic_denominators(self, wrap_result: bool = True
                                  ) -> "ExprContainer | Expr":
        """
        Replace all orbital energy denominators in the expression by tensors,
        e.g., (e_a + e_b - e_i - e_j)^{-1} will be replaced by D^{ab}_{ij},
        where D is a SymmetricTensor.
        """
        from ..eri_orbenergy import EriOrbenergy

        term = EriOrbenergy(self)
        symbolic_denom = term.symbolic_denominator()
        if wrap_result:
            return term.pref * symbolic_denom * term.num * term.eri
        return (
            term.pref * symbolic_denom.inner * term.num.inner * term.eri.inner
        )

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
            Instead of printing the spin of an index as suffix (name_spin)
            use an overbar for beta spin and no indication for alpha. Because
            alpha indices and indices without spin are not distinguishable
            anymore, this only works if all indices have a spin set (the
            expression is completely represented in spatial orbitals).
        """
        # - sign and prefactor
        pref = self.prefactor
        tex_str = "+ " if pref >= S.Zero else "- "
        # term only consists of a number (only pref)
        if self.inner.is_number:
            return tex_str + f"{latex(abs(pref))}"
        # avoid printing +- 1 prefactors
        if pref not in [+1, -1]:
            tex_str += f"{latex(abs(pref))} "

        # - latex strings for the remaining objects
        tex_str += " ".join([
            o.to_latex_str(only_pull_out_pref, spin_as_overbar)
            for o in self.objects if not o.inner.is_number
        ])
        return tex_str
