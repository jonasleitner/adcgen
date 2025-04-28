from collections.abc import Sequence
from collections import defaultdict
from itertools import chain, combinations

from sympy import S

from .indices import Index, sort_idx_canonical
from . import expr_container as e
from .misc import cached_member, cached_property, Inputerror
from .eri_orbenergy import EriOrbenergy


class Permutation(tuple):
    """
    Represents a permutation operator P_{pq} that permutes the indices p and q.
    """

    def __new__(cls, p: Index, q: Index):
        if sort_idx_canonical(p) < sort_idx_canonical(q):
            args = (p, q)
        else:
            args = (q, p)
        return super().__new__(cls, args)

    def __str__(self):
        return f"P_{self[0].name}{self[1].name}"

    def __repr__(self):
        return f"P_{self[0].name}{self[1].name}"


class PermutationProduct(tuple):
    """
    Represents a product of permutation operators P_{pq}P{rs}.
    The permutations are sorted taking into account that it is only possible
    to rearrange permutation operators if the indices belong to different
    spaces, e.g., P_{ab}P_{ij} = P_{ij}P_{ab}.
    """

    def __new__(cls, args: Sequence[Permutation]):
        # identify spaces that are linked to each other
        # the order of permutations within a linked group has to be maintained!
        # e.g. P_ia * P_ij * P_ab * P_pq
        # the spaces o and v are linked -> the order of the first 3
        # permutations has to be maintained, while P_pq can be moved
        # to any arbitrary place
        args = tuple(args)
        splitted = cls.split_in_separable_parts(args)
        args = [val for _, val in sorted(splitted.items())]
        return super().__new__(cls, chain.from_iterable(args))

    @staticmethod
    def split_in_separable_parts(permutations):
        """
        Splits the permutations in subsets that can be treated independently
        of each other.
        """

        # split the permutations according to their index space
        # and identify spaces that are linked to each other through at least
        # 1 permutation
        perm_spaces = []
        links = []
        for perm in permutations:
            p, q = perm
            space = set((p.space[0] + p.spin, q.space[0] + q.spin))
            perm_spaces.append(space)

            if len(space) > 1:  # identify linking permutations
                if space not in links:
                    links.append(space)

        if len(links) == 0:  # no links, all spaces separated
            linked_spaces = []
        elif len(links) == 1:  # exactly 2 spaces are linked
            linked_spaces = links
        else:  # more than 2 spaces linked: either ov, ox or ov, xy
            treated = set()
            linked_spaces = []
            for i, linked_sp in enumerate(links):
                if i in treated:
                    continue
                linked = linked_sp.copy()
                for other_i in range(i+1, len(links)):
                    if other_i in treated:
                        continue
                    if linked_sp & links[other_i]:
                        linked.update(links[other_i])
                        treated.add(other_i)
                linked_spaces.append(linked)

        # sort them in groups that can be treated independently
        ret = {}
        for perm, space in zip(permutations, perm_spaces):
            # if the current space is linked to other spaces
            # -> replace the space by the linked space
            for linked_sp in linked_spaces:
                if any(sp in linked_sp for sp in space):
                    space = linked_sp
                    break
            space = "".join(sorted(space))
            if space not in ret:
                ret[space] = []
            ret[space].append(perm)
        return ret


class LazyTermMap:
    """
    Establishes a term map for an expression that contains information about
    terms that can be mapped onto each other when permuting target indices of
    the expression.
    """

    def __init__(self, expr: e.Expr):
        self._expr = expr
        self._terms: tuple[e.Term] = expr.terms  # init all term objects
        self._term_map = {}  # {(perms, factor): {i: other_i}}

    def evaluate(self, antisymmetric_result_tensor: bool = True):
        """
        Fully evaluates the term map of the expression by probing all
        possible permutations of target indices.
        Due to an ambiguous definition of the symmetry by means of products of
        permutation operators
        (ijk -> kij can be obtained by applying P_{ij}P_{ik} or P_{ik}P_{jk})
        it might still be possible to encounter unevaluated entries at a
        later point.

        Parameters
        ----------
        antisymmetric_result_tensor: bool, optional
            The result tensor is either represented by an AntiSymmetricTensor
            (True) or by a SymmetricTensor (False). (default: True)
        """
        from .sympy_objects import AntiSymmetricTensor, SymmetricTensor

        # if we put all indices in lower bra-ket sym is not important
        if antisymmetric_result_tensor:
            tensor = AntiSymmetricTensor("x", tuple(), self.target_indices)
        else:
            tensor = SymmetricTensor("x", tuple(), self.target_indices)
        tensor = e.Expr(tensor).terms[0]
        for sym in tensor.symmetry().items():
            self[sym]
        return self._term_map

    def __getitem__(self, symmetry: tuple):
        """
        Checks whether a given symmetry as already been evaluated and probes
        the expression for the symmetry if this is not the case.

        Parameters
        ----------
        symmetry : tuple
            A tuple containing the permutations and the corresponding factor:
            +1 to probe for symmetry (+ P_{pq}P_{rs}...) and
            -1 to probe for antisymmetry (- P_{pq}P_{rs}...).
        """
        # did we already compute the map for the desired symmetry?
        if symmetry in self._term_map:
            return self._term_map[symmetry]
        # split the permutations according to their index space.
        # invert the permutations in possible space combinations
        # and check if we computed any of the partially or fully inverted
        # symmetries
        permutations, factor = symmetry
        splitted = list(
            PermutationProduct.split_in_separable_parts(permutations).items()
        )
        # also check the sorted version before inverting
        if not isinstance(permutations, PermutationProduct):
            permutations = tuple(chain.from_iterable(
                [val for _, val in sorted(splitted)]
            ))
            sym = (permutations, factor)
            if sym in self._term_map:
                return self._term_map[sym]

        invertable_subsets = [i for i, (_, perms) in enumerate(splitted)
                              if len(perms) > 1]
        for n_inverts in range(1, len(invertable_subsets)+1):
            for to_invert in combinations(invertable_subsets, n_inverts):
                inv_perms = []
                for i, val in enumerate(splitted):
                    if i in to_invert:  # invert the order of the permutations
                        inv_perms.append((val[0], val[1][::-1]))
                    else:
                        inv_perms.append(val)
                inv_perms = tuple(chain.from_iterable(
                    [val for _, val in sorted(inv_perms)]
                ))
                # check if the inverted variant has been already computed
                sym = (inv_perms, factor)
                if sym in self._term_map:
                    return self._term_map[sym]
        # could not find any variant in the term_map
        # -> probe the expression for the original variant
        return self.probe_symmetry(permutations, factor)

    @cached_property
    def target_indices(self):
        """Returns the target indices of the expression."""

        if self._expr.provided_target_idx is not None:
            return self._expr.provided_target_idx

        # determine the target indices of each term and ensure all terms hold
        # the same target indices
        target = self._terms[0].target
        if any(term.target != target for term in self._terms):
            raise NotImplementedError("Can only create a term map for an "
                                      "expression where each term is holding "
                                      "the same target indices.")
        return target

    @cached_member
    def _prescan_terms(self) -> tuple:
        """
        Prescan the terms of the expression collecting compatible terms that
        might be mapped onto each other.

        Returns
        -------
        tuple[bool, list]
            First entry: Indicates whether the corresopnding terms have an
                         orbital energy denominator.
            Second entry: The terms by their index.
        """

        filtered_terms = defaultdict(list)
        for term_i, term in enumerate(self._terms):
            # split the term in pref, orbital energy frac and remainder
            term = EriOrbenergy(term)
            # get the description of all objects in the remainder (eri) part
            # don't include target indices in the description since thats
            # what we want to probe the expr for (contracted permutations
            # can be simplified, which is assumed to have happened before.)
            eri_descriptions: tuple[str] = tuple(sorted(
                o.description(include_target_idx=False)
                for o in term.eri.objects
            ))
            # space of contracted indices
            idx_space = "".join(sorted(
                s.space[0] + s.spin for s in term.eri.contracted
            ))
            # the number and length of brackets in the denominator
            key = (eri_descriptions, term.denom_description(), idx_space)
            filtered_terms[key].append(term_i)
        # rearrange the term idx lists so the information whether they
        # contain a denominator is directly available
        # Also remove lists with a single entry... cant map them onto
        # anything else anyway
        return tuple(
            (False, term_i_list) if key[1] is None else (True, term_i_list)
            for key, term_i_list in filtered_terms.items()
            if len(term_i_list) > 1
        )

    def probe_symmetry(self, permutations: PermutationProduct,
                       sym_factor: int) -> dict:
        """
        Probes which terms in the expression can be mapped onto each other
        by applying the given symmetry.

        Parameters
        ----------
        permutations : PermutationProduct
            A prdocut of permutations of target indices of the expression.
        sym_factor : int
            Possible values:
            +1 -> probe for symmetry (Term + P_{pq}P_{rs}... Term)
            -1 -> probe for antisymmetry (Term - P_{pq}P_{rs}... Term)

        Returns
        -------
        dict
            Contains the index of terms which, when the provided permutations
            are applied, become equal to other non-permuted terms.
            key: The index of the permuted term.
            value: The index of the term it can be mapped onto.
        """
        from itertools import chain
        from .reduce_expr import factor_eri_parts, factor_denom
        from .simplify import simplify

        def simplify_with_denom(expr: e.Expr) -> e.Expr:
            if expr.sympy.is_number:  # trivial
                return expr

            factored = chain.from_iterable(
                factor_denom(sub_e) for sub_e in factor_eri_parts(expr)
            )
            ret = e.Expr(0, **expr.assumptions)
            for term in factored:
                ret += term.factor()
            return ret

        if sym_factor not in [1, -1]:
            raise Inputerror(f"Invalid symmetry factor {sym_factor}. +-1 "
                             "is valid.")

        # check that the given permutations only contain target indices
        target_indices = self.target_indices
        if any(s not in target_indices
               for s in chain.from_iterable(permutations)):
            raise NotImplementedError("Found non target index in "
                                      f"{permutations}. Target indices are "
                                      f"{target_indices}.")

        map_contribution = {}
        for has_denom, term_i_list in self._prescan_terms():
            # go through the terms and filter out terms that are symmetric or
            # antisymmetric with respect to the given symmetry
            relevant_terms = []
            for term_i in term_i_list:
                term: e.Term = self._terms[term_i]
                perm_term: e.Expr = term.permute(*permutations)
                # check that the permutations are valid
                if perm_term.sympy is S.Zero and term.sympy is not S.Zero:
                    continue
                # only look for the desired symmetry which is defined by
                # sym_factor
                if sym_factor == -1:  # looking for antisym: P_pq X != -X
                    if perm_term.sympy + term.sympy is not S.Zero:
                        relevant_terms.append((term_i, perm_term))
                else:  # looking for sym: P_pq X != X
                    if perm_term.sympy - term.sympy is not S.Zero:
                        relevant_terms.append((term_i, perm_term))
            # choose a function for simplifying the sum/difference of 2 terms
            # it might be neccessary to permute contracted indices to
            # achieve equality of the 2 terms
            simplify_terms = simplify_with_denom if has_denom else simplify
            # now compare all relevant terms with each other
            for term_i, perm_term in relevant_terms:
                for other_term_i, _ in relevant_terms:
                    if term_i == other_term_i:  # dont compare to itself
                        continue
                    # looking for antisym: X - (P_pq X) = X - X'
                    # P_pq X + (- X') = 0
                    if sym_factor == -1:
                        sum = simplify_terms(
                            perm_term + self._terms[other_term_i]
                        )
                    # looking for sym: X + (P_pq X) = X + X'
                    # P_pq X - X' = 0
                    else:  # +1
                        sum = simplify_terms(
                            perm_term - self._terms[other_term_i]
                        )
                    # was it possible to map the terms onto each other?
                    if sum.sympy is S.Zero:
                        map_contribution[term_i] = other_term_i
                        # can break the loop: if we are assuming that the
                        # expression is completely simplified, it will not
                        # be possible to find another match for term_i
                        # (otherwise 2 other_term_i would have to be identical)
                        break
        self._term_map[(tuple(permutations), sym_factor)] = map_contribution
        return map_contribution
