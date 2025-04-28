from collections import defaultdict
import itertools

from sympy import Add, S

from .eri_orbenergy import EriOrbenergy
from .expression import ExprContainer, TermContainer
from .indices import get_symbols, sort_idx_canonical
from .misc import Inputerror
from .simplify import simplify
from .symmetry import Permutation
from .sympy_objects import AntiSymmetricTensor, KroneckerDelta, SymmetricTensor


def by_delta_types(expr: ExprContainer
                   ) -> dict[tuple[str, ...], ExprContainer]:
    """Sort the terms in an expression according to their space and spin."""
    assert isinstance(expr, ExprContainer)
    expr = expr.expand()
    ret: dict[tuple[str, ...], ExprContainer] = {}
    for term in expr.terms:
        d_blocks = []
        for delta in term.objects:
            if not isinstance(delta.base, KroneckerDelta):
                continue
            spin = delta.spin
            if all(c == "n" for c in spin):  # no indices with spin
                block = delta.space
            else:
                block = f"{delta.space}_{spin}"
            exp = delta.exponent
            assert exp.is_Integer
            d_blocks.extend(block for _ in range(int(exp)))
        d_blocks = tuple(sorted(d_blocks))
        if not d_blocks:
            d_blocks = ('none',)
        if d_blocks not in ret:
            ret[d_blocks] = ExprContainer(0, **term.assumptions)
        ret[d_blocks] += term
    return ret


def by_delta_indices(expr: ExprContainer
                     ) -> dict[tuple[str, ...], ExprContainer]:
    """
    Sort the terms in an expression according to the names and spin of indices
    on the KroneckerDeltas in each term.
    """
    assert isinstance(expr, ExprContainer)
    expr = expr.expand()
    ret: dict[tuple[str, ...], ExprContainer] = {}
    for term in expr.terms:
        d_idx = tuple(sorted(
            "".join(str(s) for s in o.idx) for o in term.objects
            if isinstance(o.base, KroneckerDelta)
            for _ in range(int(o.exponent))
        ))
        if not d_idx:
            d_idx = ('none',)
        if d_idx not in ret:
            ret[d_idx] = ExprContainer(0, **term.assumptions)
        ret[d_idx] += term
    return ret


def by_tensor_block(expr: ExprContainer, t_name: str
                    ) -> dict[tuple[str, ...], ExprContainer]:
    """
    Sort the terms in an expression according to the blocks of a tensor.
    """
    assert isinstance(t_name, str)
    assert isinstance(expr, ExprContainer)
    expr = expr.expand()
    ret: dict[tuple[str, ...], ExprContainer] = {}
    for term in expr.terms:
        t_blocks = []
        for tensor in term.objects:
            if tensor.name != t_name:
                continue
            spin = tensor.spin
            if all(c == "n" for c in spin):
                block = tensor.space
            else:
                block = f"{tensor.space}_{spin}"
            exp = tensor.exponent
            assert exp.is_Integer
            t_blocks.extend(block for _ in range(int(exp)))
        t_blocks = tuple(sorted(t_blocks))
        if not t_blocks:
            t_blocks = ("none",)
        if t_blocks not in ret:
            ret[t_blocks] = ExprContainer(0, **term.assumptions)
        ret[t_blocks] += term
    return ret


def by_tensor_target_block(expr: ExprContainer, t_name: str
                           ) -> dict[tuple[str, ...], ExprContainer]:
    """
    Sort the terms in an expression according to the type of target indices on
    the specified tensor, e.g. f_cc Y_ij^ac, where i, j and a are target
    indices:
    -> if sorting according to the indices on Y: (oov,);
    if sorting acording to the indices on f: (none,).
    """
    assert isinstance(t_name, str)
    assert isinstance(expr, ExprContainer)
    expr = expr.expand()
    ret: dict[tuple[str, ...], ExprContainer] = {}
    for term in expr.terms:
        key = []
        target = term.target
        for tensor in term.objects:
            if tensor.name == t_name:
                # indices are in canonical order
                tensor_target = [s for s in tensor.idx if s in target]
                if not tensor_target:  # no target indices on the tensor
                    key.append("none")
                    continue
                tensor_target_block = "".join(
                    s.space[0] for s in tensor_target
                )
                if any(s.spin for s in tensor_target):  # spin is defined
                    spin = "".join(
                        s.spin if s.spin else "n" for s in tensor_target
                    )
                    tensor_target_block += f"_{spin}"
                key.append(tensor_target_block)
        key = tuple(sorted(key))  # in case of multiple occurences
        if not key:  # did not find a single occurence of the tensor
            key = (f'no_{t_name}',)
        if key not in ret:
            ret[key] = ExprContainer(0, **term.assumptions)
        ret[key] += term
    return ret


def by_tensor_target_indices(expr: ExprContainer, t_name: str
                             ) -> dict[tuple[str, ...], ExprContainer]:
    """
    Sort the terms in an expression according to the names of target indices on
    the specified tensor.
    """
    assert isinstance(t_name, str)
    assert isinstance(expr, ExprContainer)
    expr = expr.expand()
    ret: dict[tuple[str, ...], ExprContainer] = {}
    for term in expr.terms:
        key = []
        target = term.target
        for obj in term.objects:
            if obj.name == t_name:
                # indices are in canonical order
                obj_target_idx = "".join(
                    [s.name for s in obj.idx if s in target]
                )
                if not obj_target_idx:
                    obj_target_idx = "none"
                key.append(obj_target_idx)
        key = tuple(sorted(key))  # in case the tensor occurs more than once
        if not key:  # tensor did not occur in the term
            key = (f"no_{t_name}",)
        if key not in ret:
            ret[key] = ExprContainer(0, **term.assumptions)
        ret[key] += term
    return ret


def exploit_perm_sym(
        expr: ExprContainer, target_indices: str | None = None,
        target_spin: str | None = None, bra_ket_sym: int = 0,
        antisymmetric_result_tensor: bool = True
        ) -> dict[tuple[tuple[tuple[Permutation, ...], int], ...], ExprContainer]:  # noqa E501
    """
    Reduces the number of terms in an expression by exploiting the symmetry:
    by applying permutations of target indices it might be poossible to map
    terms onto each other reducing the overall number of terms.

    Parameters
    ----------
    expr : Expr
        The expression to probe for symmetry.
    target_indices : str | None, optional
        The names of target indices of the expression. Bra and ket indices
        should be separated by a ',' to lower the amount of permutations the
        expression has to be probed for, e.g., to differentiate 'ia,jb'
        from 'ij,ab'. If not provided, the function will try to determine the
        target indices automatically and probe for the complete symmetry found
        for these indices.
    target_spin : str | None , optional
        The spin of the target indices, e.g., 'aabb' to indicate that the
        first 2 target indices have alpha spin, while number 3 and 4 have
        beta spin. If not given, target indices without spin will be used.
    bra_ket_sym : int, optional
        Defines the bra-ket symmetry of the result tensor of the expression.
        Only considered if the names of target indices are separated by a ','.
    antisymmetric_result_tensor : bool, optional
        If set, the result tensor will be treated as AntiSymmetricTensor
        d_{ij}^{ab} = - d_{ji}^{ab}. Otherwise, a SymmetricTensor will be used
        to mimic the symmetry of the result tensor, i.e.,
        d_{ij}^{ab} = d_{ji}^{ab}. (default: True)

    Returns
    -------
    dict
        The remaining terms sorted by the found permutations.
        key: The permutations.
        value: The part of the expression to which the permutations have to be
               applied in order to recover the original expression.
    """
    from .reduce_expr import factor_eri_parts, factor_denom

    def simplify_terms_with_denom(sub_expr: ExprContainer):
        factored = itertools.chain.from_iterable(
            factor_denom(sub_e) for sub_e in factor_eri_parts(sub_expr)
        )
        ret = ExprContainer(0, **sub_expr.assumptions)
        for term in factored:
            ret += term.factor()
        return ret

    assert isinstance(expr, ExprContainer)
    if expr.inner.is_number:
        return {tuple(): expr}
    expr.expand()
    terms: tuple[TermContainer, ...] = expr.terms

    # check that each term in the expr contains the same target indices
    ref_target = terms[0].target
    if not expr.provided_target_idx and \
            any(term.target != ref_target for term in terms):
        raise Inputerror("Each term in the expression needs to contain the "
                         "same target indices.")

    # if target indices have been provided
    # -> check that they match with the found target indices
    if target_indices is not None:
        # split in upper/lower indices if possible
        if "," in target_indices:
            upper, lower = target_indices.split(",")
        else:
            if bra_ket_sym:
                raise Inputerror("Target indices need to be separated by a "
                                 "',' to indicate where to split them in "
                                 "upper and lower indices if the target tensor"
                                 "has bra-ket-symmetry.")
            upper, lower = target_indices, ""
        # treat the spin
        if target_spin is not None:
            if "," in target_spin:
                upper_spin, lower_spin = target_spin.split(",")
            else:
                upper_spin = target_spin[:len(upper)]
                lower_spin = target_spin[len(upper):]
            if len(upper) != len(upper_spin) or len(lower) != len(lower_spin):
                raise Inputerror(f"The target indices {target_indices} are "
                                 " not compatible with the provided spin "
                                 f"{target_spin}.")
        else:
            upper_spin, lower_spin = None, None

        upper = get_symbols(upper, upper_spin)
        lower = get_symbols(lower, lower_spin)
        sorted_provided_target = tuple(sorted(
            upper + lower, key=sort_idx_canonical
        ))
        if sorted_provided_target != ref_target:
            raise Inputerror(f"The provided target indices {target_indices} "
                             "are not equal to the target indices found in "
                             f"the expr: {ref_target}.")
    else:  # just use the found target indices
        # if no target indices have been provided all indices are in upper
        # -> bra ket sym is irrelevant
        upper, lower = ref_target, tuple()
        bra_ket_sym = 0
    # build a tensor holding the target indices and determine its symmetry
    if antisymmetric_result_tensor:
        tensor = AntiSymmetricTensor("x", upper, lower, bra_ket_sym)
    else:
        tensor = SymmetricTensor("x", upper, lower, bra_ket_sym)
    symmetry = ExprContainer(tensor).terms[0].symmetry()

    # prefilter the terms according to the contained objects (name, space, exp)
    # and if a denominator is present -> number and length of the brackets
    filtered_terms: defaultdict[tuple, list[int]] = defaultdict(list)
    has_denom: list[bool] = []
    for term_i, term in enumerate(terms):
        term_splitted = EriOrbenergy(term)
        has_denom.append(not term_splitted.denom.inner.is_number)
        eri_descr: tuple[str, ...] = tuple(sorted(
            o.description(target_idx=None)
            for o in term_splitted.eri.objects
        ))
        idx_space = "".join(sorted(
            s.space[0] + s.spin for s in term_splitted.eri.contracted
        ))
        key = (eri_descr, term_splitted.denom_description(), idx_space)
        filtered_terms[key].append(term_i)

    ret: dict[tuple[tuple[tuple[Permutation, ...], int], ...], ExprContainer] = {}  # noqa E501
    removed_terms: set[int] = set()
    for term_idx_list in filtered_terms.values():
        # term is unique -> nothing to compare with
        # can not map this term onto any other terms
        if len(term_idx_list) == 1:
            if tuple() not in ret:
                ret[tuple()] = ExprContainer(0, **expr.assumptions)
            ret[tuple()] += terms[term_idx_list[0]]
            continue

        # decide which function to use for comparing the terms
        terms_have_denom = has_denom[term_idx_list[0]]
        assert all(
            terms_have_denom == has_denom[term_i] for term_i in term_idx_list
        )
        if terms_have_denom:
            simplify_terms = simplify_terms_with_denom
        else:
            simplify_terms = simplify

        # first loop over terms!!
        # Otherwise it is not garuanteed that all matches for a term can
        # be found: consider 4 terms with ia, ja, ib and jb
        # we want to find: P_ab, P_ij and P_ijP_ab for ia (or any other term)
        # if we first loop over perms, e.g., P_ab we may find
        # ia -> ib, ja -> jb for instance.
        #  -> we will not be able to find the full symmetry of the terms
        for term_i in term_idx_list:
            if term_i in removed_terms:
                continue
            term = terms[term_i]
            found_sym: list[tuple[tuple[Permutation, ...], int]] = []
            for perms, factor in symmetry.items():
                # apply the permutations to the current term
                perm_term = term.permute(*perms)
                # permutations are not valid for the current term
                if perm_term.inner is S.Zero and term.inner is not S.Zero:
                    continue
                # check if the permutations did change the term
                # if the term is still the same (up to the sign) continue
                # thereby only looking for the desired symmetry
                if factor == -1:
                    # looking for antisym: P_pq X = - X -> P_pq X + X = 0?
                    if Add(perm_term.inner, term.inner) is S.Zero:
                        continue
                elif factor == 1:
                    # looking for sym: P_pq X = + X -> P_pq X - X = 0?
                    if Add(perm_term.inner, -term.inner) is S.Zero:
                        continue
                else:
                    raise ValueError(f"Invalid sym factor {factor}.")
                # perm term != term -> compare to other terms
                for other_term_i in term_idx_list:
                    if term_i == other_term_i or other_term_i in removed_terms:
                        continue
                    # compare the terms: again only look for the desired
                    # symmetry
                    if factor == -1:
                        # looking for antisymmetry: X - X'
                        # P_pq X + (-X') = 0   | P_pq X = +X'
                        simplified = (
                            simplify_terms(perm_term + terms[other_term_i])
                        )
                    else:  # factor == 1
                        # looking for symmetry: X + (X')
                        # P_pq X - X' = 0  | P_pq X = +X'
                        simplified = (
                            simplify_terms(perm_term - terms[other_term_i])
                        )
                    # could not map the terms onto each other
                    if simplified.inner is not S.Zero:
                        continue
                    # mapped the terms onto each other
                    removed_terms.add(other_term_i)
                    found_sym.append((perms, factor))
                    break
            # use the found symmetry as dict key
            found_sym_tpl = tuple(found_sym)
            if found_sym_tpl not in ret:
                ret[found_sym_tpl] = ExprContainer(0, **expr.assumptions)
            ret[found_sym_tpl] += term
    return ret
