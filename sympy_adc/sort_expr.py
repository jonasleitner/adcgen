from . import expr_container as e
from .misc import Inputerror
from .indices import index_space, get_symbols, idx_sort_key


def by_delta_types(expr: e.expr) -> dict[tuple[str], e.expr]:
    """Sort the terms in an expression according to the space of the deltas
       contained in the term."""
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)
    ret = {}
    for term in expr.terms:
        d_spaces = tuple(sorted([o.space for o in term.deltas
                         for _ in range(o.exponent)]))
        if not d_spaces:
            d_spaces = ('none',)
        if d_spaces not in ret:
            ret[d_spaces] = e.expr(0, **term.assumptions)
        ret[d_spaces] += term
    return ret


def by_delta_indices(expr: e.expr) -> dict[tuple[str], e.expr]:
    """Sort the terms in an expression according to the indices on the
       KroneckerDeltas in the term."""
    from .indices import extract_names
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)
    ret = {}
    for term in expr.terms:
        d_idx = tuple(sorted(
            ["".join(extract_names(o.idx)) for o in term.deltas
             for _ in range(o.exponent)]
        ))
        if not d_idx:
            d_idx = ('none',)
        if d_idx not in ret:
            ret[d_idx] = e.expr(0, **term.assumptions)
        ret[d_idx] += term
    return ret


def by_tensor_block(expr: e.expr, t_string: str,
                    bra_ket_sym: int = None) -> dict[tuple[str], e.expr]:
    """Sorts the terms in an expression according to the blocks of a tensor,
       e.g. collect all terms that define the 'oo' block of the density matrix.
       If the desired tensor occures more than once in a term, the sorted list
       of blocks will be used."""

    if not isinstance(t_string, str):
        raise Inputerror("Tensor name must be provided as string.")
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)
    # ckeck that there is no contradiction with the bra_ket symmetry and
    # set the symmetry in the expr if it is not already set
    if bra_ket_sym is not None:
        if bra_ket_sym not in [0, 1, -1]:
            raise Inputerror(f"Invalid bra_ket symmetry {bra_ket_sym}. Valid "
                             "are 0, 1 and -1.")
        sym_tensors = expr.sym_tensors
        antisym_tensors = expr.antisym_tensors
        if (bra_ket_sym == 0 and t_string in sym_tensors + antisym_tensors) \
                or (bra_ket_sym == 1 and t_string in antisym_tensors) or \
                (bra_ket_sym == -1) and t_string in sym_tensors:
            raise ValueError("The set tensor symmetry in the expression and "
                             "the provided bra_ket symmetry are not "
                             "compatible.")
        elif bra_ket_sym == 1 and t_string not in sym_tensors:
            # set the tensor as symemtric
            sym_tensors = sym_tensors + (t_string,)
            expr.set_sym_tensors(sym_tensors)
        elif bra_ket_sym == -1 and t_string not in antisym_tensors:
            # set the tensor to be antisymmetric
            antisym_tensors = antisym_tensors + (t_string,)
            expr.set_antisym_tensors(antisym_tensors)

    ret = {}
    for term in expr.terms:
        blocks = tuple(sorted(
            [t.space for t in term.tensors for _ in range(t.exponent)
             if t.name == t_string]
        ))
        if not blocks:
            blocks = ("none",)
        if blocks not in ret:
            ret[blocks] = e.expr(0, **term.assumptions)
        ret[blocks] += term
    return ret


def by_tensor_target_space(expr: e.expr, t_string: str) -> dict[tuple[str], e.expr]:  # noqa E501
    """Sorts the terms in an expression according to the number and type of
       target indices on a specified tensor, e.g. f_cc Y_ij^ac -> if sorting
       according to Y: oov; if sorting acording to f: none.
       """

    if not isinstance(t_string, str):
        raise Inputerror("Tensor name must be provided as string.")
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)

    ret = {}
    for term in expr.terms:
        key = []
        target = term.target
        for o in term.tensors:
            if o.name == t_string:
                # indices are in canonical order -> the space should also
                obj_target_sp = "".join(
                    [index_space(s.name)[0] for s in o.idx if s in target]
                )
                if not obj_target_sp:
                    obj_target_sp = 'none'
                key.append(obj_target_sp)
        key = tuple(sorted(key))  # in case of multiple occurences
        if not key:  # did not find a single occurence of the tensor
            key = (f'no_{t_string}',)
        if key not in ret:
            ret[key] = 0
        ret[key] += term
    return ret

def by_tensor_target_idx(expr: e.expr, t_string: str) -> dict[tuple[str], e.expr]:  # noqa E501
    """Sorts the terms in an expression according to the target indices on
       the specified tensor."""

    if not isinstance(t_string, str):
        raise Inputerror("Tensor name must be provided as string.")
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)

    ret = {}
    for term in expr.terms:
        key = []
        target = term.target
        for obj in term.tensors:
            if obj.name == t_string:
                # indices are in canonical order
                obj_target_idx = "".join(
                    [s.name for s in obj.idx if s in target]
                )
                if not obj_target_idx:
                    obj_target_idx = "none"
                key.append(obj_target_idx)
        key = tuple(sorted(key))  # in case the tensor occurs more than once
        if not key:  # tensor did not occur in the term
            key = (f"no_{t_string}",)
        if key not in ret:
            ret[key] = 0
        ret[key] += term
    return ret


def exploit_perm_sym(expr: e.expr, target_indices: str = None,
                     bra_ket_sym: int = 0) -> dict[tuple, e.expr]:
    """Reduces the number of terms in an expression by exploiting its
       symmetry, i.e., splits the expression in sub expressions that
       are assigned to specific permutations. Applying those permutations
       to the sub expressions regenerates the full expression.
       To speed up this process the target indices can be provided as str,
       where a comma separates the bra and ket indices. In combination with the
       bra ket symmetry this reduces the amount of permutations
       the expression is probed for, e.g., the target indices ijab may either
       belong to a doubles vector with ij and ab antisym or to the
       ph/ph block of the secular matrix with ij-ab sym.
       """
    from .sympy_objects import AntiSymmetricTensor
    from .eri_orbenergy import eri_orbenergy
    from .simplify import simplify
    from .reduce_expr import factor_eri_parts, factor_denom
    from itertools import chain
    from sympy import S
    from collections import defaultdict

    def simplify_terms_with_denom(sub_expr: e.expr):
        factored = chain.from_iterable(
            factor_denom(sub_e) for sub_e in factor_eri_parts(sub_expr)
        )
        ret = e.expr(0, **sub_expr.assumptions)
        for term in factored:
            ret += term.factor()
        return ret

    if not isinstance(expr, e.expr):
        raise Inputerror("Expression needs to be provided as expr instance.")

    if expr.sympy.is_number:
        return {tuple(): expr}

    expr: e.expr = expr.expand()
    terms: list[e.term] = expr.terms

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
        if ',' in target_indices:
            upper, lower = target_indices.split(',')
        else:
            if bra_ket_sym:
                raise Inputerror("Target indices need to be separated by a "
                                 "',' to indicate where to separate them in "
                                 "upper and lower indices if the target tensor"
                                 "has bra-ket-symmetry.")
            upper, lower = target_indices, ""
        upper, lower = get_symbols(upper), get_symbols(lower)
        sorted_provided_target = tuple(sorted(upper + lower, key=idx_sort_key))
        if sorted_provided_target != ref_target:
            raise Inputerror(f"The provided target indices {target_indices} "
                             "are not equal to the target indices found in "
                             f"the expr: {ref_target}.")
    else:  # just use the found target indices
        upper, lower = ref_target, tuple()
    # build a tensor holding the target indices and determine its symmetry
    # if no target indices have been provided all indices are in upper
    # -> bra ket sym is irrelevant
    tensor = AntiSymmetricTensor('x', upper, lower, bra_ket_sym)
    symmetry = e.expr(tensor).terms[0].symmetry()

    # prefilter the terms according to the contained objects (name, space, exp)
    # and if a denominator is present -> number and length of the brackets
    filtered_terms = defaultdict(list)
    has_denom: list[bool] = []
    for term_i, term in enumerate(terms):
        term = eri_orbenergy(term)
        has_denom.append(not term.denom.is_number)
        eri_descr: tuple[str] = tuple(sorted(
            o.description(include_target_idx=False)
            for o in term.eri.objects
        ))
        idx_space = "".join(sorted(
            index_space(s.name)[0] for s in term.eri.contracted
        ))
        key = (eri_descr, term.denom_description(), idx_space)
        filtered_terms[key].append(term_i)

    ret = {}
    removed_terms = set()
    for term_idx_list in filtered_terms.values():
        # term is unique -> nothing to compare with
        # can not map this term onto any other terms
        if len(term_idx_list) == 1:
            if tuple() not in ret:
                ret[tuple()] = 0
            ret[tuple()] += terms[term_idx_list[0]]
            continue

        # decide which function to use for comparing the terms
        terms_have_denom = has_denom[term_idx_list[0]]
        assert all(terms_have_denom == has_denom[term_i]
                   for term_i in term_idx_list)
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
            term: e.expr = terms[term_i]
            found_sym = []
            for perms, factor in symmetry.items():
                # apply the permutations to the current term
                perm_term = term.permute(*perms)
                # permutations are not valid for the current term
                if perm_term.sympy is S.Zero and term.sympy is not S.Zero:
                    continue
                # check if the permutations did change the term
                # if the term is still the same (up to the sign) continue
                # thereby only looking for the desired symmetry
                if factor == -1:
                    # looking for antisym: P_pq X = - X -> P_pq X + X = 0?
                    if perm_term.sympy + term.sympy is S.Zero:
                        continue
                elif factor == 1:
                    # looking for sym: P_pq X = + X -> P_pq X - X = 0?
                    if perm_term.sympy - term.sympy is S.Zero:
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
                    if simplified.sympy is not S.Zero:
                        continue
                    # mapped the terms onto each other
                    removed_terms.add(other_term_i)
                    found_sym.append((perms, factor))
                    break
            # use the found symmetry as dict key
            found_sym = tuple(found_sym)

            if found_sym not in ret:
                ret[found_sym] = 0
            ret[found_sym] += term
    return ret
