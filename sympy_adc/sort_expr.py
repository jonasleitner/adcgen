from . import expr_container as e
from .misc import Inputerror
from .indices import index_space


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


def by_tensor_target_idx(expr: e.expr, t_string: str) -> dict[tuple[str], e.expr]:  # noqa E501
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
            ret[key] = e.compatible_int(0)
        ret[key] += term
    return ret


def exploit_perm_sym(expr: e.expr, target_upper: str = None,
                     target_lower: str = None,
                     target_bra_ket_sym: int = 0) -> dict[tuple, e.expr]:
    from .indices import get_symbols
    from .sympy_objects import AntiSymmetricTensor
    from sympy import S

    def collect_denom_terms(sub_expr: e.expr):
        from .reduce_expr import factor_eri, factor_denom
        from itertools import chain

        factored_expr = chain.from_iterable(
            factor_denom(sub_e) for sub_e in factor_eri(sub_expr)
        )
        factored_expr = [factored.factor() for factored in factored_expr]
        ret = e.expr(0, **sub_expr.assumptions)
        for term in factored_expr:
            ret += term
        return ret.expand()

    def collect_terms(sub_expr: e.expr):
        from .simplify import simplify
        return simplify(sub_expr)

    if not isinstance(expr, e.expr):
        raise Inputerror("Expression needs to be provided as expr instance.")
    expr = expr.expand()

    if expr.sympy.is_number:
        return {tuple(): expr}

    # iterate over all terms split according to denom and determine target
    # indices
    terms = expr.terms
    term_data = []
    target = terms[0].target
    for term in terms:
        if term.target != target:
            raise ValueError("Target indices need to be equal for all terms. "
                             f"Found {term.target} in current term and "
                             f"{target} in the first term")
        has_denom = True if term.polynoms else False
        tensors = tuple(sorted((t.name, t.space) for t in term.tensors
                               for _ in range(t.exponent)))
        deltas = tuple(
            sorted([d.space for d in term.deltas for _ in range(d.exponent)])
        )
        term_data.append((has_denom, tensors, deltas))

    # determine the permutations that needs to be probed
    # e.g. M_ia,jb -> M^ia_jb -> only + P_ij P_ab perm required
    #      r_ijab  -> only - P_ij and - P_ab required.
    # both have target indices ijab. need to know how to separate the target
    # indices in order to avoid probing unnecessary permutations
    if target_lower or target_upper:
        upper = tuple() if target_upper is None else get_symbols(target_upper)
        lower = tuple() if target_lower is None else get_symbols(target_lower)
        canonical_provided = sorted(lower + upper, key=lambda s:
                                    (int(s.name[1:]) if s.name[1:] else 0,
                                     s.name[0]))
        if tuple(canonical_provided) != target:
            raise Inputerror(f"the provided target indices {target_upper} and "
                             f"{target_lower} do not agree with the found "
                             f"target indices {target}.")
        target_tensor = AntiSymmetricTensor('x', upper, lower,
                                            target_bra_ket_sym)
        symmetry = e.expr(target_tensor).terms[0].symmetry()
    else:  # can not separate spaces -> possibly unnecessary permutations
        target_tensor = AntiSymmetricTensor('x', tuple(), target)
        symmetry = e.expr(target_tensor).terms[0].symmetry()

    # go through all permutations for each term and check whether permuted
    # term is identical to another term of the expr.
    # if this is the case: save the permutations and the found factor (+-1)
    #                      and remove the other term from the expr
    ret = {}
    removed_terms = set()
    for i, (term, data) in enumerate(zip(terms, term_data)):
        if i in removed_terms:
            continue
        # get the indices of all similar terms
        terms_to_compare = [
            other_i for other_i, other_data in enumerate(term_data)
            if (i != other_i and other_i not in removed_terms and
                data == other_data)
        ]
        if not terms_to_compare:
            if tuple() not in ret:
                ret[tuple()] = e.compatible_int(0)
            ret[tuple()] += term
            continue

        # reduce the number of permutations to check - if a symmetry is already
        # ineherent to the term -> no need to check for that.
        term_sym = term.symmetry(only_target=True)
        perms_to_check = [(k, v) for k, v in symmetry.items()
                          if k not in term_sym]

        # depending on the presence of a denominator choose the appropriate
        # function to try to collect two terms
        collect = collect_denom_terms if data[0] else collect_terms

        found_perms = []
        for perms, _ in perms_to_check:
            perm_term = term.permute(*perms)
            for other_i in terms_to_compare:
                if other_i in removed_terms:
                    continue
                sum = collect(perm_term + terms[other_i])
                if sum.sympy is S.Zero:  # P_pq X + (- P_pq X) = 0
                    term_factor = -1
                elif len(sum) == 1:  # P_pq X + (+ P_pq X) = 2 P_pq X
                    term_factor = +1
                else:
                    continue
                removed_terms.add(other_i)
                found_perms.append((perms, term_factor))
                break

        if (key := tuple(found_perms)) not in ret:
            ret[key] = e.compatible_int(0)
        ret[tuple(found_perms)] += term
    return ret
