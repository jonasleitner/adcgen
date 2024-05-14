from .indices import (get_symbols, order_substitutions, Index,
                      get_lowest_avail_indices, minimize_tensor_indices)
from .misc import Inputerror
from .sympy_objects import (
    KroneckerDelta, Amplitude, AntiSymmetricTensor, NonSymmetricTensor
)
from . import expr_container as e

from sympy import Add, Pow, S, sqrt, Rational
from collections import Counter, defaultdict


def filter_tensor(expr, t_strings: list[str], strict: str = 'low',
                  ignore_amplitudes: bool = True) -> e.Expr:
    """
    Filter an expression keeping only terms that contain the desired tensors.

    Parameters
    ----------
    t_strings : list[str]
        List containing the desired tensor names.
    struct : str, optional
        3 possible options:
        - 'high': return all terms that ONLY contain the desired tensors the
                  requested amount of times, e.g., ['V', 'V'] returns only
                  terms that contain not other tensors than 'V*V'
                  Setting ignore_amplitudes, ignores all not requested
                  t and ADC ampltiudes amplitudes.
        - 'medium': return all terms that contain the desired tensors the
                    requested amount, but other tensors may additionally be
                    present in the term. E.g. ['V', 'V'] also returns terms
                    that contain 'V*V*x', where x may be any amount of
                    arbitrary other tensors.
        - 'low': return all terms that contain all of the requested tensors,
                 e.g., ['V', 'V'] returns all terms that contain 'V' at least
                 once.

    Returns
    Expr
        The filtered expression.
    """

    def check_term(term):
        available = [o.name for o in term.tensors for _ in range(o.exponent)]
        # True if all requested tensors are in the term
        if strict == 'low':
            return all(t in available for t in set(t_strings))
        # True if all requested Tensors occure the correct amount of times
        elif strict == 'medium':
            available = Counter(available)
            desired = Counter(t_strings)
            return desired.items() <= available.items()
        # True if only the requested Tensors are in the term in the correct
        # amount
        elif strict == 'high':
            if ignore_amplitudes:
                requested_amplitudes = [
                    name for name in t_strings if name[0] in ['X', 'Y'] or
                    (name[0] == 't' and name[1:].replace('c', '').isnumeric())
                ]
                ignored_amplitudes = {name for name in available
                                      if name[0] in ['t', 'X', 'Y'] and
                                      name not in requested_amplitudes}
                available = Counter([t for t in available
                                     if t not in ignored_amplitudes])
            else:
                available = Counter(available)
            desired = Counter(t_strings)
            return desired == available

    if not all(isinstance(t, str) for t in t_strings):
        raise Inputerror("Tensor names need to be provided as str.")
    if strict not in ['low', 'medium', 'high']:
        raise Inputerror(f"{strict} is not a valid option for strict. Valid"
                         "options are 'low', 'medium' or 'high'.")

    expr = expr.expand()
    if not isinstance(expr, e.Expr):
        expr = e.Expr(expr)

    filtered = Add(*[term.sympy for term in expr.terms if check_term(term)])
    return e.Expr(filtered, **expr.assumptions)


def find_compatible_terms(terms: list[e.Term]) -> dict:
    """
    Determines the substitutions of contracted needed to map terms onto each
    other.

    Parameters
    ----------
    terms: list[Term]
        The list of terms to compare and map onto each other.

    Returns
    -------
    dict
        Nested dictionary containing the indices of terms and the substitution
        dict to map the terms onto each other, e.g., the substitutions to
        map term j onto term i are stored as
        {i: {j: substitutions}}.
        If it was not possible to find a match for term_i, the inner dictionary
        will be empty {i: {}}.
    """
    from itertools import product, combinations

    def compare_terms(pattern: dict, other_pattern: dict, target: tuple,
                      term: e.Term, other_term: e.Term) -> None | list:
        # function to compare two terms that are compatible, i.e., have the
        # same amount of indices in each space, the same amount and type of
        # objects and the same target indices
        sub_list: list[dict] = []
        for ov, idx_pattern in pattern.items():
            # only compare indices that belong to the same space
            other_idx_pattern = other_pattern[ov]
            # list to hold the substitution dictionaries of the current space
            ov_sub_list: list[dict] = []

            for idx, pat in idx_pattern.items():
                # find all possible matches for the current idx
                # if its a target idx -> only allow mapping on other target idx
                is_target = idx in target
                matching_idx = []  # list to collect all possible matches
                for other_idx, other_pat in other_idx_pattern.items():
                    other_is_target = other_idx in target
                    # only 1 index is a target index -> cant map
                    # or both are different target indices
                    # -> cant map because we cant substitute target indices
                    if is_target != other_is_target or \
                            (is_target and other_is_target and
                             idx is not other_idx):
                        continue
                    # the pattern of both indices is identical
                    # -> possible match
                    if pat == other_pat:
                        matching_idx.append(other_idx)
                # could not find a match for idx -> no need to check further
                if not matching_idx:
                    break

                if not ov_sub_list:  # initialize the subdicts
                    ov_sub_list.extend({s: idx} for s in matching_idx)
                else:  # already initialized -> add when possible
                    new_ov_sub_list = []
                    for sub, other_idx in product(ov_sub_list, matching_idx):
                        # other_idx is already mapped onto another idx
                        if other_idx in sub:
                            continue
                        # copy the sub_dict to avoid inplace modification
                        extended_sub = sub.copy()
                        extended_sub[other_idx] = idx
                        new_ov_sub_list.append(extended_sub)
                    ov_sub_list = new_ov_sub_list
                    if not ov_sub_list:  # did not find any valid combination
                        # will not be able to construct complete sub dicts
                        # say we matched idx1 to some indices and then obtain
                        # no valid sub dicts after matching idx2
                        # -> can only obtain sub dicts that do not contain idx1
                        #    and idx2 -> they can not be valid
                        # -> terms can not match!
                        return None
            # Done with comparing the indices of a space
            # -> check the result and create total substitution dicts

            # remove incomplete sub lists
            # This might not be necessary anymore
            ov_sub_list = [sub for sub in ov_sub_list if
                           sub.keys() == other_idx_pattern.keys()]

            if not ov_sub_list:  # did not find a single complete sub dict
                return None

            # initialize the final substitution dicts
            if not sub_list:
                sub_list.extend(ov_sub_list)
            else:  # combine the sub dicts. different spaces can not overlap
                sub_list = [other_sp_sub | sub for other_sp_sub, sub in
                            product(sub_list, ov_sub_list)]

        # test all sub dicts to identify the correct one (if one exists)
        for sub in sub_list:
            sub = order_substitutions(sub)
            sub_other_term = other_term.sympy.subs(sub)
            # sub is not valid for other term: evaluates to 0 due to
            # some antisymmetry e.g. t_ijcd -> t_ijcc = 0
            if sub_other_term is S.Zero and other_term.sympy is not S.Zero:
                continue
            # diff (or sum) is a single term (no Add obj)
            # can either sum up to 0 or to a single term with a different pref
            # -> check for type of result and not for result value
            if not isinstance(term.sympy - sub_other_term, Add):
                return sub
        return None  # no valid sub dict -> return None

    def repeating_idx_sp(idx_list: list):
        repeating_idx = []
        for idx1, idx2 in combinations(idx_list, 2):
            descr1, descr2 = idx1[0], idx2[0]
            for i1, i2 in product(idx1[1:], idx2[1:]):
                repeated = i1 & i2
                if len(repeated) > 1:
                    repeated = "".join(sorted(s.space[0] + s.spin
                                              for s in repeated))
                    repeating_idx.append((repeated, *sorted([descr1, descr2])))
        return tuple(sorted(repeating_idx))

    if not all(isinstance(term, e.Term) for term in terms):
        raise Inputerror("Expected terms as a list of term Containers.")

    # prefilter terms according to
    # - number of objects, excluding prefactor
    # - type, name, space, spin, obj target indices and exponent of objects
    # - the space of repeating indices subsets (2, 3, ...) that repeat on
    #   on multiple objects together in a common index subspace (upper/lower)
    # - number of indices in each space
    # - the target indices
    filtered_terms = defaultdict(list)
    term_pattern = []
    term_target = []
    for term_i, term in enumerate(terms):
        # target indices
        target = term.target
        term_target.append(target)
        # pattern
        pattern = term.pattern
        term_pattern.append(pattern)
        # obj name, space, exponent, obj_target_indices, repeating_indices
        descriptions = []
        tensor_idx_list = []
        length = 0
        for o in term.objects:
            if (descr := o.description()) == 'prefactor':
                continue
            elif 'antisymtensor' in descr:
                tensor = o.base
                tensor_idx_list.append(
                    (descr, set(tensor.upper), set(tensor.lower))
                )
            elif 'delta' in descr or 'nonsymtensor' in descr:
                tensor_idx_list.append((descr, set(o.idx), set()))
            length += 1
            descriptions.append(descr)
        pattern_key = tuple(sorted(
            (sp, len(idx_pat)) for sp, idx_pat in pattern.items()
        ))
        key = (length, tuple(sorted(descriptions)),
               repeating_idx_sp(tensor_idx_list), pattern_key, target)
        filtered_terms[key].append(term_i)

    compatible_terms = {}
    for term_idx_list in filtered_terms.values():
        # set to keep track of the already mapped terms
        matched = set()
        for i, term_i in enumerate(term_idx_list):
            if term_i in matched:  # term already mapped
                continue

            compatible_terms[term_i] = {}

            # data of the current term
            term = terms[term_i]
            target = term_target[term_i]
            pattern = term_pattern[term_i]

            for other_i in range(i+1, len(term_idx_list)):
                other_term_i = term_idx_list[other_i]
                if other_term_i in matched:  # term already mapped
                    continue

                sub = compare_terms(pattern, term_pattern[other_term_i],
                                    target, term, terms[other_term_i])
                # was possible to map the terms onto each other!
                if sub is not None:
                    compatible_terms[term_i][other_term_i] = sub
                    matched.add(other_term_i)
    return compatible_terms


def simplify(expr: e.Expr) -> e.Expr:
    """
    Simplify an expression by permuting contracted indices. Thereby, terms
    are mapped onto each other reducing the number of terms.
    Currently this does not work for denominators of the form (a + b + ...).
    However, this restriction can often be bypassed by using symbolic,
    denominators, i.e., using a tensor of the correct symmetry to represent the
    denominator. Alternatively, the functions found in 'reduce_expr' are
    capable to handle orbital energy denominators.

    Parameters
    ----------
    expr : Expr
        The expression to simplify

    Returns
    -------
    Expr
        The simplified expression.

    Simplify an expression by renaming indices. The new index names are
       determined by establishing a mapping between the indices in different
       terms. If all indices in two terms share the same pattern (essentially
       occur on the same tensors), but have different names. The function will
       rename the indices in one of the two terms.
       """

    if not isinstance(expr, e.Expr):
        raise Inputerror("The expression to simplify needs to be provided as "
                         f"{e.Expr} object.")

    expr = expr.expand()

    if len(expr) == 1:  # trivial: only a single term
        return expr

    # create terms and try to find comaptible terms that may be
    # simplified by substituting indices
    terms = expr.terms
    equal_terms = find_compatible_terms(terms)

    # substitue the indices in other_n and keep n as is
    res = 0
    for n, matches in equal_terms.items():
        res += terms[n]
        for other_n, sub in matches.items():
            res += terms[other_n].subs(sub)
    return res


def simplify_unitary(expr: e.Expr, t_name: str,
                     evaluate_deltas: bool = False) -> e.Expr:
    """
    Simplifies an expression that contains unitary tensors by exploiting
    U_pq * U_pr * Remainder = delta_qr * Remainder,
    where the Remainder does not contain the index p.

    Parameters
    ----------
    expr : Expr
        The expression to simplify.
    t_name : str
        Name of the unitary tensor.
    evaluate_deltas: bool, optional
        If this is set, the generated KroneckerDeltas will be evaluated
        before returning.

    Returns
    -------
    Expr
        The simplified expression.
    """
    from . import func
    from itertools import combinations

    def simplify_term_unitary(term: e.Term) -> e.Term:
        obj = term.objects
        # collect the indices of all unitary tensors in the term
        unitary_tensors = [i for i, o in enumerate(obj) if o.name == t_name
                           for _ in range(o.exponent)]

        # only implemented for 2 dimensional unitary tensors
        if any(len(obj[i].idx) != 2 for i in unitary_tensors):
            raise NotImplementedError("Did only implement the case of 2D "
                                      f"unitary tensors. Found {t_name} in "
                                      f"{term}")

        # TODO: if we have a AntiSymmetricTensor as unitary tensor
        #   -> what kind of bra ket symmetry is possible?
        #   throw an error if it is set to +-1?

        # need at least 2 unitary tensors
        if len(unitary_tensors) < 2:
            return term

        # find the target indices
        target = term.target
        idx_counter = Counter(term.idx)

        # iterate over all pairs and look for matching contracted indices
        # that do only occur on the two unitary tensors we want to simplify
        for (i1, i2) in combinations(unitary_tensors, 2):
            idx1 = obj[i1].idx
            idx2 = obj[i2].idx
            # U_pq U_pr = delta_qr
            if idx1[0] == idx2[0] and idx1[0] not in target and \
                    idx_counter[idx1[0]] == 2:
                delta = KroneckerDelta(idx1[1], idx2[1])
            # U_qp U_rp = delta_qr
            elif idx1[1] == idx2[1] and idx1[1] not in target and \
                    idx_counter[idx1[1]] == 2:
                delta = KroneckerDelta(idx1[0], idx2[0])
            else:  # no matching indices
                continue

            # lower the exponent of the 2 unitary tensors and
            # add the created delta to the term
            new_term = e.Expr(delta, **term.assumptions)
            if i1 == i2:
                base, exponent = obj[i1].base_and_exponent
                new_term *= Pow(base, exponent - 2)
            else:
                b1, exponent1 = obj[i1].base_and_exponent
                b2, exponent2 = obj[i2].base_and_exponent
                new_term *= Pow(b1, exponent1 - 1)
                new_term *= Pow(b2, exponent2 - 1)

            # add remaining objects
            for i, o in enumerate(obj):
                if i == i1 or i == i2:
                    continue
                else:
                    new_term *= o
            return simplify_term_unitary(new_term.terms[0])
        # could not find simplification -> return
        return term

    if not isinstance(expr, e.Expr):
        raise TypeError(f"Expr needs to be provided as {e.Expr}.")

    res = e.Expr(0, **expr.assumptions)
    for term in expr.terms:
        res += simplify_term_unitary(term)

    # evaluate the generated deltas if requested
    if evaluate_deltas:
        res = e.Expr(func.evaluate_deltas(res.sympy), **res.assumptions)
    return res


def remove_tensor(expr: e.Expr, t_name: str) -> dict:
    """
    Removes a tensor from each term of an expression by undoing the contraction
    of the remaining term with the tensor. The resulting expression is split
    according to the blocks of the removed tensor. Note that only canonical
    tensor blocks are considered, because the non-canonical blocks can be
    generated from the canonical ones, e.g., removing a symmetric matrix d_{pq}
    from an expression can only result in expressions for the 'oo', 'ov' and
    'vv' blocks, since f_{ai} = f_{ia}.
    The symmetry of the removed tensor is taken into account, such that the
    original expression can be restored if all block expressions are
    contracted with the corresponding tensor blocks again.

    Parameters
    ----------
    expr : Expr
        The expression where the tensor should be removed.
    t_name : str
        Name of the tensor that should be removed.

    Returns
    -------
    dict
        key: Tuple of removed tensor blocks
        value: Part of the original expression that contained the corresponding
               blocks. If contracted with the tensor blocks again, a part of
               the original expression is recovered.
    """

    def remove(term: e.Term, tensor: e.Obj, target_indices: dict) -> e.Expr:
        # - get the tensor indices
        indices = list(tensor.idx)
        # print(f"\nRemoving {tensor} with indices {indices}.")
        # print(f"remaining term: {term}")

        # - split the indices that are in the remaining term according
        #   to their space
        used_indices = {}
        for s in set(s for s, _ in term._idx_counter):
            if (ov := s.space) not in used_indices:
                used_indices[ov] = set()
            used_indices[ov].add(s.name)

        # - check if the tensor is holding target indices.
        #   have to introduce a KroneckerDelta for each target index to avoid
        #   loosing indices in the term and replace the target indices on the
        #   tensor by new, unused indices:
        #   f_bc * Y^ac_ij -> delta_ik * delta_jl * delta_ad * f_bc * Y^dc_kl

        # get all target indices on the tensor, split according to their space
        tensor_target_indices = {}
        for s in indices:
            ov = s.space
            if s.name in target_indices.get(ov, []):
                if ov not in tensor_target_indices:
                    tensor_target_indices[ov] = []
                if s not in tensor_target_indices[ov]:
                    tensor_target_indices[ov].append(s)

        # - add the tensor indices to the term_indices to collect all
        #   unavailable indices
        for s in indices:
            if (ov := s.space) not in used_indices:
                used_indices[ov] = set()
            used_indices[ov].add(s.name)

        if tensor_target_indices:
            # print("Found target indices on tensor to remove:",
            #       tensor_target_indices)
            for space, idx_list in tensor_target_indices.items():
                if space not in used_indices:
                    used_indices[space] = set()
                additional_indices = get_lowest_avail_indices(
                    len(idx_list), used_indices[space], space
                )
                # add the new indices to the unavailable indices
                used_indices[space].update(additional_indices)
                # transform them from string to Dummies
                additional_indices = get_symbols(additional_indices)
                sub = {s: new_s for s, new_s in
                       zip(idx_list, additional_indices)}
                # create a delta for each index and attach to the term
                # and replace the index in tensor indices
                for s, new_s in sub.items():
                    term *= KroneckerDelta(s, new_s)
                indices = [sub.get(s, s) for s in indices]
            # print(f"modified tensor indices to: {indices}")
            # print(f"Term now reads {term}")

        # - check for repeating indices:
        #   introduce a delta in the term for each repeating index
        #   e.g. d_iiij -> d_iklj // term <- delta_ik * delta_il
        #   Problem: this might introduce unstable deltas...
        repeating_indices = {}
        for s, n in Counter(indices).items():
            if n > 1:
                if (ov := s.space) not in repeating_indices:
                    repeating_indices[ov] = []
                repeating_indices[ov].extend(s for _ in range(n-1))
        if repeating_indices:
            # print(f"Found repeating indices {repeating_indices}")
            #   - get the list indices of all tensor indices
            indices_i: dict[Index, list[int]] = {}
            for i, s in enumerate(indices):
                if s not in indices_i:
                    indices_i[s] = []
                indices_i[s].append(i)
            # - iterate through the repeating indices and generate a new
            #   index for each repeating index. Use the repeating and the
            #   new index to create a KroneckerDelta. On AntiSymmetricTensors
            #   indices can at most twice, once in upper and once in lower.
            #   On NonSymmetricTensors no such limit exists -> implement for
            #   an arbitrary amount of repetitions
            for space, idx_list in repeating_indices.items():
                additional_indices = get_lowest_avail_indices(
                    len(idx_list), used_indices.get(space, []), space
                )
                additional_indices = get_symbols(additional_indices)
                for s, new_s in zip(idx_list, additional_indices):
                    term *= KroneckerDelta(s, new_s)
                    # substitute the second occurence of s in tensor indices
                    indices[indices_i[s].pop(1)] = new_s
            # no repeating indices left
            assert max(Counter(indices).values()) == 1
            # print(f"Replaced repeating indices by new indices: {indices}")
            # print(f"The term now reads: {term}")
        # - minimize the tensor indices by permuting contracted indices.
        #   Ensure indices occur in ascending order: kijab -> ijkab.
        #   target indices are excluded from this procedure:
        #   with target indices i, a: kijab -> jikab
        # print(f"Minimized {indices} to ", end='', flush=True)
        indices, perms = minimize_tensor_indices(indices, target_indices)
        # print(indices)
        # - apply the index permuations for minimizig the indices
        #   also to the term
        term: e.Expr = term.permute(*perms)
        assert term.sympy is not S.Zero
        # - build a new tensor that holds the minimized indices
        #   further minimization might be possible taking the tensor
        #   symmetry into account, because we did not touch target indices:
        #   jikab -> d^jik_ab = - d^ijk_ab
        raw_tensor = tensor.sympy
        if isinstance(raw_tensor, AntiSymmetricTensor):
            bra_ket_sym = raw_tensor.bra_ket_sym
            if isinstance(raw_tensor, Amplitude):  # indices = lower, upper
                n_l = len(raw_tensor.lower)
                upper, lower = indices[n_l:], indices[:n_l]
            else:  # symtensor / antisymtensor, indices = upper, lower
                n_u = len(raw_tensor.upper)
                upper, lower = indices[:n_u], indices[n_u:]
            tensor = e.Expr(raw_tensor.__class__(
                raw_tensor.name, upper, lower, bra_ket_sym
            )).terms[0]
        elif isinstance(raw_tensor, NonSymmetricTensor):
            bra_ket_sym = None
            tensor = e.Expr(
                NonSymmetricTensor(raw_tensor.name, indices)
            ).terms[0]
        else:
            raise TypeError(f"Unknown tensor type {type(tensor.sympy)}")
        # print(f"The tensor now reads: {tensor}")
        # if we got a -1 -> move to the term
        term *= tensor.prefactor
        # print(f"Term now reads: {term}")
        # PREFACTOR:
        # - For a contraction d^ij_ab we obtain an additional prefactor of 1/4
        #   in the term, for d^ij_ka it is 1/2, or 1/4 for d^ij_kl
        #   -> it depends on the symmetry of the tensor we want to remove
        #      factor = n_perms + 1
        #   -> need to remove it from the term: multiply by the term by the
        #      inverse factor (4 for d^ij_ab)
        # - Additionally we need to ensure that the resulting expression
        #   preserves symmetry that was included in the input expression
        #   through the tensor we want to remove
        #   -> apply the tensor symmetry to the term
        #      d^ij_ab * X -> 1/4 (X - P_ij X - P_ab X + P_ij P_ab X)
        #   -> this leads to another factor of 1/(n_perms + 1)
        # - For usual tensors both factors cancel each other exactly:
        #   (n_perms + 1) / (n_perms + 1) = 1
        #   -> don't change the prefactor and just symmetrize the term
        # - If the tensor has additionaly bra ket symmetry:
        #    swapping bra and ket will either result in an identical
        #    tensor block (diagonal block)
        #    or will give a non canonical block which is folded into
        #    the canonical block we are treating currently
        #   - diagonal block: multiply the term by 1/2 to keep the result
        #     normalized... we will get twice as many terms from applying
        #     the tensor symmetry as without bra ket symmetry
        #   - non-diagonal block: bra ket swap gives a non canonical block
        #     which can be folded into the canonical block:
        #       f_ia + f_ai = 2 f_ia
        #     However we only want to treat canonical tensor blocks.
        #     Therefore, we need to "remove" the contributions from the
        #     non-canonical blocks by multiplying with 1/2
        #  -> if we have bra ket symmetry introduce a factor 1/2
        if bra_ket_sym is not None and bra_ket_sym is not S.Zero:
            term *= Rational(1, 2)
        # - For ADC amplitudes we only have to multiply the term by
        #   sqrt(n_perms + 1), because the other part of the factor
        #   is hidden inside the amplitude vector to keep the vector
        #   norm constant when lifting index restrictions
        #   -> we obtain an overall factor of
        #      sqrt(n_perms + 1) / (n_perms + 1) = 1 / sqrt(n_perms + 1)
        tensor_sym = tensor.symmetry()
        if t_name in ['X', 'Y']:  # are we removing an ADC amplitude?
            if bra_ket_sym is not S.Zero:
                raise ValueError("ADC amplitude vectors should have "
                                 "no bra ket symmetry.")
            term *= 1 / sqrt(len(tensor_sym) + 1)
        # print(f"Before symmetrization the terms reads {term}")
        # - add the tensor indices to the target indices of the term
        #   but only if it is not possible to determine them with the einstein
        #   sum convention -> only if target indices have been set manually
        if term.provided_target_idx is not None:
            term.set_target_idx(term.provided_target_idx + indices)
        # - apply the symmetry of the removed tensor to the term
        symmetrized_term = term.copy()
        for perms, sym_factor in tensor_sym.items():
            symmetrized_term += term.copy().permute(*perms) * sym_factor
        # - reduce the number of terms as much as possible
        return simplify(symmetrized_term)

    def process_term(term: e.Term, t_name):
        # print(f"\nProcessing term {term}")
        # collect all occurences of the desired tensor
        tensors = []
        remaining_term = e.Expr(1, **term.assumptions)
        for obj in term.objects:
            if obj.name == t_name:
                tensors.extend(obj for _ in range(obj.exponent))
            else:
                remaining_term *= obj
        if not tensors:  # could not find the tensor
            return {('none',): term}
        # extract all the target indices and split according to their space
        target_indices = {}
        for s in term.target:
            if (ov := s.space) not in target_indices:
                target_indices[ov] = set()
            target_indices[ov].add(s.name)
        # remove the first occurence of the tensor
        # and add all the remaining occurences back to the term
        for remaining_t in tensors[1:]:
            remaining_term *= remaining_t
        # the tensor might have an exponent that we need to take care of!
        tensor = tensors[0]
        base, exponent = tensor.base_and_exponent
        if exponent > 1:  # lower exponent by 1
            tensor = e.Expr(base, **tensor.assumptions).terms[0].objects[0]
            remaining_term *= Pow(base, exponent - 1)
        elif exponent < 1:
            raise NotImplementedError("Did not implement the case of removing "
                                      f"tensors with exponents < 1: {t_name} "
                                      f"in {term}")
        assert len(remaining_term) == 1
        remaining_term = remove(remaining_term.terms[0], tensor,
                                target_indices)
        # determine the space/block of the removed tensor
        # used as key in the returned dict
        t_block = [tensor.space]
        # print(t_block, remaining_term)
        if len(tensors) == 1:  # only a single occurence no need to recurse
            return {tuple(t_block): remaining_term}
        else:  # more than one occurence of the tensor
            # iterate through the terms that already have the first occurence
            # removed and recurse for each term
            ret = {}
            for t in remaining_term.terms:
                # hier gibts ein dict mit block, expr
                # bereits entfernter block + zusätzlich entfernte block
                contribution = process_term(t, t_name)
                for blocks, contrib in contribution.items():
                    key = tuple(sorted(t_block + list(blocks)))
                    if key not in ret:
                        ret[key] = 0
                    ret[key] += contrib
            return ret

    if not isinstance(expr, e.Expr):
        raise Inputerror(f"The expression needs to be provided as {e.Expr} "
                         "object.")
    if not isinstance(t_name, str):
        raise Inputerror("Tensor name needs to be provided as string.")

    ret = {}  # expr sorted by tensor block
    for term in expr.terms:
        for key, contrib in process_term(term, t_name).items():
            if key not in ret:
                ret[key] = 0
            ret[key] += contrib
    return ret
