from collections.abc import Sequence
from collections import Counter, defaultdict
import itertools

from sympy import Add, Expr, Rational, Pow, S, sqrt

from . import func
from .expression import ExprContainer, TermContainer, ObjectContainer
from .indices import (
    get_symbols, order_substitutions, Index, get_lowest_avail_indices,
    minimize_tensor_indices, _is_index_tuple
)
from .misc import Inputerror
from .sympy_objects import (
    KroneckerDelta, Amplitude, AntiSymmetricTensor, NonSymmetricTensor
)
from .tensor_names import is_adc_amplitude, is_t_amplitude


def filter_tensor(expr: ExprContainer, t_strings: Sequence[str],
                  strict: str = 'low',
                  ignore_amplitudes: bool = True) -> ExprContainer:
    """
    Filter an expression keeping only terms that contain the desired tensors.

    Parameters
    ----------
    t_strings : Sequence[str]
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

    def check_term(term: TermContainer) -> bool:
        available = []
        for obj in term.objects:
            name = obj.name
            if name is None:
                continue
            exp = obj.exponent
            assert exp.is_Integer
            available.extend(name for _ in range(int(exp)))
        # True if all requested tensors are in the term
        if strict == 'low':
            return all(t in available for t in set(t_strings))
        # True if all requested Tensors occur the correct amount of times
        elif strict == 'medium':
            available = Counter(available)
            desired = Counter(t_strings)
            return desired.items() <= available.items()
        # True if only the requested Tensors are in the term in the correct
        # amount
        elif strict == 'high':
            if ignore_amplitudes:
                requested_amplitudes = [
                    name for name in t_strings
                    if is_adc_amplitude(name) or is_t_amplitude(name)
                ]
                ignored_amplitudes = {
                    name for name in available if
                    (is_adc_amplitude(name) or is_t_amplitude(name))
                    and name not in requested_amplitudes
                }
                available = Counter([t for t in available
                                     if t not in ignored_amplitudes])
            else:
                available = Counter(available)
            desired = Counter(t_strings)
            return desired == available
        raise ValueError(f"invalid value for strict {strict}")

    if not all(isinstance(t, str) for t in t_strings):
        raise Inputerror("Tensor names need to be provided as str.")
    if strict not in ['low', 'medium', 'high']:
        raise Inputerror(f"{strict} is not a valid option for strict. Valid"
                         "options are 'low', 'medium' or 'high'.")
    assert isinstance(expr, ExprContainer)

    expr = expr.expand()
    filtered = Add(*(
        term.inner for term in expr.terms if check_term(term)
    ))
    return ExprContainer(filtered, **expr.assumptions)


def find_compatible_terms(terms: Sequence[TermContainer]
                          ) -> dict[int, dict[int, list[tuple[Index, Index]]]]:
    """
    Determines the substitutions of contracted needed to map terms onto each
    other.

    Parameters
    ----------
    terms: Sequence[Term]
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

    def compare_terms(
            pattern: dict[tuple[str, str], dict[Index, list[str]]],
            other_pattern: dict[tuple[str, str], dict[Index, list[str]]],
            target: tuple[Index, ...], term: TermContainer,
            other_term: TermContainer) -> None | list[tuple[Index, Index]]:
        # function to compare two terms that are compatible, i.e., have the
        # same amount of indices in each space, the same amount and type of
        # objects and the same target indices
        sub_list: list[dict[Index, Index]] = []
        for ov, idx_pattern in pattern.items():
            # only compare indices that belong to the same space
            other_idx_pattern = other_pattern.get(ov, None)
            # the other space is not available in the other term
            # -> they cant match
            if other_idx_pattern is None:
                return None
            # list to hold the substitution dictionaries of the current space
            ov_sub_list: list[dict[Index, Index]] = []

            for idx, pat in idx_pattern.items():
                # find all possible matches for the current idx
                # if its a target idx -> only allow mapping on other target idx
                is_target = idx in target
                # list to collect all possible matches
                matching_idx: list[Index] = []
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
                    new_ov_sub_list: list[dict[Index, Index]] = []
                    for sub, other_idx in \
                            itertools.product(ov_sub_list, matching_idx):
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
            ov_sub_list = [
                sub for sub in ov_sub_list
                if sub.keys() == other_idx_pattern.keys()
            ]

            if not ov_sub_list:  # did not find a single complete sub dict
                return None

            # initialize the final substitution dicts
            if not sub_list:
                sub_list.extend(ov_sub_list)
            else:  # combine the sub dicts. different spaces can not overlap
                sub_list = [
                    other_sp_sub | sub for other_sp_sub, sub in
                    itertools.product(sub_list, ov_sub_list)
                ]

        # test all sub dicts to identify the correct one (if one exists)
        for sub in sub_list:
            sub = order_substitutions(sub)
            sub_other_term = other_term.inner.subs(sub)
            assert isinstance(sub_other_term, Expr)
            # sub is not valid for other term: evaluates to 0 due to
            # some antisymmetry e.g. t_ijcd -> t_ijcc = 0
            if sub_other_term is S.Zero and other_term.inner is not S.Zero:
                continue
            # diff (or sum) is a single term (no Add obj)
            # can either sum up to 0 or to a single term with a different pref
            # -> check for type of result and not for result value
            if not isinstance(Add(term.inner, -sub_other_term), Add):
                return sub
        return None  # no valid sub dict -> return None

    def repeating_idx_sp(idx_list: list[tuple[str, set[Index], set[Index]]]):
        repeating_idx = []
        for idx1, idx2 in itertools.combinations(idx_list, 2):
            descr1, descr2 = idx1[0], idx2[0]
            for i1, i2 in itertools.product(idx1[1:], idx2[1:]):
                repeated = i1 & i2
                if len(repeated) > 1:
                    repeated = "".join(sorted(
                        s.space[0] + s.spin for s in repeated
                    ))
                    repeating_idx.append((repeated, *sorted([descr1, descr2])))
        return tuple(sorted(repeating_idx))

    if not all(isinstance(term, TermContainer) for term in terms):
        raise Inputerror("Expected terms as a list of term Containers.")

    # prefilter terms according to
    # - number of objects, excluding prefactor
    # - type, name, space, spin, obj target indices and exponent of objects
    # - the space of repeating indices subsets (2, 3, ...) that repeat on
    #   on multiple objects together in a common index subspace (upper/lower)
    # - number of indices in each space
    # - the target indices
    filtered_terms: defaultdict[tuple, list[int]] = defaultdict(list)
    term_pattern: list[dict[tuple[str, str], dict[Index, list[str]]]] = []
    term_target: list[tuple[Index, ...]] = []
    for term_i, term in enumerate(terms):
        # target indices
        target = term.target
        term_target.append(target)
        # pattern
        pattern = term.pattern()
        term_pattern.append(pattern)
        # obj name, space, exponent, obj_target_indices, repeating_indices
        descriptions: list[str] = []
        tensor_idx_list: list[tuple[str, set[Index], set[Index]]] = []
        length = 0
        for o in term.objects:
            base = o.base
            if (descr := o.description()) == 'prefactor':
                continue
            elif isinstance(base, AntiSymmetricTensor):
                upper, lower = base.upper, base.lower
                assert _is_index_tuple(upper) and _is_index_tuple(lower)
                tensor_idx_list.append(
                    (descr, set(upper), set(lower))
                )
            elif isinstance(base, (KroneckerDelta, NonSymmetricTensor)):
                tensor_idx_list.append((descr, set(o.idx), set()))
            length += 1
            descriptions.append(descr)
        pattern_key = tuple(sorted(
            (sp, len(idx_pat)) for sp, idx_pat in pattern.items()
        ))
        key = (length, tuple(sorted(descriptions)),
               repeating_idx_sp(tensor_idx_list), pattern_key, target)
        filtered_terms[key].append(term_i)

    compatible_terms: dict[int, dict[int, list[tuple[Index, Index]]]] = {}
    for term_idx_list in filtered_terms.values():
        # set to keep track of the already mapped terms
        matched: set[int] = set()
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

                sub = compare_terms(
                    pattern, term_pattern[other_term_i],
                    target, term, terms[other_term_i]
                )
                # was possible to map the terms onto each other!
                if sub is not None:
                    compatible_terms[term_i][other_term_i] = sub
                    matched.add(other_term_i)
    return compatible_terms


def simplify(expr: ExprContainer) -> ExprContainer:
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
    expr : ExprContainer
        The expression to simplify

    Returns
    -------
    ExprContainer
        The simplified expression.
    """
    assert isinstance(expr, ExprContainer)
    expr = expr.expand()
    if len(expr) == 1:  # trivial: only a single term
        return expr
    # create terms and try to find comaptible terms that may be
    # simplified by substituting indices
    terms = expr.terms
    equal_terms = find_compatible_terms(terms)
    # substitue the indices in other_n and keep n as is
    res = ExprContainer(0, **expr.assumptions)
    for n, matches in equal_terms.items():
        res += terms[n]
        for other_n, sub in matches.items():
            res += terms[other_n].subs(sub)
    return res


def simplify_unitary(expr: ExprContainer, t_name: str,
                     evaluate_deltas: bool = False) -> ExprContainer:
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

    def simplify_term_unitary(term: TermContainer) -> TermContainer:
        objects = term.objects
        # collect the indices of all unitary tensors in the term
        unitary_tensors: list[int] = []
        for i, obj in enumerate(objects):
            if obj.name == t_name:
                exp = obj.exponent
                assert exp.is_Integer
                unitary_tensors.extend(i for _ in range(int(exp)))

        # only implemented for 2 dimensional unitary tensors
        if any(len(objects[i].idx) != 2 for i in unitary_tensors):
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
        for (i1, i2) in itertools.combinations(unitary_tensors, 2):
            idx1 = objects[i1].idx
            idx2 = objects[i2].idx
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
            new_term = ExprContainer(delta, **term.assumptions)
            if i1 == i2:
                base, exponent = objects[i1].base_and_exponent
                assert exponent.is_Integer
                new_term *= Pow(base, int(exponent) - 2)
            else:
                b1, exponent1 = objects[i1].base_and_exponent
                b2, exponent2 = objects[i2].base_and_exponent
                assert exponent1.is_Integer and exponent2.is_Integer
                new_term *= Pow(b1, int(exponent1) - 1)
                new_term *= Pow(b2, int(exponent2) - 1)

            # add remaining objects
            for i, o in enumerate(objects):
                if i == i1 or i == i2:
                    continue
                else:
                    new_term *= o
            return simplify_term_unitary(new_term.terms[0])
        # could not find simplification -> return
        return term

    assert isinstance(expr, ExprContainer)

    res = ExprContainer(0, **expr.assumptions)
    for term in expr.terms:
        res += simplify_term_unitary(term)

    # evaluate the generated deltas if requested
    if evaluate_deltas:
        res = ExprContainer(func.evaluate_deltas(res.inner), **res.assumptions)
    return res


def remove_tensor(expr: ExprContainer, t_name: str
                  ) -> dict[tuple[str, ...], ExprContainer]:
    """
    Removes a tensor from each term of an expression by undoing the contraction
    of the remaining term with the tensor. The resulting expression is split
    according to the blocks of the removed tensor. Note that only canonical
    tensor blocks are considered, because the non-canonical blocks can be
    generated from the canonical ones, e.g., removing a symmetric matrix d_{pq}
    from an expression can only result in expressions for the 'oo', 'ov' and
    'vv' blocks, since d_{ai} = d_{ia}.
    The symmetry of the removed tensor is taken into account, such that the
    original expression can be restored if all block expressions are
    contracted with the corresponding tensor blocks again.
    Note that for ADC-Amplitudes a special prefactor is used.

    Parameters
    ----------
    expr : ExprContainer
        The expression where the tensor should be removed.
    t_name : str
        Name of the tensor that should be removed.

    Returns
    -------
    dict[tuple[str, ...], ExprContainer]
        key: Tuple of removed tensor blocks
        value: Part of the original expression that contained the corresponding
               blocks. If contracted with the tensor blocks again, a part of
               the original expression is recovered.
    """

    def remove(term: TermContainer, tensor: ObjectContainer,
               target_indices: dict[tuple[str, str], set[str]]
               ) -> ExprContainer:
        # - get the tensor indices
        indices: Sequence[Index] = list(tensor.idx)
        # - split the indices that are in the remaining term according
        #   to their space and spin to gather information about used indices
        used_indices: dict[tuple[str, str], set[str]] = {}
        for s in set(s for s, _ in term._idx_counter):
            if (idx_key := s.space_and_spin) not in used_indices:
                used_indices[idx_key] = set()
            used_indices[idx_key].add(s.name)
        # - check if the tensor is holding target indices.
        #   have to introduce a KroneckerDelta for each target index to avoid
        #   loosing indices in the term and replace the target indices on the
        #   tensor by new, unused indices:
        #   f_bc * Y^ac_ij -> delta_ik * delta_jl * delta_ad * f_bc * Y^dc_kl

        # get all target indices on the tensor, split according to their space
        # and spin
        tensor_target_indices: dict[tuple[str, str], list[Index]] = {}
        for s in indices:
            idx_key = s.space_and_spin
            if s.name in target_indices.get(idx_key, []):
                if idx_key not in tensor_target_indices:
                    tensor_target_indices[idx_key] = []
                if s not in tensor_target_indices[idx_key]:
                    tensor_target_indices[idx_key].append(s)
        # - add the tensor indices to the term_indices to collect all
        #   not available indices
        for s in indices:
            if (idx_key := s.space_and_spin) not in used_indices:
                used_indices[idx_key] = set()
            used_indices[idx_key].add(s.name)

        if tensor_target_indices:
            term_with_deltas = ExprContainer(term.inner, **term.assumptions)
            for idx_key, idx_list in tensor_target_indices.items():
                if idx_key not in used_indices:
                    used_indices[idx_key] = set()
                space, spin = idx_key
                additional_indices = get_lowest_avail_indices(
                    len(idx_list), used_indices[idx_key], space
                )
                # add the new indices to the unavailable indices
                used_indices[idx_key].update(additional_indices)
                # transform them from string to Dummies
                if spin:
                    spins = spin * len(idx_list)
                else:
                    spins = None
                additional_indices = get_symbols(additional_indices, spins)

                sub = {
                    s: new_s for s, new_s in zip(idx_list, additional_indices)
                }
                # create a delta for each index and attach to the term
                # and replace the index in tensor indices
                for s, new_s in sub.items():
                    term_with_deltas *= KroneckerDelta(s, new_s)
                indices = [sub.get(s, s) for s in indices]
            assert len(term_with_deltas) == 1
            term = term_with_deltas.terms[0]
            del term_with_deltas
        # - check for repeating indices:
        #   introduce a delta in the term for each repeating index
        #   e.g. d_iiij -> d_iklj // term <- delta_ik * delta_il
        #   Problem: this might introduce unstable deltas...
        repeating_indices = {}
        for s, n in Counter(indices).items():
            if n > 1:
                if (idx_key := s.space_and_spin) not in repeating_indices:
                    repeating_indices[idx_key] = []
                repeating_indices[idx_key].extend(s for _ in range(n-1))
        if repeating_indices:
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
            term_without_repeating = ExprContainer(
                term.inner, **term.assumptions
            )
            for idx_key, idx_list in repeating_indices.items():
                space, spin = idx_key
                additional_indices = get_lowest_avail_indices(
                    len(idx_list), used_indices.get(idx_key, []), space
                )
                if spin:
                    spins = spin * len(idx_list)
                else:
                    spins = None
                additional_indices = get_symbols(additional_indices, spins)
                for s, new_s in zip(idx_list, additional_indices):
                    term_without_repeating *= KroneckerDelta(s, new_s)
                    # substitute the second occurence of s in tensor indices
                    indices[indices_i[s].pop(1)] = new_s
            # no repeating indices left
            assert max(Counter(indices).values()) == 1
            assert len(term_without_repeating) == 1
            term = term_without_repeating.terms[0]
            del term_without_repeating
        # - minimize the tensor indices by permuting contracted indices.
        #   Ensure indices occur in ascending order: kijab -> ijkab.
        #   target indices are excluded from this procedure:
        #   with target indices i, a: kijab -> jikab
        indices, perms = minimize_tensor_indices(indices, target_indices)
        # - apply the index permuations for minimizig the indices
        #   also to the term
        res_term = term.permute(*perms)
        del term
        assert res_term.inner is not S.Zero
        # - build a new tensor that holds the minimized indices
        #   further minimization might be possible taking the tensor
        #   symmetry into account, because we did not touch target indices:
        #   jikab -> d^jik_ab = - d^ijk_ab
        raw_tensor = tensor.inner
        if isinstance(raw_tensor, AntiSymmetricTensor):
            bra_ket_sym = raw_tensor.bra_ket_sym
            if isinstance(raw_tensor, Amplitude):  # indices = lower, upper
                n_l = len(raw_tensor.lower)
                upper, lower = indices[n_l:], indices[:n_l]
            else:  # symtensor / antisymtensor, indices = upper, lower
                n_u = len(raw_tensor.upper)
                upper, lower = indices[:n_u], indices[n_u:]
            res_tensor = ExprContainer(raw_tensor.__class__(
                raw_tensor.name, upper, lower, bra_ket_sym
            )).terms[0]
        elif isinstance(raw_tensor, NonSymmetricTensor):
            bra_ket_sym = None
            res_tensor = ExprContainer(
                NonSymmetricTensor(raw_tensor.name, indices)
            ).terms[0]
        else:
            raise TypeError(f"Unknown tensor type {type(tensor.inner)}")
        del raw_tensor
        del tensor
        # if we got a -1 -> move to the term
        res_term *= res_tensor.prefactor
        assert isinstance(res_term, ExprContainer)
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
        #     -> the factor from lifting the index restrictions remains
        #        constant, while the factor for the symmetrisation is
        #        multiplied by 2:
        #        (n_perms + 1) / [2 (n_perms + 1)] = 1/2
        #   - non-diagonal block: bra ket swap gives a non canonical block
        #     which can be folded into the canonical block:
        #       f_ia + f_ai = 2 f_ia
        #     However we only want to treat canonical tensor blocks.
        #     Therefore, we need to "remove" the contributions from the
        #     non-canonical blocks by multiplying with 1/2
        #  -> if we have bra ket symmetry introduce a factor 1/2
        if bra_ket_sym is not None and bra_ket_sym is not S.Zero:
            res_term *= Rational(1, 2)
            assert isinstance(res_term, ExprContainer)
        # - For ADC amplitudes we only have to multiply the term by
        #   sqrt(n_perms + 1), because the other part of the factor
        #   is hidden inside the amplitude vector to keep the vector
        #   norm constant when lifting index restrictions
        #   -> we obtain an overall factor of
        #      sqrt(n_perms + 1) / (n_perms + 1) = 1 / sqrt(n_perms + 1)
        tensor_sym = res_tensor.symmetry()
        if is_adc_amplitude(t_name):  # are we removing an ADC amplitude?
            if bra_ket_sym is not S.Zero:
                raise ValueError("ADC amplitude vectors should have "
                                 "no bra ket symmetry.")
            res_term *= S.One / sqrt(len(tensor_sym) + 1)
            assert isinstance(res_term, ExprContainer)
        # - add the tensor indices to the target indices of the term
        #   but only if it is not possible to determine them with the einstein
        #   sum convention -> only if target indices have been set manually
        if res_term.provided_target_idx is not None:
            res_term.set_target_idx(res_term.provided_target_idx + indices)
        # - apply the symmetry of the removed tensor to the term
        symmetrized_term = res_term.copy()
        for perms, sym_factor in tensor_sym.items():
            symmetrized_term += res_term.copy().permute(*perms) * sym_factor
        # - reduce the number of terms as much as possible
        return simplify(symmetrized_term)

    def process_term(term: TermContainer, t_name: str
                     ) -> dict[tuple[str, ...], ExprContainer | TermContainer]:
        # print(f"\nProcessing term {term}")
        # collect all occurences of the desired tensor
        tensors: list[ObjectContainer] = []
        remaining_term = ExprContainer(1, **term.assumptions)
        for obj in term.objects:
            if obj.name == t_name:
                tensors.append(obj)  # we take care of the exponent later!
            else:
                remaining_term *= obj
        if not tensors:  # could not find the tensor
            return {("none",): term}
        # extract all the target indices and split according to their space
        target_indices: dict[tuple[str, str], set[str]] = {}
        for s in term.target:
            if (idx_key := s.space_and_spin) not in target_indices:
                target_indices[idx_key] = set()
            target_indices[idx_key].add(s.name)
        # remove the first occurence of the tensor
        # and add all the remaining occurences back to the term
        for remaining_t in tensors[1:]:
            remaining_term *= remaining_t
        # the tensor might have an exponent that we need to take care of!
        tensor = tensors[0]
        exponent = tensor.exponent
        # I am not 100% sure atm how to remove tensors with exponents != 1
        # so wait for an actual example to come up and implement it then.
        if exponent != 1:
            raise NotImplementedError("Did not implement the case of removing "
                                      f"tensors with exponents != 1: {t_name} "
                                      f"in {term}")
        assert len(remaining_term) == 1
        remaining_term = remove(
            remaining_term.terms[0], tensor, target_indices
        )
        # determine the space/block of the removed tensor
        # used as key in the returned dict
        spin = tensor.spin
        if all(c == "n" for c in spin):
            t_block = [tensor.space]
        else:
            t_block = [f"{tensor.space}_{spin}"]
        # print(t_block, remaining_term)
        if len(tensors) == 1:  # only a single occurence no need to recurse
            return {tuple(t_block): remaining_term}
        else:  # more than one occurence of the tensor
            # iterate through the terms that already have the first occurence
            # removed and recurse for each term
            ret = {}
            for t in remaining_term.terms:
                # add the blocks to the already removed block
                contribution = process_term(t, t_name)
                for blocks, contrib in contribution.items():
                    key = tuple(sorted(t_block + list(blocks)))
                    if key not in ret:
                        ret[key] = 0
                    ret[key] += contrib
            return ret

    assert isinstance(expr, ExprContainer)
    assert isinstance(t_name, str)
    # expr sorted by tensor block
    ret: dict[tuple[str, ...], ExprContainer] = {}
    for term in expr.terms:
        for key, contrib in process_term(term, t_name).items():
            if key not in ret:
                ret[key] = ExprContainer(0, **contrib.assumptions)
            ret[key] += contrib
    return ret
