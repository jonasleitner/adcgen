from indices import index_space, get_first_missing_index, indices
from misc import Inputerror
import expr_container as e
from sympy import Add, S
import time
import sort_expr as sort


def filter_tensor(expr, t_strings, strict='low', ignore_amplitudes=True):
    """Filter an expression and only return terms that contain certain tensors.
       The names of the desired tensors need to be provided as list/tuple of
       strings.
       This filtering is implemented in 3 different ways, controlled by strict:
        - 'high': return all terms that ONLY contain the desired tensors in the
                  requested amount. E.g.: ['V', 'V'] returns only
                  terms that contain not other tensors than 'V*V'
                  Setting ignore_amplitudes, ignores all not requested
                  amplitudes for this.
        - 'medium': return all terms that contain the desired tensors in the
                    requested amount, but other tensors may additionally be
                    present in the term. E.g. ['V', 'V'] also returns terms
                    that contain 'V*V*x', where x may be any amount of
                    arbitrary other tensors.
        - 'low': return all terms that contain all of the requested tensors.
                 E.g. ['V', 'V'] return all terms that contain 'V' at least
                 once.
       """
    from collections import Counter

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
        # True if ony the requested Tensors are in the term in the correct
        # amount
        elif strict == 'high':
            if ignore_amplitudes:
                requested_amplitudes = [
                    name for name in t_strings if name[0] in ['X', 'Y'] or
                    (name[0] == 't' and name[1:].isnumeric())
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
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)

    filtered = Add(*[term.sympy for term in expr.terms if check_term(term)])
    return e.expr(filtered, **expr.assumptions)


def find_compatible_terms(terms):
    from collections import Counter

    if not all(isinstance(term, e.term) for term in terms):
        raise Inputerror("Expected terms as a list of term Containers.")

    # extract the pattern of all terms
    pattern = [term.pattern for term in terms]
    # extract the target indices
    target = [term.target for term in terms]
    # extract tensors, deltas and the objects which hold the target indices
    # NOTE: tensors and deltas do not cover deltas and tensors in polynoms
    #       However, this should not be a problem, since a term
    #       (a+b) / (c+d) * (e+f) may be split in the three brakets which
    #       can be treated correctly.
    # additionally, one may compare the number of polynoms in each term
    # as an additional filter.
    tensors = []
    deltas = []
    target_obj = []
    for i, term in enumerate(terms):
        # extract tensors
        tensors.append([o.name for o in term.tensors
                        for i in range(o.exponent)])
        # extract deltas
        deltas.append([o.space for o in term.deltas
                       for i in range(o.exponent)])
        # extract objects that hold the target indices and also how many occ
        # virt target indices the object holds, e.g. f_cc Y_ij^ac
        # Y holds 2o, 1v
        temp = []
        for o in term.target_idx_objects:
            idx = o.idx
            found = {'o': 0, 'v': 0}
            for s in target[i]:
                if s in idx:
                    found[index_space(s.name)[0]] += 1
            temp.extend([(o.description(), found['o'], found['v'])
                        for i in range(o.exponent)])
        target_obj.append(sorted(temp))
        del temp

    # collect terms that are equal according to their pattern
    equal_terms = {}
    matched = set()
    for n, pat in enumerate(pattern):
        matched.add(n)
        for other_n, other_pat in enumerate(pattern):
            if other_n in matched:
                continue
            # check if terms are compatible: same length (up to a prefactor),
            # same amount of o/v indices, same tensors, same deltas, and that
            # the target indices occur at the same tensors (e.g. 1o/1v at Y)
            if abs(len(terms[n]) - len(terms[other_n])) > 1 or \
                    len(pat['o']) != len(other_pat['o']) or \
                    len(pat['v']) != len(other_pat['v']) or \
                    Counter(tensors[n]) != Counter(tensors[other_n]) or \
                    Counter(deltas[n]) != Counter(deltas[other_n]) or \
                    Counter(target_obj[n]) != Counter(target_obj[other_n]):
                continue

            # compare the target indices, though the function should also work
            # if they differ.
            # If target indices are different, it should not be possible to
            # map the terms on each other -> continue would be more appropriate
            if target[n] != target[other_n]:
                raise RuntimeError(f"Target indices of the terms {terms[n]} "
                                   f"and {terms[other_n]} are not identical:"
                                   f" {target[n]}, {target[other_n]}.")
            # try to map each index in other_n to an index in n
            match = True
            sub = {}  # {old: new}
            for ov, idx_pat in pat.items():
                # compare the pattern of the indices in both terms and create a
                # idx map. If a match for all indices is found -> the terms
                # may be simplified by renaming indices (or swapping bra/kets)
                idx_map = {}
                matched_idx = []
                for idx, p in idx_pat.items():
                    # check whether the current index is a target index
                    # -> only map onto other target indices
                    is_target = [idx in target[n], False]
                    count = Counter(p)
                    for other_idx, other_p in other_pat[ov].items():
                        # avoid double counting
                        if other_idx in matched_idx:
                            continue
                        is_target[1] = other_idx in target[other_n]
                        # check if either both or none are target indices
                        if is_target[0] != is_target[1]:
                            continue
                        if count == Counter(other_p):
                            # target indices need to be equal already -> can't
                            # substitute them
                            if all(is_target) and idx != other_idx:
                                continue
                            matched_idx.append(other_idx)
                            idx_map[other_idx] = idx
                            break
                # found a match for all indices
                if len(idx_map) == len(other_pat[ov]):
                    # filter target indices and other indices that already
                    # have the same name
                    sub.update({old: new for old, new in idx_map.items()
                                if old != new})
                else:
                    match = False
                    break
            # two terms are equal if all o/v indices have been matched. But
            # its not necessary that there is something to do, because the
            # sub dict is empty:
            # - indices may have the correct name and only bra/ket needs to
            #   be swapt to simplify -> make_real will handle
            # - the terms only differ by target indices that are not allowed
            #   to change within this function
            if match:
                matched.add(other_n)
                if sub:
                    if n not in equal_terms:
                        equal_terms[n] = {}
                    equal_terms[n][other_n] = sub
    return equal_terms


def simplify(expr, real=False, *sym_tensors):
    """Simplify an expression by renaming indices. The new index names are
       determined by establishing a mapping between the indices in different
       terms. If all indices in two terms share the same pattern (essentially
       have the same occurences), but have different names. The function will
       rename the indices in one of the two terms.
       If real is set, all 'c' are removed in the tensor names in order to make
       the amplitudes real. Additionally, make_real is called that tries to
       further simplify the expression by swapping bra and ket of symmetric
       tensors. By default the Fock matrix 'f' and the ERI 'V' are added to
       the provided symmetric tensors.
       """

    start = time.time()
    if not all(isinstance(t, str) for t in sym_tensors):
        raise Inputerror("Symmetric tensor names need to be provided as str.")
    sym_tensors = set(sym_tensors)
    if real:
        sym_tensors.update(['f', 'V'])

    expr = expr.expand()
    # adjust symmetric tensors of the container
    if isinstance(expr, e.expr):
        expr.set_sym_tensors(sym_tensors)
    # or put the expr in a container
    else:
        expr = e.expr(expr, real, sym_tensors)
    if real and not expr.real:
        expr = expr.make_real()

    # create terms and hand try to find comaptible terms that may be
    # simplified by substituting indices
    terms = expr.terms
    equal_terms = find_compatible_terms(terms)

    # substitue the indices in other_n and keep n as is
    res = e.compatible_int(0)
    matched = set()
    for n, sub_dict in equal_terms.items():
        matched.add(n)
        res += terms[n]
        for other_n, sub in sub_dict.items():
            matched.add(other_n)
            res += terms[other_n].subs(sub, simultaneous=True)
    # Add the unmatched remainder
    res += e.expr(Add(*[terms[n].sympy for n in range(len(terms))
                  if n not in matched]), **expr.assumptions)
    del terms  # not valid anymore (expr changed)
    print(f"simplify took {time.time()- start} seconds")
    if real:
        res = make_real(res, *sym_tensors)
    return res


def make_real(expr, *sym_tensors):
    """Makes an expression real by:
       1) removing any 'c' in the tensor names, which makes the t-amplitudes
          real
       2) try to collect terms by swapping bra and ket of all symmetric
          tensors. By default only the Fock matric 'f' and the ERI 'V' are
          assumed to be real.
       """
    from collections import Counter
    from itertools import combinations, chain

    def prescan(filtered_terms, t_strings):
        # prescan to identify terms that may be simplified by swapping
        # bra/ket. Find those terms by comparing the canonical indices
        # of the symmetric tensors.
        desired = Counter(t_strings)
        t_strings = set(t_strings)
        indices = []
        for term in filtered_terms.terms:
            t_idx = {t: [] for t in t_strings}
            for t in term.tensors:
                name = t.name
                if name in t_strings:
                    idx = t.idx
                    t_idx[name].extend([idx for _ in range(t.exponent)])
            if not all(len(idx) == desired[t] for t, idx in t_idx.items()):
                raise RuntimeError("Did not find the correct number of "
                                   f"tensors {dict(desired)} in {term}.")
            indices.append(t_idx)

        ret = []  # nested list [[0, 2, 3], [1, 5], ]
        matched = set()
        for i, t_idx in enumerate(indices):
            if i in matched:
                continue
            temp = [i]
            for j in range(i + 1, len(indices)):
                if j in matched:
                    continue
                match = True
                for t, idx_list in t_idx.items():
                    if Counter(idx_list) != Counter(indices[j][t]):
                        match = False
                        break
                if match:
                    temp.append(j)
            matched.update(temp)
            # only return if a match was found
            if len(temp) > 1:
                ret.append(temp)
        return ret

    def swap_tensors(tensor_terms, t_strings):
        # swap bra and ket of all possible combinations of the requested
        # tensors, e.g. for f,f,V -> f1 / f2 / V / f1,f2 / f1,V / f2,V /
        #                            f1,f2,V
        # This is done for every term. If a swap leads to a simplified
        # expression, the function starts over again with the new
        # expression.
        def swap_brakets(term: e.term, t_n_pairs):
            # swaps all desired tensors simultaneously
            from sympy.physics.secondquant import AntiSymmetricTensor, Pow
            swapped = e.expr(1, **term.assumptions)
            occurences = {t: 1 for t, _ in t_n_pairs}
            for o in term.objects:
                name = o.name
                if o.type != 'tensor' or name not in occurences:
                    swapped *= o
                    continue
                matches = [
                    (name, n) for n in
                    range(occurences[t], occurences[t]+o.exponent)
                    if (name, n) in t_n_pairs
                ]
                occurences[name] += o.exponent
                if not matches:
                    swapped *= o
                    continue
                n_swaps = len(matches)
                base = o.extract_pow
                swapped_tensor = AntiSymmetricTensor(name, base.lower,
                                                     base.upper)
                swapped *= (Pow(swapped_tensor, n_swaps) *
                            Pow(base, o.exponent - n_swaps))
            return swapped

        t_n_pairs = []
        for t, count in Counter(t_strings).items():
            t_n_pairs.extend([(t, n) for n in range(1, count+1)])
        # [[((t, n), )]]
        # first list: [1TensorSwapped, 2TensorSwapped]
        # secopnd list: all combinations where 1/2/3... tensors are swapped
        # first tuple: (t, n), (t', n'), ...  pairs that form a combination
        combs = [
            list(combinations(t_n_pairs, r))
            for r in range(1, len(t_n_pairs) + 1)
        ]
        for term in tensor_terms.terms:
            for n_swaps in combs:
                for comb in n_swaps:
                    swapped = swap_brakets(term, comb)
                    # keep the term as short as possible
                    if len(swapped) <= len(term):
                        new = tensor_terms - term + swapped
                        if new.sympy.is_number:
                            return new
                        elif len(new) < len(tensor_terms):
                            return swap_tensors(new, t_strings)
        return tensor_terms

    # start = time.time()
    if not all(isinstance(t, str) for t in sym_tensors):
        raise Inputerror("Symmetric tensors need to be provided as strings.")
    sym_tensors = set(sym_tensors)
    sym_tensors.update(['f', 'V'])

    # import and set up the expression
    expr = expr.expand()
    if isinstance(expr, e.expr):
        expr.set_sym_tensors(sym_tensors)
    else:
        expr = e.expr(expr, True, sym_tensors)
    if not expr.real:
        expr = expr.make_real()

    # determine the maximum amount each symmetric tensor occurs in a term.
    max_occurence = {t: 0 for t in sym_tensors}
    for term in expr.terms:
        available = Counter(
            [t.name for t in term.tensors for _ in range(t.exponent)
             if t.symmetric]
        )
        max_occurence.update(
            {t: n for t, n in available.items() if n > max_occurence[t]}
        )
    sym_tensors = [t for t in sym_tensors for _ in range(max_occurence[t])]

    # split the expression in subsets that all contain the same symmetric
    # tensors in the same amount.
    combs = [
        set(combinations(sym_tensors, r)) for r in range(1, len(sym_tensors)+1)
    ]
    filtered = {}
    for n_tensors in combs:
        for comb in n_tensors:
            comb_terms = filter_tensor(
                expr, comb, strict='high', ignore_amplitudes=True
            )
            if comb_terms.sympy is not S.Zero:
                filtered[comb] = comb_terms

    # Add all terms to the result that do not contain any symmetric tensor
    res = expr - Add(*[sub_expr.sympy for sub_expr in filtered.values()])

    # now try to simplify by swapping bra/ket of the symmetric tensors
    for t_strings, t_terms in filtered.items():
        if len(t_terms) > 1:
            matching_terms = prescan(t_terms, t_strings)
            temp = e.compatible_int(0)
            for match in matching_terms:
                to_swap = e.expr(Add(*[t_terms.terms[i].sympy for i in match]),
                                 True, expr.sym_tensors,
                                 expr.provided_target_idx)
                temp += swap_tensors(to_swap, t_strings)
            temp += Add(*[t_terms.terms[i].sympy for i in range(len(t_terms))
                        if i not in set(chain.from_iterable(matching_terms))])
            res += temp
        # if there is only a single term -> no need to prescan. Just try to
        # introduce Pow objects.
        else:
            res += swap_tensors(t_terms, t_strings)
    return res


def extract_dm(expr, symmetric=False):
    """Function that extracts the density matrix from the expectation
       value expression. Thereby, a dict is returned that contains the
       defintions of all canonical blocks of the density matrix."""
    from sympy.physics.secondquant import KroneckerDelta, AntiSymmetricTensor
    from sympy import Rational

    def minimize_d_indices(term, d_tensor):
        used_d_idx = {'occ': [], 'virt': []}
        for uplo, d_indices in d_tensor.items():
            for i, d_idx in enumerate(d_indices):
                sp = index_space(d_idx.name)
                new_idx = get_first_missing_index(used_d_idx[sp], sp)
                # is already the lowest idx -> nothing to do
                if d_idx.name == new_idx:
                    used_d_idx[sp].append(new_idx)
                    continue
                # found a lower index -> permute indices in the term
                new_idx = indices().get_indices(new_idx)[sp][0]
                term = term.permute((d_idx, new_idx))
                # and the d-tensor
                # new_idx is not necessarily present on d
                for other_uplo, other_d_indices in d_tensor.items():
                    if new_idx in other_d_indices:
                        other_i = other_d_indices.index(new_idx)
                        d_tensor[other_uplo][other_i] = d_idx
                d_tensor[uplo][i] = new_idx
                # print("permuted: ", d_idx, new_idx, d_tensor)
                return minimize_d_indices(term, d_tensor)
        return term, d_tensor

    def symmetrize_keep_pref(term, symmetry):
        symmetrized = term.copy()
        for perms, factor in symmetry.items():
            symmetrized += term.copy().permute(*perms) * factor
        return symmetrized

    def sort_canonical(idx):
        return (index_space(idx.name)[0],
                int(idx.name[1:]) if idx.name[1:] else 0,
                idx.name[0])

    # assume no polynoms are present in the term
    expr = expr.expand()
    # ensure that the expression is in a container
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)

    # sort the expression according to the blocks of the density matrix
    blocks = sort.by_tensor_block(expr, 'd', symmetric)

    for block, block_expr in blocks.items():
        removed = e.compatible_int(0)
        # - remove d in each term
        for term in block_expr.terms:
            new_term = e.compatible_int(1)
            d_tensor = None
            for t in term.objects:
                # found the d tensor
                if t.type == 'tensor' and t.name == 'd':
                    if d_tensor is not None:
                        raise RuntimeError("Found two d tensors in the term "
                                           f"{term}.")
                    upper = sorted(t.extract_pow.upper, key=sort_canonical)
                    lower = sorted(t.extract_pow.lower, key=sort_canonical)
                    d_tensor = {'upper': upper, 'lower': lower}
                    # adjust the sign of the term if necessary when the
                    # canonical block of the d-tensor is used
                    if t.sign_change:
                        new_term *= e.compatible_int(-1)
                # anything else than the d tensor
                else:
                    new_term *= t
            if d_tensor is None:
                raise RuntimeError("Could not find a d tensor in the term "
                                   f"{term}.")
            # if indices repeat on the d tensor -> introduce a delta of the
            # corresponding space. I think this should only be possible in
            # the occ space, because it should originate from a p+ q
            # contraction
            repeated_idx = [idx for idx in d_tensor['upper']
                            if idx in d_tensor['lower']]
            for idx in repeated_idx:
                # indices that are currently present in the term that belong
                # to the same space as the repeated index
                sp = index_space(idx.name)
                current_idx = [s.name for s in set(new_term.idx)
                               if index_space(s.name) == sp]
                current_idx.extend(
                    [s.name for s in d_tensor['upper'] + d_tensor['lower']
                     if index_space(s.name) == sp]
                )
                # generate a second index of the same space for the delta, e.g.
                # X_ij d^ki_kj -> X_ij delta_kl
                new_idx = indices().get_indices(
                    get_first_missing_index(set(current_idx), sp)
                )[sp][0]
                new_term *= KroneckerDelta(idx, new_idx)
                # also replace the idx in the d_tensor. Does not matter if in
                # upper or lower -> just arbitrarily use lower
                # the index can only appear once in lower. Otherwise the tensor
                # has to be 0 according to the antisymmetry.
                d_tensor['lower'][d_tensor['lower'].index(idx)] = new_idx
            # print("\nBLOCK: ", block)
            # print("ORIGINAL: ", term)
            # print("BEFORE SUB: ", new_term)

            # Now minimize the indices on the d_tensor - the target indices of
            # the term after the d tensor is removed.
            # This ensures that all terms that contribute to the ovov block,
            # for instance, have the same target indices iajb.
            # print("d-tensor before sub: ", d_tensor)
            new_term, d_tensor = minimize_d_indices(new_term, d_tensor)
            # print("AFTER SUB: ", new_term, d_tensor)
            # now symmetrize the term.
            # Here only Permutations P_ij / P_ab are taken into account,
            # because permutations P_ia will produce a term that contributes to
            # a different, non-canonical, block of the density matrix, e.g.,
            # P_ka d_ijka -> d_ijak, while P_ij d_ijab -> d_jiab is still part
            # of the canonical oovv block.
            # As a result, this function only returns terms that contribute
            # to one of the canonical blocks of the DM.
            # Therefore, when computing an expectation value, e.g., the
            # ovov block needs to be multiplied by a factor of 4, because it
            # also represents the voov, ovvo, vovo blocks, which are not
            # present in memory - due to the antisym of the density and the
            # operator matrix:
            #   D_ovov d_ovov = D_voov d_voov.
            # Symmetric DM's have additional symmetry on diagonal blocks,
            # e.g., D_ijkl = D_klij.
            # Therefore, terms that contribute to a diagonal block are
            # multiplied by 1/2 and the bra/ket permutation
            # is applied in addition to the anti-symmetry permutations.
            # Regarding the prefactor:
            #  - we have to multiply with the inverse of the prefactor that is
            #    introduced from the sum D_pqrs d_pqrs when computing the
            #    expectation value:
            #       mutliply by 4 for 2p-DM.
            #  - However, another prefactor of 1/4 has to be introduced to
            #    account for the anti-symmetry of the operator matrix:
            #       X d_ijab = 1/4 X (1 - P_ij)(1 - P_ab) d_jiab
            #     -> both prefactors cancel each other exactly. No need to
            #        adjust the prefactor
            #  - if a symmetric DM is computed for a symmetric operator matrix,
            #    terms of non-canonical blocks are added to canonical blocks,
            #    because in this case
            #       d_ovoo = d_ooov and D_ovoo = D_oovo.
            #    As a consequence if the pure ooov block is desired for a
            #    symmetric tensor
            #   -> need to multiply all tems by 1/2 if the bra/ket swap
            #      gives a non-canonical block. But only if bra/ket are of the
            #      same length. Otherwise the operator can not be symmetric
            #      anyway.
            diagonal_block = False
            if symmetric and len(d_tensor['upper']) == len(d_tensor['lower']):
                space_u = "".join(
                    [index_space(s.name)[0] for s in d_tensor['upper']]
                )
                space_l = "".join(
                    [index_space(s.name)[0] for s in d_tensor['lower']]
                )
                new_term *= Rational(1, 2)
                if space_u == space_l:
                    # print("Found diagonal block of symmetric DM -> Also "
                    #       "apply 0.5 * (1 + P_bra/ket)")
                    diagonal_block = True
                    permute_braket = [
                        (upper, d_tensor['lower'][i])
                        for i, upper in enumerate(d_tensor['upper'])
                    ]
                    # print("bra/ket permutations: ", permute_braket)
                # else:
                    # print("Found non-canonical block that is included in "
                    #       f"block {block} -> multiply terms with 1/2.")
            d_tensor = e.expr(
                AntiSymmetricTensor('d', d_tensor['upper'], d_tensor['lower'])
            )
            sym = d_tensor.terms[0].symmetry(only_contracted=False)
            # print("found symmetry: ", sym)
            symmetrized = symmetrize_keep_pref(new_term, sym)
            if diagonal_block:
                swapped_braket = new_term.copy().permute(*permute_braket)
                symmetrized += symmetrize_keep_pref(swapped_braket, sym)
            symmetrized = simplify(symmetrized, True, *symmetrized.sym_tensors)
            # print("WITH SYMMETRY: ", symmetrized)
            removed += symmetrized
        blocks[block] = removed
    return blocks
