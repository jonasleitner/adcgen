from misc import Inputerror
import expr_container as e
from sympy import Add, S
import time


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
        available = [o.name for o in term.tensors for i in range(o.exponent)]
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
                amplitudes = [t for t in ['t', 'X', 'Y'] if t not in
                              [name[0] for name in t_strings]]
                available = Counter([t for t in available
                                     if t[0] not in amplitudes])
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
    filtered = e.expr(filtered, expr.real, expr.sym_tensors)
    return filtered


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
    from collections import Counter

    start = time.time()
    if not all(isinstance(t, str) for t in sym_tensors):
        raise Inputerror("Symmetric tensors need to be provided as string.")
    sym_tensors = set(sym_tensors)
    if real:
        sym_tensors.update(['f', 'V'])

    # set up the expression correctly (tensor names are automatically adjusted
    # if real is set.)
    if isinstance(expr, e.expr):
        expr.set_sym_tensors(*sym_tensors)
    else:
        # adjust the sym_tensors in the expression
        expr = e.expr(expr, real, sym_tensors)
    if real and not expr.real:
        expr = expr.make_real

    start1 = time.time()
    terms = expr.terms
    print(f"initializing terms took {time.time() - start1} seconds")
    # extract the pattern of all terms
    start1 = time.time()
    pattern = [term.pattern(coupling=True) for term in terms]
    print(f"Pattern creation took {time.time() - start1} seconds")

    # collect terms that are equal according to their pattern
    start1 = time.time()
    equal_terms = {}
    matched = set()
    for n, p in enumerate(pattern):
        matched.add(n)
        target = terms[n].target
        for other_n, other_p in enumerate(pattern):
            if other_n in matched:
                continue
            # check if terms are compatible: same length (up to a prefactor)
            # and same amount of o/v indices
            if abs(len(terms[n]) - len(terms[other_n])) > 1 or \
                    len(p['o'].keys()) != len(other_p['o'].keys()) or \
                    len(p['v'].keys()) != len(other_p['v'].keys()):
                continue

            # compare the target indices, though the function should also work
            # if they differ.
            other_target = terms[other_n].target
            if target != other_target:
                raise RuntimeError(f"Target indices of the terms {terms[n]} "
                                   f"and {terms[other_n]} are not identical:"
                                   f" {target}, {other_target}.")
            # try to map each index in other_n to an index in n
            match = True
            sub = {}  # {old: new}
            for ov in p.keys():
                # compare the pattern of the indices in both terms create a
                # idx map. If a match for all indices is found -> the terms
                # may be simplified by renaming indices (or swapping bra/kets)
                idx_map = {}
                matched_idx = []
                for idx, pat in p[ov].items():
                    # target index -> do nothing (filtered later)
                    if idx in target:
                        idx_map[idx] = idx
                        continue
                    count = Counter(pat)
                    for other_idx, other_pat in other_p[ov].items():
                        # skip target and avoid double counting
                        if other_idx in other_target or \
                                other_idx in matched_idx:
                            continue
                        if count == Counter(other_pat):
                            matched_idx.append(other_idx)
                            idx_map[other_idx] = idx
                            break
                # found a match for all indices
                if len(idx_map.keys()) == len(other_p[ov].keys()):
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
    print(f"Comparing pattern took {time.time() - start1} seconds")

    start1 = time.time()
    # substitue the indices in other_n and keep n as is
    res = e.compatible_int(0)
    matched = set()
    for n, sub_dict in equal_terms.items():
        matched.add(n)
        res += terms[n]
        for other_n, sub in sub_dict.items():
            matched.add(other_n)
            res += terms[other_n].subs(sub, simultaneous=True)
    print(f"Substituting expression took {time.time() - start1} seconds")
    # Add the unmatched remainder
    start1 = time.time()
    res += e.expr(Add(*[terms[n].sympy for n in range(len(terms))
                  if n not in matched]), expr.real, expr.sym_tensors)
    print(f"Adding the unmatched remainder took {time.time() - start1} sec")
    terms.clear()
    print(f"new simplify took {time.time()- start} seconds")
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
                    t_idx[name].append(t.idx)
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
        # tensors, e.g. for f,f,V -> f1 / f2/ V / f1,f2 / f1,V / f2,V /
        #                            f1,f2,V
        # This is done for every term. If a swap leads to a simplified
        # expression, the function starts over again with the new
        # expression.

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
                    temp = term
                    for tensor in comb:
                        # swap the n'th occurence of the tensor in the term
                        # and check if the result leads to a simplification
                        swapped = temp.swap_braket(tensor[0], tensor[1])
                        new = tensor_terms - term + swapped
                        if len(new) < len(tensor_terms) and \
                                len(swapped) <= len(term):
                            return swap_tensors(new, t_strings)
                        temp = swapped
        return tensor_terms

    start = time.time()
    if not all(isinstance(t, str) for t in sym_tensors):
        raise Inputerror("Symmetric tensors need to be provided as strings.")
    sym_tensors = set(sym_tensors)
    sym_tensors.update(['f', 'V'])

    # import and set up the expression
    expr = expr.expand()
    if isinstance(expr, e.expr):
        expr.set_sym_tensors(*sym_tensors)
    else:
        expr = e.expr(expr, True, sym_tensors)
    if not expr.real:
        expr = expr.make_real

    # determine the maximum amount each symmetric tensor occurs in a term.
    max_occurence = {t: 0 for t in sym_tensors}
    for term in expr.terms:
        available = Counter(
            [t.name for t in term.tensors for i in range(t.exponent)
             if t.name in sym_tensors]
        )
        max_occurence.update(
            {t: n for t, n in available.items() if n > max_occurence[t]}
        )
    sym_tensors = [t for t in sym_tensors for i in range(max_occurence[t])]

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
    res = expr - Add(*[terms.sympy for terms in filtered.values()])

    # now try to simplify by swapping bra/ket of the symmetric tensors
    for t_strings, t_terms in filtered.items():
        if len(t_terms) > 1:
            matching_terms = prescan(t_terms, t_strings)
            temp = e.compatible_int(0)
            for match in matching_terms:
                to_swap = e.expr(Add(*[t_terms.terms[i].sympy for i in match]),
                                 True, expr.sym_tensors)
                temp += swap_tensors(to_swap, t_strings)
            temp += Add(*[t_terms.terms[i].sympy for i in range(len(t_terms))
                        if i not in set(chain.from_iterable(matching_terms))])
            res += temp
        # if there is only a single term -> no need to prescan. Just try to
        # introduce Pow objects.
        else:
            res += swap_tensors(t_terms, t_strings)
    print(f"new make_real took {time.time() - start} seconds.")
    return res
