from sympy import KroneckerDelta, Add, S, Mul, latex
from sympy.physics.secondquant import AntiSymmetricTensor, F, Fd
from indices import assign_index
from misc import Inputerror


def sort_by_n_deltas(expr):
    expr = expr.expand()
    if not isinstance(expr, Add):
        raise Inputerror("Can only filter an expression that is of "
                         f"type {Add}. Provided: {type(expr)}")
    deltas = {}
    for term in expr.args:
        delta_count = 0
        for t in term.args:
            if isinstance(t, KroneckerDelta):
                delta_count += 1
        try:
            deltas[delta_count].append(term)
        except KeyError:
            deltas[delta_count] = []
            deltas[delta_count].append(term)
    res = {}
    for n_d, terms in deltas.items():
        res[n_d] = Add(*[t for t in terms])
    return res


def sort_by_type_deltas(expr):
    expr = expr.expand()
    if not isinstance(expr, Add):
        raise Inputerror("Can only filter an expression that is of "
                         f"type {Add}. Provided: {type(expr)}")
    deltas = {}
    for term in expr.args:
        temp = []
        for t in term.args:
            if isinstance(t, KroneckerDelta):
                ov1 = assign_index(t.preferred_index.name)
                ov2 = assign_index(t.killable_index.name)
                temp.append("".join(sorted(ov1[0] + ov2[0])))
        temp = tuple(temp)
        if not temp:
            temp = "no_deltas"
        try:
            deltas[temp].append(term)
        except KeyError:
            deltas[temp] = []
            deltas[temp].append(term)
    res = {}
    for d, terms in deltas.items():
        res[d] = Add(*[t for t in terms])
    return res


def sort_by_type_tensor(expr, t_string):
    """Sorts an expression according the blocks of a Tensor, i.e.
       the upper and lower indices are assigned to occ/virt.
       The terms are collected in a dict with e.g. 'ov' or 'ooov'
       as keys."""

    expr = expr.expand()
    if not isinstance(expr, Add):
        raise Inputerror(f"Can only sort expressions that are of type {Add}."
                         f" Provided expression is of type {type(expr)}.")
    tensor = {}
    for term in expr.args:
        temp = []
        for t in term.args:
            if isinstance(t, AntiSymmetricTensor) and \
                    t.symbol.name == t_string:
                ov_up = [assign_index(s.name)[0] for s in t.upper]
                ov_lo = [assign_index(s.name)[0] for s in t.lower]
                temp.append("".join(ov_up + ov_lo))
        temp = tuple(sorted(temp))
        if not temp:
            temp = f"no_{t_string}"
        try:
            tensor[temp].append(term)
        except KeyError:
            tensor[temp] = []
            tensor[temp].append(term)
    res = {}
    for ten, term in tensor.items():
        res[ten] = Add(*[t for t in term])
    return res


def filter_tensor(expr, *t_string):
    """Returns terms of an expression that contain all of the requested
       tensors. The function also filters for the complex conjugate
       tensors, where it is assumed that the names of T and T* differ only
       by (one or more) 'c'.
       """

    for t in t_string:
        if "c" in t:
            raise Inputerror("Can not search for a tensor with 'c' in it's "
                             "name. The function filters it to "
                             "also return the complex conjugate of the desired"
                             f" tensor. Provided tensor strings: {t_string}.")

    def check_term(term, *t_string):
        found = []
        if isinstance(term, Mul):
            for t in term.args:
                if isinstance(t, AntiSymmetricTensor) and \
                        t.symbol.name.replace("c", "") in t_string:
                    found.append(t.symbol.name.replace("c", ""))
        else:
            if isinstance(term, AntiSymmetricTensor) and \
                    term.symbol.name.replace("c", "") in t_string:
                found.append(term.symbol.name.replace("c", ""))
        if all(t in found for t in t_string):
            return term
        return S.Zero

    expr = expr.expand()
    tensor = 0
    if isinstance(expr, Add):
        for term in expr.args:
            tensor += check_term(term, *t_string)
    else:
        tensor += check_term(expr, *t_string)
    return tensor


def sort_tensor_sum_indices(expr, t_string):
    """Sorts an expression by sorting the terms depending on the
       number and type (ovv/virt) of indices of an AntiSymmetricTensor
       that are contracted. The function assumes that indices are summed
       if they appear more than once in the term.
       """

    expr = expr.expand()
    if not isinstance(expr, Add):
        raise Inputerror("Can only filter an expression that is of "
                         f"type {Add}. Provided: {type(expr)}")
    # prefilter all terms that contain the desired tensor
    tensor = filter_tensor(expr, t_string)
    if len(tensor.args) < 2:
        print("The expression contains at most 1 term that contains "
              f"the tensor {t_string}. No need to filter according "
              "to indices.")
        return tensor
    ret = {}
    # get all terms that do not contain the tensor
    if expr - tensor != S.Zero:
        ret[f'no_{t_string}'] = expr - tensor

    temp = {}
    for term in tensor.args:
        # first get all the indices that are present in the Mul object
        # and count how often they occur
        idx = {}
        for t in term.args:
            for s in t.free_symbols:
                if s in idx:
                    idx[s] += 1
                else:
                    idx[s] = 0
        # now check how many indices in upper and lower occur more
        # than once in the expr (einstein sum convention).
        found = False
        for t in term.args:
            if isinstance(t, AntiSymmetricTensor) and \
                    t.symbol.name.replace('c', '') == t_string:
                if found:
                    print(f"Found more than occurence of the tensor {t_string}"
                          f" in {latex(term)}. The function assumes there "
                          "exists only 1.")
                    exit()
                upper_sum = [
                    assign_index(s.name)[0] for s in t.upper if idx[s]
                ]
                lower_sum = [
                    assign_index(s.name)[0] for s in t.lower if idx[s]
                ]
                sum_ov = "".join(sorted(lower_sum + upper_sum))
                if not sum_ov:
                    sum_ov = "no_indices"
                if sum_ov not in temp:
                    temp[sum_ov] = []
                temp[sum_ov].append(term)

    for ov, terms in temp.items():
        ret[ov] = Add(*[t for t in terms])
    return ret


def change_tensor_name(expr, old, new):
    """Changes the Name of a AntiSymmetricTensor from old to new,
       while keeping the indices as they are.
       """

    if not isinstance(old, str) and not isinstance(new, str):
        raise Inputerror(f"Tensor strings need to be of type {str}. "
                         f"Provided: {type(old)} old, {type(new)} new.")
    expr = expr.expand()
    if not isinstance(expr, Add):
        raise Inputerror("Can only filter an expression that is of "
                         f"type {Add}. Provided: {type(expr)}")

    to_change = filter_tensor(expr, old)
    if to_change is S.Zero:
        print(f"There is no Tensor with name '{old}' present in:\n{expr}")
        return expr
    remaining = expr - to_change  # gives S.Zero if all terms contain old

    def replace(term, old, new):
        ret = 1
        # term consists only of a single term, e.g. x
        if not isinstance(term, Mul):
            try:
                if term.symbol.name == old:
                    ret *= AntiSymmetricTensor(new, term.upper, term.lower)
            except AttributeError:
                raise RuntimeError("Something went wrong during filtering."
                                   f"Trying to replace Tensor '{old}' in "
                                   f"{expr}, but do not find the tensor.")
        else:  # multiple terms, e.g. x*y
            for t in term.args:
                if isinstance(t, AntiSymmetricTensor) and t.symbol.name == old:
                    ret *= AntiSymmetricTensor(new, t.upper, t.lower)
                else:
                    ret *= t
        return ret

    ret = 0
    if isinstance(to_change, Add):
        for term in to_change.args:
            ret += replace(term, old, new)
    # to_change consists only of 1 term
    else:
        ret += replace(to_change, old, new)
    return ret + remaining


def make_real(expr, *sym_tensors):
    """Makes all tensors real, i.e. removes the cc in their names.
       Additionally, the function tries to simplify the expression
       by allowing V_ab^ij = V_ij^ab // <ij||ab> = <ab||ij> and
       f_ij = f_ji. Additional tensors may be provided.
       Multiple occurences of the same tensor in a term are not
       supported atm.
       """
    from itertools import combinations

    def make_tensor_real(term):
        ret = 1
        if not isinstance(term, Mul):
            try:
                real_string = term.symbol.name.replace('c', '')
                ret *= AntiSymmetricTensor(real_string, term.upper, term.lower)
            except AttributeError:
                ret *= term
        else:
            for t in term.args:
                try:
                    real_string = t.symbol.name.replace('c', '')
                    ret *= AntiSymmetricTensor(real_string, t.upper, t.lower)
                except AttributeError:
                    ret *= t
        return ret

    def interchange_upper_lower(term, t_string):
        ret = 1
        if not isinstance(term, Mul):
            try:
                if term.symbol.name == t_string:
                    ret *= AntiSymmetricTensor(
                        t_string, term.lower, term.upper
                    )
            except AttributeError:
                raise RuntimeError("Something went wrong during filtering."
                                   f"Trying swap upper and lower {t_string} "
                                   f"indices in {latex(expr)}, but do not find"
                                   f" {t_string}.")
        else:
            for t in term.args:
                if isinstance(t, AntiSymmetricTensor) and \
                        t.symbol.name == t_string:
                    ret *= AntiSymmetricTensor(t_string, t.lower, t.upper)
                else:
                    ret *= t
        return ret

    def swap_tensors(t_terms, *t_strings):
        # swap the indices of all possible combinations of the requested
        # tensors, e.g. for V, f -> V / f / V,f
        # iterate over all terms and try all combinations in each term
        # if two terms match and may be combined -> start over again from
        # the beginning with the new expression
        # t_terms needs to contain all t_strings

        combs = [
            list(combinations(t_strings, i))
            for i in range(1, len(t_strings) + 1)
            ]
        for term in t_terms.args:
            for n_swaps in combs:
                for comb in n_swaps:
                    temp = term
                    for tensor in comb:
                        interchanged = interchange_upper_lower(temp, tensor)
                        new = t_terms - temp + interchanged
                        if not isinstance(new, Add):
                            return new
                        elif len(new.args) < len(t_terms.args):
                            return swap_tensors(new, *t_strings)
                        temp = interchanged
        return t_terms

    def prescan_terms(t_terms, *t_strings):
        # prescan and collect the indices of of all terms that
        # are wothwhile trying to simplify

        # collect all indices of the desired tensors
        indices = []  # list(dict(set()))
        for term in t_terms.args:
            temp = {}
            if not isinstance(term, Mul):
                if isinstance(term, AntiSymmetricTensor) and \
                        term.symbol.name in t_strings:
                    up = "".join(sorted([s.name for s in term.upper]))
                    lo = "".join(sorted([s.name for s in term.lower]))
                    temp[term.symbol.name] = {up, lo}
                else:
                    raise RuntimeError("Something went wrong during filtering."
                                       " Do not find {t_string} in {term}.")
            else:
                for t in term.args:
                    if isinstance(t, AntiSymmetricTensor) and \
                            t.symbol.name in t_strings:
                        up = "".join(sorted([s.name for s in t.upper]))
                        lo = "".join(sorted([s.name for s in t.lower]))
                        temp[t.symbol.name] = {up, lo}
            indices.append(temp)
        if len(t_terms.args) != len(indices):
            raise RuntimeError(f"Found {len(indices)} occurences of tensor "
                               f"{t_strings}, but have only "
                               f"{len(t_terms.args)} terms.")
        matching_terms = []
        # store all terms that are already matched -> avoid double counting
        matched = []
        for i, term in enumerate(indices):
            temp = [i]
            if i in matched:
                continue
            matched.append(i)
            for j in range(i + 1, len(indices)):
                match = True
                for t, idx in term.items():
                    if idx == indices[j][t]:
                        continue
                    else:
                        match = False
                if match:
                    temp.append(j)
                    matched.append(j)
            if len(temp) > 1:
                matching_terms.append(temp)
        return matching_terms

    for t in sym_tensors:
        if not isinstance(t, str):
            raise Inputerror("Tensor string must be of type str, not "
                             f"{type(t)} {t}.")
    expr = expr.expand()
    ret = 0
    # 1) replace txcc with tx
    if isinstance(expr, Add):
        for term in expr.args:
            ret += make_tensor_real(term)
    else:
        ret += make_tensor_real(expr)

    # 2) allow V_ij^ab = V_ab^ij / f_ij = f_ji / d_ij = d_ji for any other
    #    user provided tensor

    for t in ["V", "f"]:  # add V and f to tensors if not provided
        if t not in sym_tensors:
            sym_tensors += (t,)
    combs = [
        list(combinations(sym_tensors, r))
        for r in range(1, len(sym_tensors) + 1)
    ]
    # partition the terms according the tensors included, e.g. terms with
    # (only!) V, (only!) f and terms with V and f
    t_terms = {}
    for n_tensors in combs:
        for comb in n_tensors:
            terms = filter_tensor(ret, *comb)
            if terms is not S.Zero:
                t_terms[comb] = terms
    only_t = {}
    # iterate over combinations, starting with the full combination
    for comb in sorted(t_terms, key=len, reverse=True):
        # combination of all tensors -> nothing to subtract (V*f)
        if len(comb) == len(sym_tensors):
            only_t[comb] = t_terms[comb]
        # smaller combination -> need to subtract all pure higher combinations
        # to obtain only terms that contain the combination and nothing else
        # (atm V also contains terms V*f etc)
        else:
            temp = t_terms[comb]
            for only_comb, only_terms in only_t.items():
                if len(only_comb) > len(comb) and \
                        set(comb).issubset(only_comb):
                    temp -= only_terms
            only_t[comb] = temp
    # collect terms without any relevant tensors
    remaining = ret
    for terms in only_t.values():
        remaining -= terms

    # check if partition is correct
    is_zero = ret - remaining
    for terms in only_t.values():
        is_zero -= terms
    if is_zero is not S.Zero:
        raise RuntimeError("Partition did not work correctly. The following "
                           f"should be 0:\n{latex(is_zero)}")

    res = remaining
    for t_strings, terms in only_t.items():
        if not isinstance(terms, Add):
            res += terms
            continue
        # indices of terms that may be simplified
        matching_terms = prescan_terms(terms, *t_strings)
        temp = 0
        swapped = []
        for match in matching_terms:
            to_check = Add(*[terms.args[i] for i in match])
            swapped.extend(match)
            temp += swap_tensors(to_check, *t_strings)
        temp += Add(*[
            terms.args[i] for i in range(len(terms.args)) if i not in swapped
        ])
        res += temp

    return res


def remove_tensor(expr, t_string):
    """Removes a tensor that is present in each term of the expression."""

    expr = expr.expand()
    to_remove = filter_tensor(expr, t_string)
    if expr - to_remove is not S.Zero:
        raise RuntimeError(f"Not all terms in {latex(expr)} contain "
                           f"the tensor {t_string}.")

    def remove(term, t_string):
        ret = 1
        if isinstance(term, Mul):
            for t in term.args:
                if isinstance(t, AntiSymmetricTensor) and \
                        t.symbol.name == t_string:
                    continue
                ret *= t
        else:  # only a single object: x... must be the tensor to remove
            if not isinstance(term, AntiSymmetricTensor) and \
                    term.symbol.name != t_string:
                raise RuntimeError("Something went wrong during filtering. "
                                   f"The term {term} does not contain the "
                                   f"tensor {t_string}.")
            ret *= term
        return ret

    ret = 0
    if isinstance(expr, Add):
        for term in expr.args:
            ret += remove(term, t_string)
    else:
        ret += remove(expr, t_string)
    return ret


def simplify(expr, *sym_tensors):
    # simplify expr by interchanging indice names to collect terms

    def index_positions(obj, *sym_tensors):
        # input: a single object that is part of the Mul object, e.g. t_ij^ab
        # tensor: name, upper/lower // delta // create // annihilate
        ret = {}
        if isinstance(obj, AntiSymmetricTensor):
            upper = [i for i in obj.upper]
            lower = [i for i in obj.lower]
            # if tensor is symmetric f_ia = f_ai -> upper=lower
            if obj.symbol.name in sym_tensors:
                idx = upper + lower
                for i in idx:
                    if i not in ret:
                        ret[i] = []
                    ret[i].append("tensor_" + obj.symbol.name)
            else:
                for i in upper:
                    if i not in ret:
                        ret[i] = []
                    ret[i].append("tensor_" + obj.symbol.name + "_upper")
                for i in lower:
                    if i not in ret:
                        ret[i] = []
                    ret[i].append("tensor_" + obj.symbol.name + "_lower")
        elif isinstance(obj, KroneckerDelta):
            for i in obj.free_symbols:
                ret[i] = ["delta"]
        elif isinstance(obj, F):
            i = next(iter(obj.free_symbols))  # there can only be 1 index
            ret[i] = ["create"]
        elif isinstance(obj, Fd):
            i = next(iter(obj.free_symbols))  # there can only be one index
            ret[i] = ["annihilate"]
        # in case of prefactors return empty dict
        return ret

    def term_pattern(term, *sym_tensors):
        ret = {"occ": [], "virt": []}
        indices = {}
        if isinstance(term, Mul):
            for t in term.args:
                for i, occurence in index_positions(t, *sym_tensors).items():
                    if i not in indices:
                        indices[i] = []
                    indices[i].extend(occurence)
        else:  # term consists only of a single obj x
            for i, occurence in index_positions(term, *sym_tensors).items():
                if i not in indices:
                    indices[i] = []
                indices[i].extend(occurence)
        for i, occurence in indices.items():
            ret[assign_index(i.name)].append((i, *occurence))
        return ret

    for t in sym_tensors:
        if not isinstance(t, str):
            raise Inputerror(f"Tensor string {t} must be of type str, "
                             f"not {type(t)}.")

    expr = expr.expand()
    if not isinstance(expr, Add):
        print("Can't simplify an expression that consists of a single term.")
        return expr

    # collect index name, type and occurence in each term
    pattern = {}  # {#_term: {length: l, pattern: {ov: [(name, occurence)]}}}
    for i, term in enumerate(expr.args):
        pattern[i] = {}
        # number of tensors deltas etc. in the term, including prefactors!
        pattern[i]["length"] = len(term.args) if isinstance(term, Mul) else 1
        pattern[i]["pattern"] = term_pattern(term, *sym_tensors)
    # print(pattern)

    equal_terms = {}
    matched_terms = []
    # iterate over terms and compare length and pattern
    for term_n, lp in pattern.items():
        matched_terms.append(term_n)
        for other_term_n, other_lp in pattern.items():
            # skip term_n and terms that have been matched already
            if other_term_n in matched_terms:
                continue
            # the length of the terms may only differ by 1 (by a prefactor)
            # and the terms need to share the same amount of o/v indices
            if not abs(lp["length"] - other_lp["length"]) <= 1 or not \
                    len(lp["pattern"]["occ"]) == \
                    len(other_lp["pattern"]["occ"]) or not \
                    len(lp["pattern"]["virt"]) == \
                    len(other_lp["pattern"]["virt"]):
                continue
            matched_ov = []
            sub = []  # collect equal indices as tuple
            for ov in ["occ", "virt"]:
                if not all(matched_ov):
                    continue  # occ already did not match -> skip virt
                p1 = lp["pattern"][ov]  # [(name, occurences),]
                p2 = other_lp["pattern"][ov]
                equal_idx = []  # count equal indices independent of their name
                # can only map each index in term2 to 1 index in term1
                # makes logically sense and otherwise weird stuf may happen
                matched_idx = []
                # check which indices in both terms are equal
                # if all occurences are equal the indices are equal
                for idx1 in p1:
                    for idx2 in p2:
                        if idx2[0] in matched_idx:
                            continue
                        if all(ele in idx2[1:] for ele in idx1[1:]):
                            matched_idx.append(idx2[0])
                            equal_idx.append((idx2[0], idx1[0]))
                            break
                # found a match for all indices
                if len(equal_idx) == len(p1):
                    matched_ov.append(True)
                    sub_idx = []  # only need idx pairs that are not equal
                    for pair in equal_idx:
                        if pair[0] != pair[1]:
                            sub_idx.append(pair)
                    sub.extend(sub_idx)
            # term and other_term are equal if all indices are matched for o/v.
            # But only when there is something to do, i.e. some indices need to
            # be in sub (possible to be empty if a symmetric tensor is defined,
            # but indices are already equal and only need to be swapt
            # -> make real needs to be used in this case)
            if matched_ov and all(matched_ov) and sub:
                if term_n not in equal_terms:
                    equal_terms[term_n] = {}
                equal_terms[term_n][other_term_n] = sub
                matched_terms.append(other_term_n)
    ret = 0
    matched_terms = []  # to construct the unchanged remainder
    for term_n in equal_terms:
        matched_terms.append(term_n)
        ret += expr.args[term_n]
        for other_term_n in equal_terms[term_n]:
            matched_terms.append(other_term_n)
            # substitute indices in other_term
            sub = equal_terms[term_n][other_term_n]
            ret += expr.args[other_term_n].subs(sub, simultaneous=True)
    remainder = Add(*[expr.args[i] for i in range(len(expr.args))
                      if i not in matched_terms])
    return ret + remainder
