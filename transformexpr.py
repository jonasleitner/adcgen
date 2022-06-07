from sympy import KroneckerDelta, Add, S, Mul, Pow, latex, Dummy
from sympy.physics.secondquant import AntiSymmetricTensor, F, Fd
from indices import index_space
from misc import Inputerror, transform_to_tuple
import time


# TODO: support non integer denominator, i.e. Add may occur as part of a
#       Mul object. Currently all functions only support Mul as
#       part of Add. (needed for canonical amplitude equation)
# TODO: support NO objects

# factor can factor out a common factor in a expression but only if all
# components of the expression contain the same common factor

def sort_by_n_deltas(expr):
    """Sort terms in the expression according to the number of deltas"""

    def count_deltas(term):
        counter = 0
        if isinstance(term, Mul):
            for t in term.args:
                data = __obj_data(t)
                if data["type"] == "delta":
                    counter += 1
        else:
            data = __obj_data(term)
            if data["type"] == "delta":
                counter += data["exponent"]
        return counter

    expr = expr.expand()
    ret = {}
    if isinstance(expr, Add):
        for term in expr.args:
            key = count_deltas(term)
            if key not in ret:
                ret[key] = 0
            ret[key] += term
    else:
        ret[count_deltas(expr)] = expr
    return ret


def sort_by_type_deltas(expr):
    """Sort the terms in an expression according to the type of deltas
       (if the delta indices are occ or virt indices).
       """

    def delta_types(term):
        ret = []
        if isinstance(term, Mul):
            for t in term.args:
                data = __obj_data(t)
                if data["type"] == "delta":
                    ov1 = index_space(data["preferred"].name)[0]
                    ov2 = index_space(data["killable"].name)[0]
                    ret.append("".join(sorted(ov1 + ov2)))
        else:
            data = __obj_data(term)
            if data["type"] == "delta":
                ov1 = index_space(data["preferred"].name)[0]
                ov2 = index_space(data["killable"].name)[0]
                ret.append("".join(sorted(ov1 + ov2)))
        if not ret:
            ret = ["no_deltas"]
        return tuple(ret)

    expr = expr.expand()
    ret = {}
    if isinstance(expr, Add):
        for term in expr.args:
            key = delta_types(term)
            if key not in ret:
                ret[key] = 0
            ret[key] += term
    else:
        ret[delta_types(expr)] = expr
    return ret


def __obj_data(obj):
    """Function that extracts information of an object and returns them
       as dict. Supports Tensors, deltas, create and annihilate as wells as
       Power objects of all the afore mentioned objects."""

    types = {
        AntiSymmetricTensor: "tensor",
        KroneckerDelta: "delta",
        F: "annihilate",
        Fd: "create",
    }

    ret = {}
    if isinstance(obj, Pow):  # Power object
        ret.update(__obj_data(obj.args[0]))
        ret["exponent"] = obj.args[1]
        return ret
    try:
        ret["type"] = types[type(obj)]
    except KeyError:   # prefactor / unknown objects
        ret["type"] = "prefactor"
        if len(obj.free_symbols) != 0:
            raise Inputerror(f"Unknown object {latex(obj)} of type "
                             f"{type(obj)}.")
        return ret
    if isinstance(obj, AntiSymmetricTensor):
        ret["name"] = obj.symbol.name
        ret["upper"] = obj.upper
        ret["lower"] = obj.lower
    elif isinstance(obj, KroneckerDelta):
        ret["preferred"] = obj.preferred_index
        ret["killable"] = obj.killable_index
    else:  # create/annihilate
        ret["index"] = obj.args[0]
    ret["exponent"] = 1
    return ret


def sort_by_type_tensor(expr, t_string, symmetric=False):
    """Sorts an expression according the blocks of a Tensor, i.e.
       the upper and lower indices are assigned to occ/virt.
       The terms are collected in a dict with e.g. 'ov' or 'ooov'
       as keys."""

    def assign_term(term, t_string, symmetric):
        # determines which block of the desired tensor occures in the term
        # (if the tensor occures at all)
        ret = []
        if isinstance(term, Mul):
            for t in term.args:
                data = __obj_data(t)
                if data["type"] == "tensor" and data["name"] == t_string:
                    ov_up = sorted(
                        [index_space(s.name)[0] for s in data["upper"]]
                    )
                    ov_lo = sorted(
                        [index_space(s.name)[0] for s in data["lower"]]
                    )
                    if symmetric:
                        ov = "".join(sorted(ov_up + ov_lo))
                    else:
                        ov = "".join(ov_up + ov_lo)
                    ret.append(ov)
        else:
            data = __obj_data(term)
            if data["type"] == "tensor" and data["name"] == t_string:
                ov_up = sorted(
                    [index_space(s.name)[0] for s in data["upper"]]
                )
                ov_lo = sorted(
                    [index_space(s.name)[0] for s in data["lower"]]
                )
                if symmetric:
                    ov = "".join(sorted(ov_up + ov_lo))
                else:
                    ov = "".join(ov_up, ov_lo)
                ret.append(ov)
        if not ret:
            ret = [f"no_{t_string}"]
        return tuple(ret)

    expr = expr.expand()
    ret = {}
    if isinstance(expr, Add):
        for term in expr.args:
            key = assign_term(term, t_string, symmetric)
            if key not in ret:
                ret[key] = 0
            ret[key] += term
    else:
        ret[assign_term(expr, t_string, symmetric)] = expr
    return ret


def filter_tensor(expr, t_strings, strict=False):
    """Returns terms of an expression that contain all of the requested
       tensors. The function also filters for the complex conjugate
       tensors, where it is assumed that the names of T and T* differ only
       by (one or more) 'c'.
       If strict is set to True, the function also checks how often each of the
       requested tensors occurs in the term, i.e. if ['f', 'f'] is requested
       only terms with f*f are returned. If strict is set to False ['f', 'f']
       will return any terms that contain at least 1 f. Note that in both cases
       the other tensors in the term are not checked, i.e. 'f,f', strict=True
       also returns terms V*f*f etc.
       f^2 is treated as f*f.
       """
    from collections import Counter

    t_strings = transform_to_tuple(t_strings)
    for t in t_strings:
        if "c" in t:
            raise Inputerror("Can not search for a tensor with 'c' in it's "
                             "name. The function filters it to "
                             "also return the complex conjugate of the desired"
                             f" tensor. Provided tensor strings: {t_strings}.")

    def check_term(term, t_strings, strict):
        found = []
        if isinstance(term, Mul):
            for t in term.args:
                # just a tensor
                if isinstance(t, AntiSymmetricTensor) and \
                        t.symbol.name.replace("c", "") in t_strings:
                    found.append(t.symbol.name.replace("c", ""))
                # tensor raised to a power -> count e.g. f^2 as f*f
                elif isinstance(t, Pow):
                    if isinstance(t.args[0], AntiSymmetricTensor) and \
                            t.args[0].symbol.name.replace("c", "") in \
                            t_strings:
                        for i in range(t.args[1]):
                            found.append(
                                t.args[0].symbol.name.replace("c", "")
                            )
        # single power object
        elif isinstance(term, Pow):
            if isinstance(term.args[0], AntiSymmetricTensor) and \
                    term.args[0].symbol.name.replace("c", "") in t_strings:
                for i in range(term.args[1]):
                    found.append(term.args[0].symbol.name.replace("c", ""))
        else:  # single tensor object (or nothing of not a tensor)
            if isinstance(term, AntiSymmetricTensor) and \
                    term.symbol.name.replace("c", "") in t_strings:
                found.append(term.symbol.name.replace("c", ""))
        if not strict and all(t in found for t in t_strings):
            return term
        # only return if the number of all requested indices match
        # exactly. But still all not requested tensors are not
        # taken into account!
        elif strict and Counter(t_strings) == Counter(found):
            return term
        return S.Zero

    expr = expr.expand()
    tensor = 0
    if isinstance(expr, Add):
        for term in expr.args:
            tensor += check_term(term, t_strings, strict)
    else:
        tensor += check_term(expr, t_strings, strict)
    return tensor


def __term_contracted_indices(term):
    """Count how often an index occurs in a term, i.e. if the index is
       contracted or a target index (0 corresponds to 1 occurence).
       """
    idx = {}
    if isinstance(term, Mul):
        for t in term.args:
            for s in t.free_symbols:
                if isinstance(s, Dummy):
                    if s in idx:
                        idx[s] += 1
                    else:
                        idx[s] = 0
    else:
        for s in term.free_symbols:
            if isinstance(s, Dummy):
                if s in idx:
                    idx[s] += 1
                else:
                    idx[s] = 0
    return idx


def sort_tensor_contracted_indices(expr, t_string):
    """Sorts an expression by sorting the terms depending on the
       number and type (ovv/virt) of indices of an AntiSymmetricTensor
       that are contracted. The function assumes that indices are summed
       if they appear more than once in the term.
       """

    def term_tensor_contr(term, t_string):
        # 1) count how often each index occurs in the term
        idx = __term_contracted_indices(term)
        # 2) check how often each index of the desired tensor occurs in the
        #    term, i.e. if it is contracted or not (einstein sum convention)
        ret = {}
        contracted = ""
        if isinstance(term, Mul):
            for t in term.args:
                data = __obj_data(t)
                if data["type"] == "tensor" and data["name"] == t_string:
                    contr_up = [
                        index_space(s.name)[0] for s in data["upper"]
                        if idx[s]
                    ]
                    contr_lo = [
                        index_space(s.name)[0] for s in data["lower"]
                        if idx[s]
                    ]
                    c = "".join(sorted(contr_up + contr_lo))
                    if contracted and contracted != c:
                        raise Inputerror("Found more than one occurence of "
                                         f"tensor {t_string} in {latex(term)} "
                                         "with different contracted indices.")
                    contracted = c
        else:  # single obj
            data = __obj_data(term)
            if data["type"] == "tensor" and data["name"] == t_string:
                contr_up = [
                    index_space(s.name) for s in data["upper"] if idx[s]
                ]
                contr_lo = [
                    index_space(s.name) for s in data["lower"] if idx[s]
                ]
                c = "".join(sorted(contr_up + contr_lo))
                if contracted and contracted != c:
                    raise Inputerror("Found more than one occurence of "
                                     f"tensor {t_string} in {latex(term)} "
                                     "with different contracted indices.")
        if not contracted:
            contracted = "none"
        ret[contracted] = term
        return ret

    expr = expr.expand()
    # prefilter all terms that contain the desired tensor
    tensor = filter_tensor(expr, t_string, strict=False)

    ret = {}
    # get all terms that do not contain the tensor
    if expr - tensor != S.Zero:
        ret[f'no_{t_string}'] = expr - tensor
        # we only have terms that do not contain the tensor-
        if ret[f'no_{t_string}'] == expr:
            return ret

    ret = {}
    if isinstance(expr, Add):
        for term in tensor.args:
            term_contr = term_tensor_contr(term, t_string)
            for contr, t in term_contr.items():
                if contr not in ret:
                    ret[contr] = 0
                ret[contr] += t
    else:  # only a single term
        term_contr = term_tensor_contr(expr, t_string)
        for contr, t in term_contr.items():
            ret[contr] = t
    return ret


def change_tensor_name(expr, old, new):
    """Changes the Name of a AntiSymmetricTensor from old to new,
       while leaving the indices unchanged.
       """

    if not isinstance(old, str) and not isinstance(new, str):
        raise Inputerror(f"Tensor strings need to be of type {str}. "
                         f"Provided: {type(old)} old, {type(new)} new.")
    expr = expr.expand()

    to_change = filter_tensor(expr, old, strict=False)
    if to_change is S.Zero:
        print(f"There is no Tensor with name '{old}' present in:\n{expr}")
        return expr
    remaining = expr - to_change  # gives S.Zero if all terms contain old

    def replace(term, old, new):
        ret = 1
        if isinstance(term, Mul):
            for t in term.args:
                data = __obj_data(t)
                if data["type"] == "tensor" and data["name"] == old:
                    ret *= Pow(
                        AntiSymmetricTensor(new, data["upper"], data["lower"]),
                        data["exponent"]
                    )
                else:
                    ret *= t
        else:  # single obj
            data = __obj_data(term)
            if data["type"] == "tensor" and data["name"] == old:
                ret *= Pow(
                    AntiSymmetricTensor(new, data["upper"], data["lower"]),
                    data["exponent"]
                )
            else:
                ret *= term
        return ret

    ret = remaining
    if isinstance(to_change, Add):
        for term in to_change.args:
            ret += replace(term, old, new)
    # to_change consists only of 1 term
    else:
        ret += replace(to_change, old, new)
    return ret


def make_tensor_names_real(expr):
    """Renames all tensors by removing any 'c' in their name.
       This essentially makes makes all t-amplitudes real (and possibly
       ADC amplitudes), but only works if 'c' is not used in names of tensors
       in any other context than defining the complex conjugate.
       """

    def make_tensor_real(term):
        ret = 1
        if isinstance(term, Mul):
            for t in term.args:
                data = __obj_data(t)
                if data["type"] != "tensor":  # no tensor
                    ret *= t
                    continue
                # remove 'c' from name and create new Tensor/Pow obj
                # (Pow obj with exponent 1 is a Tensor obj)
                r_name = data["name"].replace("c", "")
                ret *= Pow(
                    AntiSymmetricTensor(r_name, data["upper"], data["lower"]),
                    data["exponent"]
                )
        else:
            data = __obj_data(term)
            if data["type"] != "tensor":  # no tensor
                ret *= term
            else:
                r_name = data["name"].replace("c", "")
                ret *= Pow(
                    AntiSymmetricTensor(r_name, data["upper"], data["lower"]),
                    data["exponent"]
                )
        return ret

    expr = expr.expand()
    ret = 0
    if isinstance(expr, Add):
        for term in expr.args:
            ret += make_tensor_real(term)
    else:
        ret += make_tensor_real(expr)
    return ret


def make_real(expr, *sym_tensors):
    """Makes all tensors real, i.e. removes the cc in their names.
       By default the fock matrix f and the ERI's V are assumed
       to be symmetric. Other tensors strings need to be provided as
       function arguments.
       The function tries to simplify the expression
       by allowing V_ab^ij = V_ij^ab // <ij||ab> = <ab||ij> and
       f_ij = f_ji.
       """
    from itertools import combinations
    from collections import Counter

    def max_tensor_occurence(expr, *t_strings):
        # count how often a tensor occurs in each term
        # the maximum amount is stored and returned in a dict

        def term_occurence(term, *t_strings):
            ret = {}
            for tensor in t_strings:
                ret[tensor] = 0
            if isinstance(term, Mul):
                for t in term.args:
                    data = __obj_data(t)
                    if data["type"] != "tensor":
                        continue
                    if data["name"] in t_strings:
                        ret[data["name"]] += data["exponent"]
            else:
                data = __obj_data(term)
                if data["type"] != "tensor":
                    return ret
                if data["name"] in t_strings:
                    ret[data["name"]] += data["exponent"]
            return ret

        ret = {}
        for tensor in t_strings:
            ret[tensor] = 0
        if isinstance(expr, Add):
            for term in expr.args:
                temp = term_occurence(term, *t_strings)
                for tensor, n in temp.items():
                    if ret[tensor] < n:
                        ret[tensor] = n
        else:
            ret = term_occurence(expr, *t_strings)
        return ret

    def interchange_upper_lower(term, t_string, n=1):
        # interchanges the upper and lower indices of the n'th occurence of a
        # single tensor in a single term
        # n starts from 1 which marks the first tensor in the term, 2 marks
        # the second etc.
        # the power to which a tensor is raised is taken into account
        # -> f'*f^2: f' is first, f second and third

        def swap(data):
            # swap upper and lower indices and lower the exponent of the
            # original tensor by 1 (exponent=0 -> 1, exponent=1 -> Tensor)
            ret = AntiSymmetricTensor(t_string, data["lower"], data["upper"])
            ret *= Pow(
                AntiSymmetricTensor(t_string, data["upper"], data["lower"]),
                data["exponent"]-1
            )
            return ret

        ret = 1
        swapped = {}
        if isinstance(term, Mul):
            counter = 1
            for t in term.args:
                data = __obj_data(t)
                if data["type"] != "tensor":  # no tensor
                    ret *= t
                    continue
                if data["name"] == t_string:
                    if counter <= n < counter+data["exponent"]:
                        ret *= swap(data)
                        swapped.update(data)
                    else:  # wrong counter
                        ret *= t
                    counter += data["exponent"]
                else:  # wrong tensor
                    ret *= t
        else:  # single tensor/Pow object
            data = __obj_data(term)
            if data["type"] != "tensor":  # no tensor
                ret *= term
            if data and data["name"] == t_string:
                if 0 < n <= data["exponent"]:
                    ret *= swap(data)
                    swapped.update(data)
                else:  # wrong counter
                    ret *= term
            else:  # wrong tensor
                ret *= term
        # if the result is identical, but upper and lower indices of the
        # swapped tensor differ there is something wrong
        if term-ret is S.Zero and swapped["upper"] != swapped["lower"]:
            raise RuntimeError(f"Something went wrong. Trying to swap {n}'th"
                               f" {t_string} tensor in term {latex(term)}. "
                               f"But result is identical to input: "
                               f"{latex(ret)}.")
        return ret

    def swap_tensors(t_terms, *t_strings):
        # swap the indices of all possible combinations of the requested
        # tensors, e.g. for V, f -> V / f / V,f
        # iterate over all terms and try all combinations in each term
        # if two terms match and may be combined -> start over again from
        # the beginning with the new expression
        # t_terms needs to contain all t_strings

        # attach a number to each tensor to keep track of which occurence of
        # a tensor in a term is meant.
        t_n_pairs = []
        for t, count in Counter(t_strings).items():
            t_n_pairs.extend([(t, n) for n in range(1, count+1)])
        combs = [
            list(combinations(t_n_pairs, i))
            for i in range(1, len(t_n_pairs) + 1)
        ]  # [[((t, n), )]] 2 list 2 tuple
        if isinstance(t_terms, Add):
            for term in t_terms.args:
                # n_swaps is a nested list of combinations. Each list in combs
                # contains a list of combinations that contain a different
                # amount of tensors to swap in the term
                for n_swaps in combs:
                    # each combination contains n tensors to be swapt one after
                    # another in the term
                    for comb in n_swaps:
                        temp = term
                        # swap one tensor after another updating temp to
                        # simultaneously swap multiple tensors in the term
                        for tensor in comb:
                            interchanged = interchange_upper_lower(
                                temp, tensor[0], tensor[1]
                            )
                            new = t_terms - term + interchanged
                            # start over again from scratch with the new expr
                            # if:
                            # - only 1 term left -> try simplify (check if Pow
                            #   can be created)
                            # - the interchanged term is not longer than the
                            #   original (f^2 -> f*f') and we can collect two
                            #   terms
                            if not isinstance(new, Add):
                                return swap_tensors(new, *t_strings)
                            elif len(new.args) < len(t_terms.args) and \
                                    len(interchanged.args) <= len(term.args):
                                return swap_tensors(new, *t_strings)
                            temp = interchanged
        # check whether its possible to simplify any further by creating Pow
        # objects
        elif isinstance(t_terms, Mul):
            for n_swaps in combs:
                for comb in n_swaps:
                    temp = t_terms
                    for tensor in comb:
                        new = interchange_upper_lower(
                            temp, tensor[0], tensor[1]
                        )
                        if not isinstance(new, Mul):
                            return new
                        elif len(new.args) < len(t_terms.args):
                            return swap_tensors(new, *t_strings)
                        temp = new
        # for Pow object don't do anything... or a single obj
        return t_terms

    def prescan_terms(t_terms, *t_strings):
        # pre filter terms that contain all of the desired tensors
        # only terms with e.g. f*f will reach this function
        # -> no need to count the occurences and assign the indices a number
        # 1) collect all indices of the desired tensors in each term
        # 2) compare if indices match in different terms, i.e.
        #    if they may be equal when swapping
        #    upper and lower

        # returns a list of a list
        # only Add obj need to get in here

        # collect all indices of the desired tensors in each term
        indices = []  # list({[set(up, lo),]}) // [{t_name: [{up, low},]},]
        for term in t_terms.args:
            temp = {}
            for tensor in set(t_strings):
                temp[tensor] = []
            if isinstance(term, Mul):
                for t in term.args:
                    data = __obj_data(t)
                    if data["type"] != "tensor":  # no tensor
                        continue
                    if data["name"] in t_strings:
                        up = "".join(sorted([s.name for s in data["upper"]]))
                        lo = "".join(sorted([s.name for s in data["lower"]]))
                        # count a Pow obj n times
                        temp[data["name"]].extend(
                            [{up, lo} for i in range(data["exponent"])]
                        )
            else:  # single obj/Pow
                data = __obj_data(term)
                if data["type"] != "tensor":
                    continue
                if data["name"] in t_strings:
                    up = "".join(sorted([s.name for s in data["upper"]]))
                    lo = "".join(sorted([s.name for s in data["lower"]]))
                    # count Pow obj n times
                    temp[data["name"]].extend(
                        [{up, lo} for i in range(data["exponent"])]
                    )
            # check if a match for all tensors in set(t_strings) is found
            if not all(temp[tensor] for tensor in set(t_strings)):
                raise RuntimeError("Something went wrong during filtering or "
                                   f"partitioning. Do not find all {t_strings}"
                                   f" in {term}. Found {temp}.")
            # also check whether the correct amount of tensors (and indices)
            # was found, e.g. only 1 or the desired 2 f tensors
            count = Counter(t_strings)
            for tensor, idx in temp.items():
                if count[tensor] != len(idx):
                    raise RuntimeError(f"Did not collect {count[tensor]} "
                                       f"indices for tensor {tensor} in "
                                       f"{latex(term)}. Collected: {idx}")
            indices.append(temp)

        # list of term indices that may be simplified by swapping
        matching_terms = []  # nested list: [[0,1,2], [3,5]]
        # store all terms that are already matched -> avoid double counting
        matched = []
        # compare the indices, if the indices are identical independent of
        # their position (upper/lower) the terms may be simplified
        for i, term in enumerate(indices):
            temp = [i]
            if i in matched:
                continue
            for j in range(i + 1, len(indices)):
                if j in matched:
                    continue
                match = True
                # terms match if I find exactly the same indices attatched to
                # the same tensors in both terms. Check this from 'both sides'.
                # But first check whether the same amount of indices was found
                # in both terms, i.e. the same amount of tensors in both terms
                for t, idxl in term.items():
                    # best would be to use Counter, but dict does not work with
                    # set as key
                    if len(idxl) != len(indices[j][t]) or \
                            not all(idx in indices[j][t] for idx in idxl) or \
                            not all(idx in idxl for idx in indices[j][t]):
                        match = False
                        break
                if match:
                    temp.append(j)
            matched.extend(temp)
            # did I find a match for i?
            if len(temp) > 1:
                matching_terms.append(temp)
        return matching_terms

    start = time.time()
    for t in sym_tensors:
        if not isinstance(t, str):
            raise Inputerror("Tensor string must be of type str, not "
                             f"{type(t)} {t}.")
    for t in ["V", "f"]:  # add V and f to tensors if not provided
        if t not in sym_tensors:
            sym_tensors += (t,)
    # get rid of multiple occurences of tensors in user input
    sym_tensors = set(sym_tensors)

    # start working on the expression
    expr = expr.expand()
    # 1) replace txcc with tx
    expr = make_tensor_names_real(expr)

    # The complicate part:
    # 2) allow V_ij^ab = V_ab^ij / f_ij = f_ji
    #    and d_ij = d_ji for any tensor in sym_tensors

    # First partition the expr according to the sym_tensors included in the
    # term, e.g. terms with: only V / only f / V * f / f*f etc.

    # - get the maximum occurences of all sym_tensor in a term
    #   if more than one occurence -> attach the tensor again
    #   to cover all combinations of tensors that may occur in a term
    n_sym_tensors = max_tensor_occurence(expr, *sym_tensors)
    relevant_sym_tensors = tuple()
    for t in sym_tensors:
        relevant_sym_tensors += tuple(t for i in range(n_sym_tensors[t]))
    combs = [
        list(combinations(relevant_sym_tensors, r))
        for r in range(1, len(relevant_sym_tensors) + 1)
    ]
    # call filter_tensor for all combinations
    t_terms = {}
    for n_tensors in combs:
        # collect combinations that are not present in the expression
        zero = []
        for comb in n_tensors:
            # avoid calling filter_tensor twice for an identical comb
            if comb in t_terms or comb in zero:
                continue
            terms = filter_tensor(expr, comb, strict=True)
            # found some terms with the combination of tensors
            if terms is not S.Zero:
                t_terms[comb] = terms
            else:
                zero.append(comb)

    # iterate over combinations that are present in the expression, starting
    # with the largest combination, e.g. V*f
    # Because of the way how filter_tensor works, this combination should
    # should contain only terms with exactly that combination of tensors
    # (+ some irrelevant tensors like ADC or t-Amplitudes)
    # for smaller combinations, some of the higher pure combinations need to
    # be subtracted, because there are overlapping terms that are present in
    # both filter_tensor(comb) expressions. More precisely all higher
    # combinations that differ by a tensor that is not present in the
    # smaller combination. For example we have the tensors a,b and c with the
    # maximum occurence of tensors in a term: 'aabc'. Filter tensor will
    # return pure terms for the combinations: 'aabc' and 'abc'. All other
    # combinations require the subtraction of other pure combinations:
    # aab/aac: - aabc // ab/ac: - abc // bc: - aabc,abc // aa: - aabc,aab,aac
    # a: - abc,ab,ac // b: aabc,abc,aab,ab,bc // c: aabc,abc,aac,ac,bc
    only_t = {}  # {comb: terms} // comb = tuple of t_strings
    for comb in sorted(t_terms, key=len, reverse=True):
        # combinations without anything to subtract.
        if all(tensor in comb for tensor in set(relevant_sym_tensors)):
            only_t[comb] = t_terms[comb]
        # combinations with overlapping terms
        else:
            temp = t_terms[comb]
            # all higher combinations should already have been calculated
            for only_comb, only_terms in only_t.items():
                if len(comb) < len(only_comb):
                    count = Counter(comb)
                    only_count = Counter(only_comb)
                    subtract = []
                    for t, n in count.items():
                        if t in only_count and n == only_count[t]:
                            subtract.append(True)
                        else:
                            subtract.append(False)
                    if subtract and all(subtract):
                        temp -= only_terms
            if temp is not S.Zero:
                only_t[comb] = temp
    # collect terms without any relevant tensors
    remaining = expr
    for terms in only_t.values():
        remaining -= terms

    # now try to collect terms in all the pure sub expressions by swapping all
    # sym tensors in each term (one after another and all possible
    # combinations).
    res = remaining
    for t_strings, terms in only_t.items():
        if isinstance(terms, Add):
            # the position of terms that possibly may be simplified according
            # to the indices of the relevant tensors. Format: [[0,1],[3,4,6]]
            matching_terms = prescan_terms(terms, *t_strings)
            temp = 0
            swapped = []  # keep track of the terms for which a match was found
            for match in matching_terms:
                to_check = Add(*[terms.args[i] for i in match])
                temp += swap_tensors(to_check, *t_strings)
                swapped.extend(match)
            # Add terms for which no match was found
            temp += Add(*[
                terms.args[i] for i in range(len(terms.args))
                if i not in swapped
            ])
            res += temp
        # Also a single term may be simplified by introducing Pow objects!
        else:
            res += swap_tensors(terms, *t_strings)
    print(f"old make_real took {time.time() - start} seconds.")
    return res


def remove_tensor(expr, t_string):
    """Removes a tensor that is present in each term of the expression."""
    # TODO: probably also collect the indices of the removed tensor and return
    #       them too.
    # TODO: treat Pow objects

    expr = expr.expand()
    to_remove = filter_tensor(expr, t_string, strict=True)
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


def simplify(expr, real=False, *sym_tensors):
    """Simplify an expression by interchanging indices. Only indices that are
       already present in the input expression are used. The function just
       tries to map the indices in one term to the indices in another by
       comparing the occurences of the indices (the pattern) in each term.

       If real is set to True, the ERI 'V' and the fock matrix 'f' are assumed
       to be symmetric. Further additional tensors may be provided via
       sym_tensors (just their name string). Additionally all 'c' in all tensor
       names in the expression will be removed (this makes t-Amplitudes and
       possibly a c.c. ADC amplitude real). Finally, make real is called
       to fully simplify the expression by trying to swap the indices of all
       symmetric tensors (by default only 'V' and 'f').
       """
    from collections import Counter

    def descriptive_string(data):
        name = data["type"]
        if name == "tensor":
            len_up = len(data["upper"])
            len_lo = len(data["lower"])
            name = "_".join(
                [name, data["name"], str(len_up)+str(len_lo),
                 str(data["exponent"])]
            )
        elif name in ["delta", "annihilate", "create"]:
            name = "_".join([name, str(data["exponent"])])
        # nothing to do for prefactor
        return name

    def index_positions(obj, coupling="", *sym_tensors):
        # input: a single object that is part of the Mul object, e.g. t_ij^ab
        # return a dictionary with the positions of all indices of the obj
        # as descriptive string
        # coupling defines the coupling to other tensors in the same term or
        # assigns equivalent tensors a number to differentiate them
        # if necessary

        ret = {}
        data = __obj_data(obj)
        name = descriptive_string(data)
        if data["type"] == "prefactor":  # return empty dict
            pass
        # attach the position of the indices in the tensor to the descriptive
        # string of the tensor
        elif data["type"] == "tensor":
            for uplo in ["upper", "lower"]:
                for s in data[uplo]:
                    if s not in ret:
                        ret[s] = []
                    # upper = lower -> no need to differentiate for sym tensor
                    if data["name"] in sym_tensors:
                        pos = "_".join([name, "ul"])
                    else:
                        pos = "_".join([name, uplo[0]])
                    if coupling:
                        pos += "_" + coupling
                    ret[s].extend([pos for i in range(data["exponent"])])
        # because the name differs for Pow obj, it should be fine to just add
        # the coupling here for delta/create/annihilate
        else:
            for i in obj.free_symbols:
                ret[i] = [name + coupling]
        return ret

    def obj_coupling(obj1, obj2, *sym_tensors):
        # check whether obj1 couples to obj2, i.e. do they share indices?
        idx_pos1 = index_positions(obj1, "", *sym_tensors)
        idx_pos2 = index_positions(obj2, "", *sym_tensors)
        coupling = []
        for s1 in idx_pos1.keys():
            for s2, pos2 in idx_pos2.items():
                if s1 != s2:
                    continue
                # if indices for the two objects are equal
                coupling.extend(pos2)
        return coupling

    def term_coupling(term, *sym_tensors):
        # returns {n: [coupling]}
        # determines the relevant couplings of objects in a term
        # e.g. X_ia X_jb t^ak_ic t^bc_jk
        # first X couples to t, second X to t, first t to
        # second t and X, and second t to first t and X
        # this term would not include any relevant coupling, because the
        # coupling of both X and t is identical
        # e.g. X_ia X_jb t^ac_ik t^cd_kl
        # first X couples to t, second X to nothing, first t to X and
        # second t, second t to first t
        # here the coupling of the two X is not identical and therefore
        # relevant (the two X are not identical). The same is true for the t's.

        if not isinstance(term, Mul):
            # in case a non mul obj slips in... but should not be possible
            return {}
        # determine the coupling of all objects in the Mul term
        coupling = {}  # {name: {n: [position_in_other_object]}}
        for i, t in enumerate(term.args):
            for other_i, other_t in enumerate(term.args):
                if i == other_i:
                    continue
                c = obj_coupling(t, other_t, *sym_tensors)
                if not c:  # the obj are not coupled
                    continue
                data = __obj_data(t)
                name = descriptive_string(data)
                if name not in coupling:
                    coupling[name] = {}
                if i not in coupling[name]:
                    coupling[name][i] = []
                coupling[name][i].extend(c)
        # check whether the collected coupling is relevant. Not relevant when:
        # - the tensor only occurs once
        # - the tensor occurs more than once, but coupling occurs always to the
        #   same other tensor in the same position
        # If the coupling is not relevant for, because its equal for all
        # occurences of the tensor, the tensors are labeled by a number to
        # differentiate them from another.
        ret = {}  # {n: [coupling]}
        for t, coupl in coupling.items():
            # only one object of the same type -> no need to check coupling
            if len(coupl.keys()) == 1:
                continue
            # iterate over obj index (position in term) and compare the
            # coupling to the coupling of other identical tensors
            equal_couplings = []
            matched = []
            for i, c in coupl.items():
                if i in matched:
                    continue
                matched.append(i)
                temp = [i]
                count_c = Counter(c)
                for other_i, other_c in coupl.items():
                    if other_i in matched:  # self and double counting
                        continue
                    if count_c == Counter(other_c):  # coupling is identical
                        temp.append(other_i)
                        matched.append(other_i)
                equal_couplings.append(temp)
            # iterate over the equal couplings (e.g. of in total 4 'X' tensors,
            # pairs of two share the same coupling: c1 and c2) Attach a counter
            # to each coupling to differentiate identical tensors with
            # identical coupling from each other, i.e. objects with identical
            # coupling are numbered (f*f -> f1*f2)
            for equal_coupl in equal_couplings:
                temp = {}
                for n, i in enumerate(equal_coupl):
                    coupl[i].append(str(n+1))
                    temp[i] = coupl[i]
                ret.update(temp)
        # terms without coupling don't need to be treated seperately,
        # because in this case all indices should be target indices
        # that will be not touched anyways.
        return ret

    def term_pattern(term, *sym_tensors):
        # determine the pattern of all indices in a term.
        # only return the pattern for indices that are contracted according
        # to the einstein sum convention
        # returns {ov: [(name, occurences),]}

        # indices and their occurences are first collected in the indices dict
        indices = {}  # {idx: [occurences]}
        if isinstance(term, Mul):
            # the coupling modifies the tensor position strings
            # determined by index positions
            # therefore, if we have e.g. two t in the term their names will
            # be different (if their coupling is different) and it will make
            # a difference whether an index occurs at first or second t
            coupling = term_coupling(term, *sym_tensors)
            for i, t in enumerate(term.args):
                c = ""
                if i in coupling:
                    c = "_".join(sorted(coupling[i]))
                positions = index_positions(t, c, *sym_tensors)
                for s, occurence in positions.items():
                    if s not in indices:
                        indices[s] = []
                    indices[s].extend(occurence)
        else:  # term consists only of a single obj x (like tensor or Pow)
            positions = index_positions(term, "", *sym_tensors)
            for s, occurence in positions.items():
                if s not in indices:
                    indices[s] = []
                indices[s].extend(occurence)
        # now the indices are splitted in occ/virt indices and a tuple
        # (idx_name, occurences) is stored in the respective list
        ret = {"occ": [], "virt": []}
        for s, occurence in indices.items():
            # filter target indices here already and only return contracted idx
            if len(occurence) > 1:
                ret[index_space(s.name)].append((s, *occurence))
        return ret

    start = time.time()
    for t in sym_tensors:
        if not isinstance(t, str):
            raise Inputerror(f"Tensor string {t} must be of type str, "
                             f"not {type(t)}.")

    expr = expr.expand()
    # cant collect any terms... can only make real if requested
    if not isinstance(expr, Add):
        if real:
            expr = make_real(expr, *sym_tensors)
        return expr

    # if real is given all tensors need to be real, i.e.
    # - replace txcc -> tx and possibly X/Ycc -> X/Y
    # - tensors that are symmetric in this case (by default f and V) are added
    #   to sym_tensors. Other tensors need to be specified via sym_tensors
    if real:
        # NOTE: this should be resolved by introducing the numbering in
        #       term coupling (attach a number to the descriptive string of
        #       objects with equal coupling)
        # first try simplify without real to catch some more terms
        # (function does not work with real for e.g.
        # + Y^ac_ij Ycc^ab_ij d^b_c - Y^bc_ij Ycc^ab_ij d^a_c
        # both Y and Ycc are equal for symmetric d
        # result:
        # + Y^ab_ij Y^ac_ij d^b_c - Y^ab_ij Y^bc_ij d^a_c
        # No reason to apply P_ab, because both Y are just connected to d
        # instead of: y1 to d_upper and y2 to d_lower)
        # other_expr = simplify(expr, real=False)
        # if not isinstance(other_expr, Add):
        #     return other_expr
        # elif len(other_expr.args) < len(expr.args):
        #     expr = other_expr
        sym_tensors += ("f", "V")
        expr = make_tensor_names_real(expr)

    # collect index name, type and occurence in each term
    pattern = {}  # {#_term: {length: l, pattern: {ov: [(name, occurence)]}}}
    for i, term in enumerate(expr.args):
        pattern[i] = {}
        # number of tensors deltas etc. in the term, including prefactors!
        pattern[i]["length"] = len(term.args) if isinstance(term, Mul) else 1
        pattern[i]["pattern"] = term_pattern(term, *sym_tensors)

    # iterate over all terms and compare the occurences (the pattern)
    equal_terms = {}
    matched_terms = []  # avoid double counting
    for term_n, data in pattern.items():
        matched_terms.append(term_n)
        for other_term_n, other_data in pattern.items():
            # skip term_n and terms that have been matched already
            if other_term_n in matched_terms:
                continue
            # the length of the terms may only differ by 1 (by a prefactor),
            # the terms need to share the same amount of o/v indices
            if not abs(data["length"] - other_data["length"]) <= 1 or \
                    len(data["pattern"]["occ"]) != \
                    len(other_data["pattern"]["occ"]) or \
                    len(data["pattern"]["virt"]) != \
                    len(other_data["pattern"]["virt"]):
                continue
            matched_ov = []
            sub = []  # collect tuples (old, new) of indices if old != new
            for ov in ["occ", "virt"]:
                if not all(matched_ov):
                    continue  # occ already did not match -> skip virt
                p1 = data["pattern"][ov]  # [(name, occurences),]
                p2 = other_data["pattern"][ov]
                equal_idx = []  # (t2_idx, t1_idx) if patterns are equal
                # double counting: can not map 1 index in term1 to multiple
                #                  indices in term2
                matched_idx = []
                # check which indices in both terms are equal
                # if all occurences are equal the indices are equal
                # in this case possibly idx2 will be renamed to idx1
                for idx1 in p1:
                    count1 = Counter(idx1[1:])
                    for idx2 in p2:
                        if idx2[0] in matched_idx:
                            continue
                        # if pattern are identical -> found a match for idx1 in
                        # other_term -> break and continue with next idx1
                        if count1 == Counter(idx2[1:]):
                            matched_idx.append(idx2[0])
                            equal_idx.append((idx2[0], idx1[0]))
                            break
                # found a match for all indices
                if len(equal_idx) == len(p1):
                    matched_ov.append(True)
                    sub_idx = []
                    # only need to do sth if indices are not equal
                    for pair in equal_idx:
                        if pair[0] != pair[1]:
                            sub_idx.append(pair)
                    sub.extend(sub_idx)
                else:  # terms do not match
                    matched_ov.append(False)
            # term and other_term are equal if all indices are matched for o/v.
            # But only when there is something to do, i.e. some indices need to
            # be in sub (possible to be empty if a symmetric tensor is defined,
            # but indices are already equal and only need to be swapt
            # -> make real will fix this //
            # alternatively the indices may match however the substitution
            # only includes target indices that are not contracted
            if matched_ov and all(matched_ov):
                # if sub is empty there is nothing to do in this function
                # but they are matched. Make real should catch
                # the terms
                if sub:
                    if term_n not in equal_terms:
                        equal_terms[term_n] = {}
                    equal_terms[term_n][other_term_n] = sub
                matched_terms.append(other_term_n)

    # substitute the indices in other_term_n and keep term_n as is
    ret = 0
    matched_terms = []  # to construct the unchanged remainder
    for term_n in equal_terms:
        # keep term_n as is
        matched_terms.append(term_n)
        ret += expr.args[term_n]
        for other_term_n in equal_terms[term_n]:
            matched_terms.append(other_term_n)
            # substitute indices in other_term
            sub = equal_terms[term_n][other_term_n]
            ret += expr.args[other_term_n].subs(sub, simultaneous=True)
    # Add the remaining terms that were not matched
    ret += Add(*[expr.args[i] for i in range(len(expr.args))
                 if i not in matched_terms])
    print(f"old simplify took {time.time() - start} seconds.")
    # try swapping indices of symmetric tensors to fully simplify the expr
    if real:
        ret = make_real(ret, *sym_tensors)
    return ret


def make_canonical(expr, only_block_diagonal=False):
    """Diagonalize the Fock matrix by replacing elements
       f_pq with delta_pq * f_pq. The deltas are evaluated manually to
       avoid loss of information in the expression (loosing a target index).
       However, this implies that if evaluate deltas is called afterwards,
       the target index is lost anyway. (only important for f_pq with p and q
       target indices). Probably do all other manipulations before using this
       function.
       """

    def replace_f(term, only_block_diagonal):
        # 1) determine which indices are contracted in the term
        idx = __term_contracted_indices(term)
        # 2) replace f_ij -> delta_ij * f_ii
        ret = 1
        deltas = []
        if isinstance(term, Mul):
            for t in term.args:
                data = __obj_data(t)
                if data["type"] == "tensor" and data["name"] == "f":
                    delta = KroneckerDelta(data["upper"][0], data["lower"][0])
                    # delta_ov = 0 -> term is zero (off diagonal fock block)
                    if delta is S.Zero:
                        return S.Zero
                    if only_block_diagonal:
                        ret *= t
                    else:
                        ret *= delta * t
                        if delta is not S.One:
                            deltas.append(delta)
                else:
                    ret *= t
        else:
            data = __obj_data(term)
            if data["type"] == "tensor" and data["name"] == "f":
                delta = KroneckerDelta(data["upper"][0], data["lower"][0])
                if delta is S.Zero:
                    return S.Zero
                if only_block_diagonal:
                    ret *= term
                else:
                    ret *= delta * term
                    if delta is not S.One:
                        deltas.append(delta)
            else:
                ret *= term

        # manually evalute the deltas. In case of more than one delta
        # (fock element) per term: do it recursively
        for delta in deltas:
            # killable is contracted -> kill him
            if idx[delta.killable_index]:
                ret = ret.subs(delta.killable_index, delta.preferred_index)
                if len(deltas) > 1:
                    return replace_f(ret)
            # only preferred is contracted -> kill him instead
            elif idx[delta.preferred_index]:
                ret = ret.subs(delta.preferred_index, delta.killable_index)
                if len(deltas) > 1:
                    return replace_f(ret)
            # None is contracted -> kill None and keep delta
            else:
                continue
        return ret

    expr = expr.expand()
    # get all terms that contain f
    fock_terms = filter_tensor(expr, "f", strict=False)
    # terms without fock
    remaining = expr - fock_terms

    ret = remaining
    if isinstance(fock_terms, Add):
        for term in fock_terms.args:
            ret += replace_f(term, only_block_diagonal)
    else:
        ret += replace_f(fock_terms, only_block_diagonal)
    return ret
