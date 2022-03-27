from sympy import KroneckerDelta, Add, S, Mul, latex, Dummy
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


def sort_by_type_tensor(expr, t_string, symmetric=False):
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
                if symmetric:
                    temp.append("".join(sorted(ov_up + ov_lo)))
                else:
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
       tensors. Note that this function does not check how often a tensor
       occurs in the terms, i.e. if filter for 'f' terms with multiple 'f'
       are returned too. Note that in this case also terms with e.g. V*f,
       V*f*f etc. are included.
       The function also filters for the complex conjugate
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
                if isinstance(s, Dummy):
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


def make_tensor_names_real(expr):
    """Renames all tensors by removing any 'c' in their name.
       This essentially makes makes all t-amplitudes real,
       but only works if 'c' is not used in any other context than
       defining the complex conjugate.
       """

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
       Multiple occurences of the same tensor in a term are not
       supported atm.
       """
    from itertools import combinations
    # TODO: support multiple occurences the same tensor in a term

    def differ_by_repeated_tensor(comb1, comb2, n_sym_tensors):
        # check wheter comb1 and comb2 differ only by a tensor that can
        # occure multiple times in a term
        for t, n in n_sym_tensors.items():
            if n == 1:  # tensor occurs only once
                continue

    def interchange_upper_lower(term, t_string):
        # interchanges the upper and lower indices of a single
        # tensor in a single term

        ret = 1
        if isinstance(term, Mul):
            for t in term.args:
                if isinstance(t, AntiSymmetricTensor) and \
                        t.symbol.name == t_string:
                    ret *= AntiSymmetricTensor(t_string, t.lower, t.upper)
                else:
                    ret *= t
        else:
            if isinstance(term, AntiSymmetricTensor) and \
                    term.symbol.name == t_string:
                ret *= AntiSymmetricTensor(t_string, term.lower, term.upper)
            else:
                ret *= term
        if term-ret is S.Zero:
            raise RuntimeError("Probably something went wrong during filtering"
                               f". Trying swap upper and lower {t_string} "
                               f"indices in {latex(expr)} but the result "
                               f"{latex(ret)} is identical to the input.")
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
                        new = t_terms - term + interchanged
                        if not isinstance(new, Add):
                            return new
                        elif len(new.args) < len(t_terms.args):
                            return swap_tensors(new, *t_strings)
                        temp = interchanged
        return t_terms

    def prescan_terms(t_terms, *t_strings):
        # pre filter terms that contain all of the desired tensors
        # 1) collect all indices of the desired tensors in each term
        # 2) compare if indices match in different terms, i.e.
        #    if they may be equal when swapping
        #    upper and lower
        # returns a list of a list

        # collect all indices of the desired tensors in each term
        indices = []  # list(dict(set())) // [{t_name: {up, low}},]
        for term in t_terms.args:
            temp = {}
            if isinstance(term, Mul):
                for t in term.args:
                    if isinstance(t, AntiSymmetricTensor) and \
                            t.symbol.name in t_strings:
                        up = "".join(sorted([s.name for s in t.upper]))
                        lo = "".join(sorted([s.name for s in t.lower]))
                        temp[t.symbol.name] = {up, lo}
            else:
                if isinstance(term, AntiSymmetricTensor) and \
                        term.symbol.name in t_strings:
                    up = "".join(sorted([s.name for s in term.upper]))
                    lo = "".join(sorted([s.name for s in term.lower]))
                    temp[term.symbol.name] = {up, lo}
            if not all(tensor in temp.keys() for tensor in t_strings):
                raise RuntimeError("Something went wrong during filtering or "
                                   f"partitioning. Do not find all {t_strings}"
                                   f" in {term}. Found {list(temp.keys())}.")
            indices.append(temp)

        matching_terms = []
        # store all terms that are already matched -> avoid double counting
        matched = []
        # compare the indices, if the indices are identical independent of
        # their position (upper/lower) the terms may be simplified
        for i, term in enumerate(indices):
            temp = [i]
            if i in matched:
                continue
            matched.append(i)
            for j in range(i + 1, len(indices)):
                if j in matched:
                    continue
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
    for t in ["V", "f"]:  # add V and f to tensors if not provided
        if t not in sym_tensors:
            sym_tensors += (t,)

    expr = expr.expand()
    # 1) replace txcc with tx
    expr = make_tensor_names_real(expr)

    # 2) allow V_ij^ab = V_ab^ij / f_ij = f_ji
    #    and d_ij = d_ji for any tensor in sym_tensors

    # First partition the expr according to the sym_tensors included in the
    # term, e.g. terms with: only V / only f / V and f

    combs = [
        list(combinations(sym_tensors, r))
        for r in range(1, len(sym_tensors) + 1)
    ]

    t_terms = {}  # collect the results of filter_tensor
    for n_tensors in combs:
        for comb in n_tensors:
            if comb in t_terms:
                continue
            terms = filter_tensor(expr, *comb)
            if terms is not S.Zero:
                t_terms[comb] = terms
    only_t = {}  # {comb: terms} // comb = tuple of t_strings
    # iterate over combinations, starting with the full combination, e.g. V*f
    # for smaller combinations, all pure higher combinations need to be
    # subtracted, because e.g. filter(V) returns terms with V, but also V*f
    for comb in sorted(t_terms, key=len, reverse=True):
        if len(comb) == len(sym_tensors):  # full combination
            print(comb, latex(t_terms[comb]))
            only_t[comb] = t_terms[comb]
        else:  # smaller combination
            temp = t_terms[comb]
            # for example the tensors: a,b,c
            # combinations: abc, ab, ac, bc, a, b, c
            # for ab/ac/ab abc needs to be subtracted
            # for a the pure ab and ac terms need to be subtracted
            for only_comb, only_terms in only_t.items():
                # subtract the next higher hierarchy
                # - only if the lower hierarchy is in the higher (bc not
                #   relevant for a)
                if len(only_comb) - len(comb) == 1 and \
                        set(comb).issubset(only_comb):
                    temp -= only_terms
            only_t[comb] = temp
    # collect terms without any relevant tensors
    remaining = expr
    for terms in only_t.values():
        remaining -= terms

    # now try to simplify the expression by swapping the indices of
    # all relevant tensors in all terms.
    res = remaining
    for t_strings, terms in only_t.items():
        if not isinstance(terms, Add):
            res += terms
            continue
        # the position of terms that possibly may be simplified according
        # to the indices of the relevant tensors. Format: [[0,1],[3,4,6]]
        matching_terms = prescan_terms(terms, *t_strings)
        temp = 0
        swapped = []  # keep track of the terms for which a match was found
        for match in matching_terms:
            to_check = Add(*[terms.args[i] for i in match])
            swapped.extend(match)
            temp += swap_tensors(to_check, *t_strings)
        # Add terms for which no match was found
        temp += Add(*[
            terms.args[i] for i in range(len(terms.args)) if i not in swapped
        ])
        res += temp

    return res


def remove_tensor(expr, t_string):
    """Removes a tensor that is present in each term of the expression."""
    # TODO: probably also collect the indices of the removed tensor and return
    #       them too.

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


def simplify(expr, real=False, *sym_tensors):
    """Simplify an expression by interchanging indices. Only indices that are
       already present in the input expression are used. The function just
       tries to map the indices in 1 term to the indices in another by
       comparing the occurences of the indices the pattern in each term.

       If real is set to True, the ERI 'V' and the fock matrix 'f' are assumed
       to be symmetric. Further additional tensors may be provided via
       sym_tensors (just their name string). Additionally all 'c' in all tensor
       names in the expression will be removed (this makes t-Amplitudes and
       possibly a c.c. ADC amplitude real). Finally, make real is called
       to fully simplify the expression by trying to swap the indices of all
       symmetric tensors (by default only 'V' and 'f').
       """
    # this should already work for multiple occurences of tensors in a term

    # dict that contains name strings for all objects that may occur in the
    # expression (tensors are treated differently and precursors don't need a
    # name)
    naming = {
        KroneckerDelta: "delta",
        F: "annihilate",
        Fd: "create",
    }

    def obj_name(obj):
        # returns a string that describes the object
        # (whether its a delta/tensor/create/annihilate)
        name = None
        try:  # delta / create / annihilate
            name = naming[type(obj)]
        except KeyError:  # tensor
            # tensors are defined by their name and the number of upper/lower
            # symbols
            name = f"tensor_{obj.symbol.name}_{len(obj.upper)}{len(obj.lower)}"
        except AttributeError:  # prefactor
            if len(obj.free_symbols) != 0:
                raise KeyError(f"Do not have a name for an {type(obj)} object."
                               f" Known names: {list(naming.values())}.")
        return name

    def index_positions(obj, coupling="", *sym_tensors):
        # input: a single object that is part of the Mul object, e.g. t_ij^ab
        if len(obj.free_symbols) == 0:  # prefactor
            return {}
        pos = {}
        name = obj_name(obj)
        if name is None:
            raise RuntimeError(f"Something went wrong with naming. Could not "
                               "find a name, but also did not raise an error. "
                               f"Got None for an {type(obj)} object.")
        if isinstance(obj, AntiSymmetricTensor):
            # if tensor is symmetric f_ia = f_ai -> upper=lower
            sym = False
            if name.split("_")[1] in sym_tensors:
                sym = True
            symbols = [(obj.upper, "u"), (obj.lower, "l")]
            for uplo in symbols:
                for s in uplo[0]:
                    if s not in pos:
                        pos[s] = []
                    if sym:  # u/l are equal
                        p = name + "_ul"
                    else:
                        p = name + "_" + uplo[1]
                    if coupling:
                        p += "_" + coupling
                    pos[s].append(p)
        else:  # delta/create/annihilate
            for i in obj.free_symbols:
                pos[i] = [name + coupling]
        return pos

    def obj_coupling(obj1, obj2, *sym_tensors):
        # check whether obj1 couples to obj2, i.e. do they share indices?
        idx_pos1 = index_positions(obj1, "", *sym_tensors)
        idx_pos2 = index_positions(obj2, "", *sym_tensors)
        coupling = []
        for s1, pos1 in idx_pos1.items():
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
            return {}
        coupling = {}  # {name: {n: [position_in_other_object]}}
        for i, t in enumerate(term.args):
            for other_i, other_t in enumerate(term.args):
                if i == other_i:
                    continue
                c = obj_coupling(t, other_t, *sym_tensors)
                if not c:  # the obj are not coupled
                    continue
                name = obj_name(t)
                if name not in coupling:
                    coupling[name] = {}
                if i not in coupling[name]:
                    coupling[name][i] = []
                coupling[name][i].extend(c)
        ret = {}  # {n: [coupling]}
        # check whether the coupling is relevant. Not relevant when:
        # - the tensor only occurs once
        # - the tensor occurs more than once, but coupling occurs always to the
        #   same other tensor in the same position
        for t, coupl in coupling.items():
            # only one object of the same type -> no need to check coupling
            if len(coupl.keys()) == 1:
                continue
            elif len(coupl.keys()) < 2:  # just in case i messed up
                raise Inputerror("Something went wrong while defining the "
                                 f"coupling {coupling} for term {latex(term)}")
            equal = True
            checked = []
            # iterate over obj index (position in term) and compare the
            # coupling to the coupling of other identical tensors
            for i, c in coupl.items():
                if not equal:
                    break
                checked.append(i)
                for other_i, other_c in coupl.items():
                    if other_i in checked:  # avoid double counting
                        continue
                    if not all(name in other_c for name in c):
                        equal = False
                        break
            # coupling of all identical tensors is not equal -> relevant
            if not equal:
                ret.update(coupl)
        return ret

    def term_pattern(term, *sym_tensors):
        # get the pattern of all indices that are present in a term
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
        else:  # term consists only of a single obj x
            positions = index_positions(term, None, *sym_tensors)
            for s, occurence in positions.items():
                if s not in indices:
                    indices[s] = []
                indices[s].extend(occurence)
        # now the indices are splitted in occ/virt indices and a tuple
        # (idx_name, occurences) is stored in the respective list
        ret = {"occ": [], "virt": []}
        for s, occurence in indices.items():
            # filter target indices here already and only return contractes idx
            if len(occurence) > 1:
                ret[assign_index(s.name)].append((s, *occurence))
        return ret

    for t in sym_tensors:
        if not isinstance(t, str):
            raise Inputerror(f"Tensor string {t} must be of type str, "
                             f"not {type(t)}.")

    expr = expr.expand()
    if not isinstance(expr, Add):
        print("Can't simplify an expression that consists of a single term.")
        return expr

    # if real is given all tensors need to be real, i.e.
    # - replace txcc -> tx and possibly X/Ycc -> X/Y
    # - tensors that are symmetric in this case (by default f and V) are added
    #   to sym_tensors. Other tensors need to be specified via sym_tensors
    if real:
        # first try simplify without real to catch some more terms
        # (function does not work with real for e.g.
        # + Y^ac_ij Ycc^ab_ij d^b_c - Y^bc_ij Ycc^ab_ij d^a_c
        # both Y and Ycc are equal for symmetric d
        # result:
        # + Y^ab_ij Y^ac_ij d^b_c - Y^ab_ij Y^bc_ij d^a_c
        # No reason to apply P_ab, because both Y are just connected to d
        # instead of: y1 to d_upper and y2 to d_lower)
        other_expr = simplify(expr, real=False)
        if not isinstance(other_expr, Add):
            return other_expr
        elif len(other_expr.args) < len(expr.args):
            expr = other_expr
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
            # only includes target indices that are not contracted)
            if matched_ov and all(matched_ov) and sub:
                if term_n not in equal_terms:
                    equal_terms[term_n] = {}
                equal_terms[term_n][other_term_n] = sub
                matched_terms.append(other_term_n)

    # substitute the indices in other_term_n and keep term_n as is
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
    # Add the remaining terms that could have been matched
    ret += Add(*[expr.args[i] for i in range(len(expr.args))
                 if i not in matched_terms])
    if real:  # try swapping indices of symmetric tensors
        ret = make_real(ret, *sym_tensors)
    return ret
