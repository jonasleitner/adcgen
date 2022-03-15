from sympy import KroneckerDelta, Add, S, Mul, latex
from sympy.physics.secondquant import AntiSymmetricTensor
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


def make_real(expr):
    """Makes all tensors real, i.e. removes the cc in their names.
       Additionally, the function tries to simplify the expression
       by allowing V_ab^ij = V_ij^ab // <ij||ab> = <ab||ij>.
       """
    # 3) for symmetric tensors: d_ij = d_ji... but how to define symmetric
    #    tensors?
    #    maybe create another function where the user can specify symmetric
    #    tensors
    #    But e.g. the fock matrix is always symmetric
    # and how to deal with V and symmetric f simultaneously? would require:
    # switching first f -> check all ERI if simplify possible -> second f etc.

    expr = expr.expand()

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

    ret = 0
    # 1) replace txcc with tx
    if isinstance(expr, Add):
        for term in expr.args:
            ret += make_tensor_real(term)
    else:
        ret += make_tensor_real(expr)

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
                                   f"Trying swap upper and lower ERI indices "
                                   f"in {expr}, but do not find the ERI.")
        else:
            for t in term.args:
                if isinstance(t, AntiSymmetricTensor) and \
                        t.symbol.name == t_string:
                    ret *= AntiSymmetricTensor(t_string, t.lower, t.upper)
                else:
                    ret *= t
        return ret

    def swap_eri_braket(eri_terms):
        # iterate over all ERI terms. If it is possible to simplify
        # the expression by interchanging the ERI bra/ket, the function
        # will be called again with the simplified expression, until
        # no simplifications are possible anymore (or only a single
        # term is left).
        for term in eri_terms.args:
            interchanged = interchange_upper_lower(term, "V")
            new = eri_terms - term + interchanged
            if not isinstance(new, Add):  # only 1 term left -> finished
                return new
            elif len(new.args) < len(eri_terms.args):
                return swap_eri_braket(new)
        return eri_terms

    # 2) allow V_ij^ab = V_ab^ij
    eri_terms = filter_tensor(ret, "V")
    remaining = ret - eri_terms
    if isinstance(eri_terms, Add):
        eri_terms = swap_eri_braket(eri_terms)
        ret = remaining + eri_terms
    return ret


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
