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


def filter_tensor(expr, t_string):
    """Returns all terms of an expression that contain an AntiSymmetriTensor
       with a certain name. Also returns terms that contain the comlpex
       conjugate assuming that the c.c. Tensor shares the same name but
       includes just additional 'c' (possible multiple).
       """

    expr = expr.expand()
    if not isinstance(expr, Add):
        raise Inputerror("Can only filter an expression that is of "
                         f"type {Add}. Provided: {type(expr)}")
    tensor = []
    for term in expr.args:
        # if the term only only consists of a single object the loop
        # below does not work
        if not isinstance(term, Mul):
            try:
                if term.symbol.name.replace('c', '') == t_string:
                    tensor.append(term)
            except AttributeError:
                continue
        for t in term.args:
            if isinstance(t, AntiSymmetricTensor) and \
                    t.symbol.name.replace('c', '') == t_string:
                tensor.append(term)
    return Add(*tensor)


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
