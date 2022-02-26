from sympy import KroneckerDelta, Add, S, latex
from sympy.physics.secondquant import AntiSymmetricTensor
from indices import assign_index
from misc import Inputerror


def sort_by_n_deltas(expr):
    expr = expr.expand()
    if not isinstance(expr, Add):
        print(f"Can only sort an expression that is of type {Add}."
              f"Provided expression is of type {type(expr)}")
        exit()
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
        print(f"Can only sort an expression that is of type {Add}."
              f"Provided eypression is of type {type(expr)}.")
        exit()
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
                ov1 = assign_index(t.upper[0].name)
                ov2 = assign_index(t.lower[0].name)
                temp.append("".join(sorted(ov1[0] + ov2[0])))
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
       conjugate assuming that the cc. Tensor shares the same name but includes
       just one or two 'c' additionally."""

    expr = expr.expand()
    if not isinstance(expr, Add):
        print(f"Can only filter an expression that is of type {Add}."
              f" Provided expression is of type {type(expr)}.")
        exit()
    tensor = []
    for term in expr.args:
        for t in term.args:
            if isinstance(t, AntiSymmetricTensor) and \
                    t.symbol.name.replace('c', '') == t_string:
                tensor.append(term)
    return Add(*tensor)


def sort_tensor_sum_indices(expr, t_string):
    """Sorts an expression by sorting the terms depending on the
       number of indices of an AntiSymmetricTensor that are summed
       over. There is an additional splitting that depends on the
       kind of index (occ/virt).
       """

    expr = expr.expand()
    if not isinstance(expr, Add):
        print(f"Can only filter an expression that is of type {Add}."
              f" Provided expression is of type {type(expr)}.")
        exit()
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
