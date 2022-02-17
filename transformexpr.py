from sympy import KroneckerDelta, Add
from sympy.physics.secondquant import AntiSymmetricTensor
from indices import assign_index


def sort_by_n_deltas(expr):
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
    if not isinstance(expr, Add):
        print(f"Can only sort an expression that is of type {Add}."
              f"Proovided eypression is of type {type(expr)}.")
        exit()
    deltas = {}
    for term in expr.args:
        temp = []
        for t in term.args:
            if isinstance(t, KroneckerDelta):
                ov1 = assign_index(t.preferred_index.name)
                ov2 = assign_index(t.killable_index.name)
                temp.append(ov1[0] + ov2[0])
        temp = tuple(sorted(temp))
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


def sort_by_type_fock(expr):
    if not isinstance(expr, Add):
        print(f"Can only sort an expression that is of type {Add}."
              f"Proovided eypression is of type {type(expr)}.")
        exit()
    fock = {}
    for term in expr.args:
        temp = []
        for t in term.args:
            if isinstance(t, AntiSymmetricTensor):
                if t.symbol == "f":
                    ov1 = assign_index(t.upper)
                    ov2 = assign_index(t.lower)
                    temp.append(ov1[0] + ov2[0])
        temp = tuple(sorted(temp))
        if not temp:
            temp = "no_fock"
        try:
            fock[temp].append(term)
        except KeyError:
            fock[temp] = []
            fock[temp].append(term)
    res = {}
    for f, term in fock.items():
        res[f] = Add(*[t for t in term])
    return res
