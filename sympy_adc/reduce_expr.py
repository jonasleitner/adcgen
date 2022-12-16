from . import expr_container as e
from .eri_orbenergy import eri_orbenergy
from .misc import Inputerror
from sympy import S
import time


def reduce_expr(expr):
    """Function that reduces the number of terms in an expression as much as
       possible by expanding all available intermediates and simplifying the
       resulting expression as much as possible by canceling orbital energy
       fractions."""
    from itertools import chain

    if not isinstance(expr, e.expr):
        raise Inputerror(f"Expr to reduce needs to be an instance of {e.expr}")

    # 1) Insert the canonical orbital basis (only diagonal Fock matrix elements
    #    remain)
    print("Inserting canonical orbital basis in expression of length "
          f"{len(expr)}... ", end='')
    start = time.perf_counter()
    expr = expr.diagonalize_fock()
    print(f"{len(expr)} canonical terms remaining. Took "
          f"{time.perf_counter()-start:.3f} seconds.")

    # check if we are already done
    if expr.sympy is S.Zero:
        return expr

    # 2) Insert the definitions of all defined intermediates in the expr
    print("Expanding intermediates... ", end='')
    start = time.perf_counter()
    expr = expr.expand_intermediates()
    expr = expr.expand()
    expr = expr.substitute_contracted()
    print(f"Expanded in expression of length {len(expr)}. Took "
          f"{time.perf_counter()-start:.3f} seconds.")

    # 3) Split the expression in a orbital energy fraction and a eri remainder.
    #    Compare the remainder pattern and try to find Permutations of
    #    contracted indices that allow to factor the eri remainder.
    print("Factoring ERI... ", end='')
    start = time.perf_counter()
    expr = factor_eri(expr)
    print(f"Found {len(expr)} different ERI structures. "
          f"Took {time.perf_counter()-start:.3f} seconds.")

    # 4) Factor the denominators and call factor on the resulting
    #    subexpressions, which should all contain exactly equal eri and
    #    denominators -> factor should create a term of length 1 with possibly
    #    multiple terms in the numerator
    print("Factoring denominators... ", end='')
    start = time.perf_counter()
    expr = chain.from_iterable((factor_denom(sub_expr) for sub_expr in expr))
    expr = [factored.factor() for factored in expr]
    if any(len(factored) != 1 for factored in expr):
        raise RuntimeError("Expected all subexpressions to be of length 1 "
                           "after factorization.")
    print(f"Found {len(expr)} different ERI/denominator combinations. "
          f"Took {time.perf_counter()-start:.3f} seconds.")

    if not expr:  # ensure that always a expr is returned
        return e.expr(0)

    # 5) permute the orbital energy numerator
    print("Permuting Numerators... ", end='')
    start = time.perf_counter()
    for i, term in enumerate(expr):
        term = eri_orbenergy(term).permute_num()
        expr[i] = term.expr
    print(f"Done. Took {time.perf_counter()-start} seconds.")

    # 6) Cancel the orbital energy fraction
    print("Cancel denominators... ", end='')
    start = time.perf_counter()
    reduced = e.compatible_int(0)
    for term in expr:
        term = eri_orbenergy(term)
        reduced += term.cancel_orb_energy_frac()
    print(f"{len(reduced)} terms remaining. "
          f"Took {time.perf_counter()-start} seconds.")
    return reduced


def factor_eri(expr: e.expr) -> list[e.expr]:
    """Factors the eri's of an expression."""
    from .simplify import find_compatible_terms

    if len(expr) == 1:  # trivial case
        return [expr]

    terms: list[eri_orbenergy] = [eri_orbenergy(term) for term in expr.terms]
    eris: list[e.term] = [term.eri for term in terms]

    # check for equal eri without permuting indices
    equal_eri: dict[int, list[int]] = {}
    matched: set = set()
    for i, eri in enumerate(eris):
        if i in matched:
            continue
        if i not in equal_eri:
            equal_eri[i] = []
        for other_i in range(i+1, len(eris)):
            if other_i in matched:
                continue
            if eri.sympy - eris[other_i].sympy is S.Zero:  # eri are equal
                equal_eri[i].append(other_i)
                matched.add(other_i)

    if len(equal_eri) == 1:  # trivial case: all eris are equal
        i, matches = next(iter(equal_eri.items()))
        ret = terms[i].expr
        for match in matches:
            ret += terms[match].expr
        return [ret]

    # try to match more eris by permuting contracted indices
    # only treat the unique eri structures (the keys in the equal eri dict)
    unique_eri_idx: list[int] = list(equal_eri.keys())
    unique_eri: list[e.term] = [eris[i] for i in unique_eri_idx]
    compatible_eri = find_compatible_terms(unique_eri)

    # add all terms up and substitute if necessary.
    ret: list[e.expr] = []
    for unique_i, compatible in compatible_eri.items():
        i = unique_eri_idx[unique_i]
        temp = terms[i].expr
        for matches in equal_eri[i]:  # add all terms with equal eri
            temp += terms[matches].expr
        del equal_eri[i]
        # apply the found sub dicts to make more eri identical
        for other_unique_i, sub in compatible.items():
            other_i = unique_eri_idx[other_unique_i]
            temp += terms[other_i].expr.subs(sub, simultaneous=True)
            for matches in equal_eri[other_i]:  # sub all terms with equal eri
                temp += terms[matches].expr.subs(sub, simultaneous=True)
            del equal_eri[other_i]
        ret.append(temp)
    # add up and append terms with eris that could not be mapped onto another
    # eris by applyig index permutations
    for i, equal in equal_eri.items():
        temp = terms[i].expr
        for matches in equal:
            temp += terms[matches].expr
        ret.append(temp)
    return ret


def factor_denom(expr: e.expr) -> list[e.expr]:
    """Factor the orbital energy denominators of an expr."""

    if len(expr) == 1:  # trivial case
        return [expr]

    terms: list[eri_orbenergy] = [eri_orbenergy(term) for term in expr.terms]

    # check which denoms are equal (should be exactly equal already)
    # denom signs should for doubles, singles etc equal in all terms, though
    # singles should have i-a, while doubles have a+b-i-j
    equal_denom: dict[int, list[int]] = {}
    matched: set = set()
    for i, term in enumerate(terms):
        if i in matched:
            continue
        if i not in equal_denom:
            equal_denom[i] = []
        for other_i in range(i+1, len(terms)):
            if other_i in matched:
                continue
            if term.denom == terms[other_i].denom:  # denoms are equal
                equal_denom[i].append(other_i)
                matched.add(other_i)

    if len(equal_denom) == 1:  # trivial case: all denoms are equal
        i, matches = next(iter(equal_denom.items()))
        ret = terms[i].expr
        for match in matches:
            ret += terms[match].expr
        return [ret]

    # try to find more equal denominators by applying permutations that satisfy
    # P_pq ERI = +- ERI AND NOT (!) P_pq Denom = +- Denom
    permutations = {}
    for unique_denom in equal_denom:
        term = terms[unique_denom]
        permutations[unique_denom] = [
            perms for perms, factor in
            term.denom_eri_sym(only_contracted=True).items() if factor is None
        ]

    ret: list[e.expr] = []
    matched: set = set()
    for unique_denom, matches in equal_denom.items():
        if unique_denom in matched:
            continue
        matched.add(unique_denom)
        term = terms[unique_denom]
        temp = term.expr
        # add all matches
        for match in matches:
            temp += terms[match].expr
        # iterate over other unique denoms and apply the corresponding perms
        for other_unique_denom, other_matches in equal_denom.items():
            if other_unique_denom in matched:
                continue
            other_term = terms[other_unique_denom]
            # check if there is a permutation that makes both denoms equal
            for perms in permutations[other_unique_denom]:
                if term.denom == other_term.denom.copy().permute(*perms):
                    matched.add(other_unique_denom)
                    # permute the whole term and also all matches
                    temp += other_term.expr.permute(*perms)
                    for match in other_matches:
                        temp += terms[match].expr.permute(*perms)
                    break
        ret.append(temp)
    return ret
