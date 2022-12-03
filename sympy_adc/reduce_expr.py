import sympy_adc.expr_container as e
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
    real = expr.real
    assumptions = expr.assumptions
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
    expr = factor_eri(expr, real)
    print(f"Found {len(expr)} different ERI structures. "
          f"Took {time.perf_counter()-start:.3f} seconds.")

    # 4) Factor the denominators and call factor on the resulting
    #    subexpressions, which should all contain exactly equal eri and
    #    denominators -> factor should create a term of length 1 with possibly
    #    multiple terms in the numerator
    print("Factoring denominators... ", end='')
    start = time.perf_counter()
    expr = [
        factored.factor() for factored in
        chain.from_iterable((factor_denom(sub_expr) for sub_expr in expr))
    ]
    if any(len(factored) != 1 for factored in expr):
        raise RuntimeError("Expected all subexpressions to be of length 1 "
                           "after factorization.")
    print(f"Found {len(expr)} different ERI/denominator combinations. "
          f"Took {time.perf_counter()-start:.3f} seconds.")

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
    reduced = e.expr(0, **assumptions)
    for term in expr:
        term = eri_orbenergy(term)
        reduced += term.cancel_orb_energy_frac()
    print(f"{len(reduced)} terms remaining. "
          f"Took {time.perf_counter()-start} seconds.")
    return reduced


def factor_eri(expr, real=False):
    from .simplify import find_compatible_terms

    terms = [eri_orbenergy(term) for term in expr.terms]
    eris = [term.eri for term in terms]

    # check for equal eri without permuting indices
    equal_eri = {}
    matched = set()
    for i, eri in enumerate(eris):
        if i in matched:
            continue
        matched.add(i)
        equal_eri[i] = []
        for other_i, other_eri in enumerate(eris):
            if other_i in matched:
                continue
            # if real is given, use make real to also cover <ab||ij> = <ij||ab>
            if eri.sympy - other_eri.sympy is S.Zero:
                matched.add(other_i)
                equal_eri[i].append(other_i)

    # try to match more eri by applying index permutations (only contracted
    # indices)
    # only treat the unique eri structures (the keys in the equal eri dict)
    # the keys should be automatically sorted, since the outer loop
    # inserts them one after another in ascending order
    unique_eri_idx = list(equal_eri.keys())
    unique_eri = [eris[i] for i in unique_eri_idx]
    matching_eri = find_compatible_terms(unique_eri)

    # add all terms up and substitute if necessary. Use exactly the same eri
    # for all terms, because they might still differ by bra/ket permutations.
    ret = []
    matched = set()
    for unique_i, compatible in matching_eri.items():
        i = unique_eri_idx[unique_i]
        matched.add(i)
        temp = terms[i].expr
        eri = eris[i]
        for other_unique_i, sub in compatible.items():
            other_i = unique_eri_idx[other_unique_i]
            matched.add(other_i)
            term = eri_orbenergy(
                terms[other_i].expr.subs(sub, simultaneous=True)
            )
            # double check that the current pattern is sufficient
            if not eri.sympy - term.eri.sympy is S.Zero:
                raise RuntimeError(f"The substitutions {sub} are not "
                                   f"sufficient to make {eris[other_i]} "
                                   f"equal to {eri}.")
            # use the eri of the parent term to ensure that all terms
            # have exactly the same eri
            temp += term.num * term.pref / term.denom * eri
            # also sub all other identical terms
            for other_matches in equal_eri[other_i]:
                term = eri_orbenergy(
                    terms[other_matches].expr.subs(sub, simultaneous=True)
                )
                temp += term.num * term.pref / term.denom * eri
        for matches in equal_eri[i]:
            term = terms[matches]
            temp += term.num * term.pref / term.denom * eri
        ret.append(temp)
    # add up and append terms for which no additional compatible terms could be
    # found through index substitution
    for i, equal in equal_eri.items():
        if i in matched:
            continue
        temp = terms[i].expr
        eri = eris[i]
        for other_i in equal:
            term = terms[other_i]
            temp += term.num * term.pref / term.denom * eri
        ret.append(temp)
    return ret


def factor_denom(expr):
    """Factor denominators."""
    splitted = [eri_orbenergy(term) for term in expr.terms]

    # check which denoms are equal (should be exactly equal already)
    # denom signs should for doubles, singles etc equal in all terms, though
    # singles should have i-a, while doubles have a+b-i-j
    equal = {}
    matched = set()
    for i, term in enumerate(splitted):
        if i in matched:
            continue
        matched.add(i)
        equal[i] = []
        for other_i, other_term in enumerate(splitted):
            if other_i in matched:
                continue
            if term.denom == other_term.denom:
                matched.add(other_i)
                equal[i].append(other_i)
    # now after finding all identical denoms -> try to find more by applying
    # permutations that satisfy
    # P_pq ERI = +- ERI AND NOT P_pq Denom = +- Denom
    permutations = {}
    for unique_denom in equal:
        term = splitted[unique_denom]
        permutations[unique_denom] = [
            perms for perms, factor in
            term.denom_eri_sym(only_contracted=True).items() if factor is None
        ]

    ret = []
    matched = set()
    for unique_denom, matches in equal.items():
        if unique_denom in matched:
            continue
        matched.add(unique_denom)
        term = splitted[unique_denom]
        equal_denom = term.expr
        # add all matches
        for match in matches:
            equal_denom += splitted[match].expr
        # iterate over other unique denoms and apply the corresponding perms
        for other_unique_denom, other_matches in equal.items():
            if other_unique_denom in matched:
                continue
            other_term = splitted[other_unique_denom]
            # check if there is a permutation that makes both denoms equal
            for perms in permutations[other_unique_denom]:
                if term.denom == other_term.denom.copy().permute(*perms):
                    matched.add(other_unique_denom)
                    # permute the whole term and also all matches
                    equal_denom += other_term.expr.permute(*perms)
                    for match in other_matches:
                        equal_denom += splitted[match].expr.permute(*perms)
                    break
        ret.append(equal_denom)
    return ret
