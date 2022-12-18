from . import expr_container as e
from .misc import Inputerror
from .eri_orbenergy import eri_orbenergy
from sympy import S
from collections import Counter, defaultdict


def factor_intermediates(expr, max_order: int = None):
    from .intermediates import intermediates
    from time import perf_counter

    if not isinstance(expr, e.expr):
        raise Inputerror("The expression to factor needs to be provided "
                         f"as {e.expr} instance.")
    expr = expr.expand()
    factored = []
    # later: 1. factor all t-amplitudes
    #        2. factor densities and other intermediates
    available = intermediates().available
    for name, itmd_instance in available.items():
        start = perf_counter()
        print(f"Factoring {name}... ", end="")
        expr = itmd_instance.factor_itmd(expr, factored, max_order)
        factored.append(name)
        print(f"Done. {len(expr)} terms remaining. Took ",
              perf_counter()-start)
        # for i, term in enumerate(expr.terms):
        #     term = eri_orbenergy(term)
        #     print(f"Term {i}:\n{term.pref = }\neri = {term.eri}\nnum = "
        #           f"{term.num}\ndenom = {term.denom}\n")
        # print("\n")
    expr = expr.substitute_contracted()
    return expr


def _factor_long_intermediate(expr: e.expr, itmd: list[eri_orbenergy],
                              itmd_data: list[dict],
                              itmd_term_map: dict,
                              itmd_instance) -> e.expr:
    """Function for factoring a long intermediate, i.e., a intermediate that
       consists of more than one term."""
    from sympy import Add

    if expr.sympy.is_number:
        return expr

    terms = expr.terms
    found_intermediates = {}
    for term_i, term in enumerate(terms):
        term = eri_orbenergy(term).canonicalize_sign()
        pattern = term.eri_pattern(include_exponent=False,
                                   target_idx_string=False)
        indices = [o.idx for o in term.eri.objects]
        term_data = {'pattern': pattern, 'indices': indices}
        for itmd_term_i, (itmd_term_data, itmd_term) in \
                enumerate(zip(itmd_data, itmd)):
            combinations = _compare_terms(term, itmd_term,
                                          term_data=term_data,
                                          itmd_term_data=itmd_term_data)
            # collect all remainders to avoid finding multiple identical
            # intermediates that only differ by sign
            found_remainders = defaultdict(list)
            for term_data in combinations:
                # extract the remainder that remains after factoring the itmd
                # excluding the prefactor that has to be added manually
                remainder: e.expr = _remainder(term, term_data['obj_i'],
                                               term_data['denom_i'])
                # determine the minimal target indices of the itmd to factor
                remainder, idx, factor = itmd_instance._minimal_itmd_indices(
                    remainder, term_data['sub']
                )
                remainder = eri_orbenergy(remainder)
                # check if the remainder has already been found
                # no need to try to build multiple identical (up to the sign)
                # intermediates
                idx_str = "".join([s.name for s in idx])
                if any(_compare_remainder(remainder, idx, ref) is not None
                       for ref in found_remainders[idx_str]):
                    continue
                else:
                    found_remainders[idx_str].append(remainder)
                factor *= term_data['factor']
                pref = _determine_prefactor(term, itmd_term, factor)
                # check if the current term can be mapped onto other terms
                # according to the symmetry of the remainder, i.e., if
                # (X - P_pq X) * remainder = 2X * remainder
                matching_itmd_terms, pref = _map_on_other_terms(
                    itmd_term_i, remainder, idx, pref,
                    itmd_term_map, itmd_instance
                    )
                if matching_itmd_terms is None:
                    continue
                # print(f"\nMAPPED {term} onto {itmd_term}\n")
                # print(f"{term_i = } {idx = }")
                # print(f"{matching_itmd_terms = } {pref = }")
                # print("remainder: ", remainder, "\n")
                data = {'idx': idx, 'term': term_i, 'remainder': remainder,
                        'pref': pref}
                found_intermediates = _assign_term_to_itmd(
                    found_intermediates, matching_itmd_terms, data, len(itmd)
                )
    factored_terms = set()
    factored_expr = e.expr(0, **expr.assumptions)
    factored_itmd = False
    for _, itmd_list in found_intermediates.items():
        for found_itmd in itmd_list:
            # did not find all terms of the itmd or some terms have already
            # been factored
            # TODO: allow to find itmd of only few terms are missing
            # print()
            # print(_, found_itmd.count(0))
            if found_itmd.count(0) != 0 or \
                    any(d["term"] in factored_terms for d in found_itmd):
                continue
            prefs = [d['pref'] for d in found_itmd]
            # print(prefs)
            # print(f"found {_} with prefs {prefs}")
            if not all(pref == prefs[0] for pref in prefs):
                continue
            most_common_pref = Counter(prefs).most_common(2)
            if most_common_pref[0][1] < len(itmd) // 2:
                continue
            if len(most_common_pref) != 1 and \
                    most_common_pref[0][1] == most_common_pref[1][1]:
                raise RuntimeError("Found multiple possible prefactors "
                                   f"{prefs = }.")
            most_common_pref = most_common_pref[0][0]
            for data in found_itmd:
                if data['pref'] != most_common_pref:
                    # diff = data['pref'] - most_common_pref
                    raise NotImplementedError()
            data = found_itmd[0]
            factored_expr += data['remainder'].expr * most_common_pref * \
                itmd_instance.tensor(indices=data['idx']).sympy
            factored_terms.update([d['term'] for d in found_itmd])
            factored_itmd = True
            # term_indices = [d['term'] for d in found_itmd]
            # print(f"Building intermediate {itmd_instance.name} with "
            #       f"prefactor {most_common_pref} and indices {_} "
            #       "from the terms:")
            # for i, term_i in enumerate(term_indices):
            #     print(f"{terms[term_i]}\ncorrepsponds to\n{itmd[i]}\n\n")
    factored_expr += Add(*(t.sympy for i, t in enumerate(terms)
                           if i not in factored_terms))
    # if we found the itmd and it is symmetric or antisymmetric
    #  -> add to the sym_tensors / antisym_tensors set
    if factored_itmd:
        tensor = itmd_instance.tensor().terms[0].objects[0]
        name = tensor.name
        if (bk_sym := tensor.bra_ket_sym) is S.One and \
                name not in (sym_tensors := factored_expr.sym_tensors):
            sym_tensors = sym_tensors + (name,)
            factored_expr.set_sym_tensors(sym_tensors)
        elif bk_sym is S.NegativeOne and \
                name not in (antisym_t := factored_expr.antisym_tensors):
            antisym_t = antisym_t + (name,)
            factored_expr.set_antisym_tensors(antisym_t)
    return factored_expr


def _factor_short_intermediate(expr: e.expr, itmd: eri_orbenergy,
                               itmd_data: dict, itmd_instance) -> e.expr:
    """Function for factoring a short intermediate, i.e., an intermediate that
       consists of a single term."""

    if expr.sympy.is_number:
        return expr

    terms = expr.terms
    factored_expr = e.expr(0, **expr.assumptions)
    factored_itmd = False
    for term in terms:
        term = eri_orbenergy(term).canonicalize_sign()
        # compare the term and the itmd term
        combs = _compare_terms(term, itmd, itmd_term_data=itmd_data)
        if not combs:
            factored_expr += term.expr
            continue
        # try to find a combination that does not intersect with any of
        # the others -> due to symmetry this makes not rly any sense
        for i, d1 in enumerate(combs):
            for j in range(i+1, len(combs)):
                if any(i in combs[j]['obj_i'] for i in d1['obj_i']):
                    break
            else:  # found a non intersecting combination
                term_data = d1
                break
        else:  # did not find a non intersecting comb -> just use the first one
            term_data = combs[0]
        obj_i = term_data['obj_i']
        denom_i = term_data['denom_i']
        # extract the remainder
        remainder: e.expr = _remainder(term, obj_i, denom_i)
        # determine the minimal target indices of the itmd to factor
        remainder, idx, factor = itmd_instance._minimal_itmd_indices(
            remainder, term_data['sub']
        )
        pref = _determine_prefactor(term, itmd, factor*term_data['factor'])
        # print(f"\nFacoring {term} to ", end='')
        term = remainder * pref * itmd_instance.tensor(indices=idx).sympy
        # print(term)
        factored_itmd = True
        # can we factor the itmd another time?
        if any(all(i not in obj_i for i in set(d['obj_i'])) for d in combs):
            term = _factor_short_intermediate(term, itmd, itmd_data,
                                              itmd_instance).sympy
        factored_expr += term
    # if we found the itmd and it is symmetric or antisymmetric
    #  -> add to the sym_tensors / antisym_tensors set
    if factored_itmd:
        tensor = itmd_instance.tensor().terms[0].objects[0]
        name = tensor.name
        if (bk_sym := tensor.bra_ket_sym) is S.One and \
                name not in (sym_tensors := factored_expr.sym_tensors):
            sym_tensors = sym_tensors + (name,)
            factored_expr.set_sym_tensors(sym_tensors)
        elif bk_sym is S.NegativeOne and \
                name not in (antisym_t := factored_expr.antisym_tensors):
            antisym_t = antisym_t + (name,)
            factored_expr.set_antisym_tensors(antisym_t)
    return factored_expr


def _remainder(term: eri_orbenergy, obj_i: list[int], denom_i: list[int]) -> e.expr:  # noqa E501
    """Returns the remainding part of the provided term that survives the
       factorization of the itmd, excluding the prefactor!"""
    eri = term.cancel_eri_objects(obj_i)
    denom = term.cancel_denom_brakets(denom_i)
    rem = term.num * eri / denom
    # explicitly set the target indices, because the remainder not necessarily
    # has to contain all of them.
    rem.set_target_idx(term.expr.terms[0].target)
    return rem


def _determine_prefactor(term: eri_orbenergy, itmd_term: eri_orbenergy,
                         factor=None):
    """Determines the prefactor of the resulting term after the itmd has been
       factored."""
    if factor is None:
        factor = 1
    # Because the symmetry of the itmd objects is now already taken into
    # account during object comparison, it is not necessary to adjust the
    # prefactor of the itmd here!
    return term.pref * factor / itmd_term.pref


def _map_on_other_terms(itmd_term_i: int, remainder: e.expr, idx: list, pref,
                        itmd_term_map: dict, itmd_instance):
    """Checks on which other terms the current term can be mapped if
       taking the symmetry of the remainder into account. A set of all
       terms, the current term contributes to is returned."""
    from sympy import Rational
    from .indices import get_symbols

    # 1) determine the symmetry of the remainder, only considering
    #    itmd indices that are no target indices of the overall term
    #    -> only permutations of itmd target indices allowed
    rem_target = remainder.eri.target
    idx_to_permute = {s for s in idx if s not in rem_target}
    rem_expr = remainder.expr
    rem_expr.set_target_idx(idx_to_permute)
    rem_sym = rem_expr.terms[0].symmetry(only_target=True)
    # create a map to translate the permutations to the default indices and
    # the default permutations to the minimized permutations
    default_idx = get_symbols(itmd_instance.default_idx)
    minimal_to_default = {old: new for old, new in zip(idx, default_idx)
                          if old is not new}
    # now iterate over the permutations and see if the current term
    # can be mapped onto anothers using the given permutations.
    matching_itmd_terms: set[int] = {itmd_term_i}
    for perms, perm_factor in rem_sym.items():
        # translate the permutations
        perms = tuple(
            tuple(minimal_to_default.get(s, s) for s in perm) for perm in perms
        )
        if perms not in itmd_term_map:
            continue
        perm_map: dict[int, list[int]] = itmd_term_map[perms]
        if itmd_term_i not in perm_map:
            continue
        for other_term_i, term_factor in perm_map[itmd_term_i]:
            if perm_factor != term_factor:
                return None, pref
            matching_itmd_terms.add(other_term_i)
    pref = pref * Rational(1, len(matching_itmd_terms))
    return matching_itmd_terms, pref


def _assign_term_to_itmd(found_intermediates: dict,
                         matching_itmd_terms: set[int], data: dict,
                         itmd_length: int) -> dict:
    """Assign the term according to the provided data to an intermediate,
       i.e., either start constructing a new one or add to an already
       existing."""
    rem = data['remainder']
    idx = data['idx']
    term_i = data['term']
    # 1) tranform the itmd target indices to string
    indices = "".join([s.name for s in idx])
    if indices not in found_intermediates:
        found_intermediates[indices] = []  # create list to hold itmds
    # try to add the term to an existing intermediate
    for found_itmd in found_intermediates[indices]:
        # I guess each term in a itmd can only consist of a single term.
        # 1 term can be mapped onto multiple itmd terms, but there should
        # always only 1 term involved!
        # -> none of the terms the current term represents can be initialized
        #    already
        # also each term can only occur once in each itmd
        if any(found_itmd[i] for i in matching_itmd_terms) or \
                any(term_i == d['term'] for d in found_itmd if d):
            continue
        # check that all terms share a common remainder
        for d in found_itmd:
            if d:
                factor = _compare_remainder(remainder=rem, indices=idx,
                                            ref_remainder=d['remainder'])
                if factor is None:  # remainder are not identical
                    continue
                data['remainder'] = d['remainder']
                data['pref'] *= factor
                break
        else:
            continue
        # ok everything is fine -> add the term to all positions
        for itmd_term_i in matching_itmd_terms:
            found_itmd[itmd_term_i] = data
        break
    else:  # did not find a matching itmd -> create new one
        new_itmd = [data if i in matching_itmd_terms else 0
                    for i in range(itmd_length)]
        found_intermediates[indices].append(new_itmd)
    return found_intermediates


def _compare_obj(obj: e.obj, itmd_obj: e.obj, obj_coupl: list[str],
                 itmd_obj_coupl: list[str], obj_descr: str = None,
                 itmd_obj_descr: str = None, obj_idx: tuple = None,
                 itmd_obj_idx: tuple = None,
                 itmd_obj_sym: dict = None) -> list[tuple[dict, int]]:
    """Compare the two provided objects and return the substitutions as dict
        that are required to transform the itmd_obj."""
    from collections import Counter

    if obj_descr is None:
        obj_descr = obj.description(include_exponent=False)
    if itmd_obj_descr is None:
        itmd_obj_descr = itmd_obj.description(include_exponent=False)
    # check if the descriptions match
    if obj_descr != itmd_obj_descr:
        return []
    # check that the coupling of the itmd_obj is a subset of the coupling
    # of the obj
    if itmd_obj_coupl:
        obj_coupl = Counter(obj_coupl)
        if not all(count <= obj_coupl[c] if c in obj_coupl else False
                   for c, count in Counter(itmd_obj_coupl).items()):
            return []
    # obj can match -> create the substitution dict
    if obj_idx is None:
        obj_idx = obj.idx
    if itmd_obj_idx is None:
        itmd_obj_idx = itmd_obj.idx
    sub_list = [(dict(zip(itmd_obj_idx, obj_idx)), 1)]
    # account for the symmetry of the itmd_obj -> try all perms
    # bra_ket symmetry is automaticall accounted for!
    if itmd_obj_sym is None:
        itmd_obj_sym = itmd_obj.symmetry()
    for perms, factor in itmd_obj_sym.items():
        perm_idx = itmd_obj_idx
        for p, q in perms:  # permute the indices
            sub = {p: q, q: p}
            perm_idx = [sub.get(s, s) for s in perm_idx]
        sub_list.append((dict(zip(perm_idx, obj_idx)), factor))
    return sub_list


def _compare_eri_parts(term: eri_orbenergy, itmd_term: eri_orbenergy,
                       term_data: dict = None,
                       itmd_term_data: dict = None) -> list:
    """Compare the eri parts of two terms and return the substitutions
           that are necessary to transform the itmd_eri."""
    from sympy import Mul
    from itertools import product

    # the eri part of the term to factor has to be at least as long as the
    # eri part of the itmd (prefactors are separated!)
    if len(itmd_term.eri) > len(term.eri):
        return []
    objects = term.eri.objects
    # generate term_data if not provided
    if term_data is None:
        term_data = {}
    if (pattern := term_data.get('pattern')) is None:
        pattern = term.eri_pattern(include_exponent=False,
                                   target_idx_string=False)
    if (indices := term_data.get('indices')) is None:
        indices = [o.idx for o in objects]
    # generate itmd_data if not provided
    itmd_objects = itmd_term.eri.objects
    if itmd_term_data is None:
        itmd_term_data = {}
    if (itmd_pattern := itmd_term_data.get('itmd_pattern')) is None:
        itmd_pattern = itmd_term.eri_pattern(include_exponent=False,
                                             target_idx_string=False)
    if (itmd_indices := itmd_term_data.get('itmd_indices')) is None:
        itmd_indices = [o.idx for o in itmd_objects]
    if (itmd_obj_sym := itmd_term_data.get('itmd_obj_sym')) is None:
        itmd_obj_sym = [o.symmetry() for o in itmd_objects]

    # compare the objects in both terms and collect the necessary
    # substitutions to map the objects onto each other
    matches = {}
    for (obj_i, (descr, coupl)), obj in zip(pattern.items(), objects):
        kwargs = {'obj': obj, 'obj_idx': indices[obj_i], 'obj_descr': descr,
                  'obj_coupl': coupl}
        for (itmd_obj_i, (itmd_descr, itmd_coupl)), itmd_obj in \
                zip(itmd_pattern.items(), itmd_objects):
            itmd_idx = itmd_indices[itmd_obj_i]
            itmd_sym = itmd_obj_sym[itmd_obj_i]
            kwargs.update({'itmd_obj': itmd_obj, 'itmd_obj_idx': itmd_idx,
                           'itmd_obj_descr': itmd_descr,
                           'itmd_obj_coupl': itmd_coupl,
                           'itmd_obj_sym': itmd_sym})
            # compare the two objects
            if (sub_list := _compare_obj(**kwargs)):
                to_cancel = tuple(obj_i for _ in range(itmd_obj.exponent))
                if itmd_obj_i not in matches:
                    matches[itmd_obj_i] = {}
                matches[itmd_obj_i][to_cancel] = sub_list
    # did not find a match for all objects of the intermediate term
    if len(matches) != len(itmd_objects):
        return []

    # more than one combination might be possible -> need to find all
    # possible combinations
    matches = sorted(matches.items(), key=lambda tpl: tpl[0])
    combs = []
    for itmd_obj_i, sub_dict in matches:
        variants = [(list(obj_i_tpl), sub, factor) for obj_i_tpl, sub_list in
                    sub_dict.items() for sub, factor in sub_list]
        if not combs:
            combs.extend(variants)
            continue
        temp = []
        for comb, variant in product(combs, variants):
            idx_list, sub, factor = comb
            obj_i_list, additional_sub, additional_factor = variant
            # for instance, 0 / 1 -> 0,2 / 0,3 / 1,2 / 1,3
            if obj_i_list[0] not in idx_list and not any(
                    old in sub and sub[old] is not new
                    for old, new in additional_sub.items()
                    ):
                idx_list = idx_list + obj_i_list
                sub = sub | additional_sub
                factor = factor * additional_factor
                temp.append((idx_list, sub, factor))
        combs = temp

    # check which of the found combinations is valid
    valid = []
    itmd_coupl = sorted([sorted(c) for _, c in itmd_pattern.values()
                         if c is not None])
    for obj_indices, sub, factor in combs:
        # obj indices might occur multiple times!
        if len(set(obj_indices)) != len(itmd_objects):
            continue
        # we need to have the same coupling between the objects!
        relevant_obj = Mul(*(objects[i].sympy for i in set(obj_indices)))
        relevant_obj = e.expr(relevant_obj, **term.eri.assumptions).terms[0]
        coupl = relevant_obj.coupling(target_idx_string=False,
                                      include_exponent=False)
        if sorted([sorted(c) for c in coupl.values()]) != itmd_coupl:
            continue

        sub_itmd = itmd_term.eri.subs(sub, simultaneous=True)
        if sub_itmd.sympy.is_number:
            continue
        # remove the prefactor (possibly a -1 has been introduced due to the
        # substitutions). Only the prefactor arising from the permutations
        # (introduced above) is relevant.
        sub_itmd *= sub_itmd.terms[0].prefactor
        if relevant_obj.sympy - sub_itmd.sympy is S.Zero:
            valid.append((obj_indices, sub, sub_itmd.terms[0], factor))
    return valid


def _compare_terms(term: eri_orbenergy, itmd_term: eri_orbenergy,
                   term_data: dict = None,
                   itmd_term_data: dict = None) -> list: # noqa E501
    """Compare two terms and return a substitution dict that makes the
        itmd_term equal to the term. Also the indices of the objects in the
        eri part and the denominator that match the intermediate's objects
        are returned."""
    from itertools import chain

    # if the itmd term has a denom -> the term also needs to have one
    if not itmd_term.denom.sympy.is_number and term.denom.sympy.is_number:
        return []
    eri_combs = _compare_eri_parts(term, itmd_term, term_data=term_data,
                                   itmd_term_data=itmd_term_data)

    # was not possible to map the eri parts onto each other
    if not eri_combs:
        return []

    # if the itmd does not have a denom -> only need to map eri
    if itmd_term.denom.sympy.is_number:
        return [
            {'obj_i': obj_indices, 'sub': sub, 'denom_i': [], 'factor': factor,
             'sub_itmd_eri': sub_itmd_eri}
            for obj_indices, sub, sub_itmd_eri, factor in eri_combs
        ]

    # itmd_term and term have to have a denominator at this point
    valid_combs = []
    itmd_denom_brakets = itmd_term.denom_brakets
    for obj_indices, sub, sub_itmd_eri, factor in eri_combs:
        matched = {}
        for itmd_idx, itmd_braket in enumerate(itmd_denom_brakets):
            if isinstance(itmd_braket, e.expr):
                itmd_braket = itmd_braket.copy()
            sub_braket = itmd_braket.subs(sub, simultaneous=True)
            if (match := term.find_matching_braket(sub_braket)):
                matched[itmd_idx] = match
        # was not possible to match all brakets in the denominator
        if len(matched) != len(itmd_denom_brakets):
            continue
        # check that we uniquely mapped the brakets onto each other
        matched = list(chain.from_iterable(matched.values()))
        if len(set(matched)) != len(itmd_denom_brakets):
            continue
        valid_combs.append({'obj_i': obj_indices, 'sub': sub,
                            'denom_i': matched, 'factor': factor,
                            'sub_itmd_eri': sub_itmd_eri})
    return valid_combs


def _compare_remainder(remainder: eri_orbenergy, indices: list,
                       ref_remainder: eri_orbenergy) -> int | None:
    """Compare the two remainders."""

    # very quick check if the remainders are already equal
    if remainder.expr.sympy - ref_remainder.expr.sympy is S.Zero:
        return 1
    # compare the remainder (eri and denoms).
    # Try to map remainder on ref_remainder
    combs = _compare_terms(term=ref_remainder, itmd_term=remainder)
    mapped_with_sign_change = False
    for comb in combs:
        sub = {old: new for old, new in comb['sub'].items() if old is not new}
        if not sub:
            raise RuntimeError("Remainder and seem to be identical without "
                               "applyig any permutations. "
                               "This should have been catched earlier.")
        # add the itmd target indices to the target indices of the remainder
        target = list(remainder.eri.target) + list(indices)
        # if any of the surviving pairs includes any of the target indices
        # -> not a valid substitution dict
        if any((old in target or new in target) for old, new in sub.items()):
            continue
        # only valid substitutions in sub!
        # substitutions in the denominator can not introduce another -1
        sub_num = remainder.num.copy().subs(sub, simultaneous=True)
        sub_denom = remainder.denom.copy().subs(sub, simultaneous=True)
        sub_remainder = (
            comb['sub_itmd_eri'] * comb['factor'] * sub_num / sub_denom
        )
        # the remainder are identical
        if sub_remainder.sympy - ref_remainder.expr.sympy is S.Zero:
            return 1
        # alternatively the remainder can be identical up to a factor of -1
        # try to find an exact match, but also test for this
        # if they agree up to a factor of -1 the prefactor has to be adjusted!
        elif not mapped_with_sign_change and \
                sub_remainder.sympy + ref_remainder.expr.sympy is S.Zero:
            mapped_with_sign_change = True
    return -1 if mapped_with_sign_change else None
