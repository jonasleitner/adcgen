from . import expr_container as e
from .misc import Inputerror, cached_property
from .eri_orbenergy import EriOrbenergy
from .indices import order_substitutions, get_symbols, minimize_tensor_indices
from .sympy_objects import AntiSymmetricTensor
from .symmetry import LazyTermMap
from sympy import S, Mul, Rational
from collections import Counter
from itertools import product, compress


def factor_intermediates(expr, types_or_names: str | list[str] = None,
                         max_order: int = None) -> e.Expr:
    from .intermediates import Intermediates
    from time import perf_counter

    if not isinstance(expr, e.Expr):
        raise Inputerror("The expression to factor needs to be provided "
                         f"as {e.Expr} instance.")

    if expr.sympy.is_number:  # nothing to factor
        return expr

    # get all intermediates that are about to be factored in the expr
    itmd = Intermediates()
    if types_or_names is not None:
        if isinstance(types_or_names, str):
            itmd_to_factor = getattr(itmd, types_or_names)
        else:  # list / tuple / set of strings
            itmd_to_factor = {}
            for t_or_n in types_or_names:
                if not isinstance(t_or_n, str):
                    raise TypeError("Intermediate types/names to factor have "
                                    "to be provided as str or list of strings."
                                    f"Got {t_or_n} of type {type(t_or_n)}.")
                itmd_to_factor |= getattr(itmd, t_or_n)
    else:
        itmd_to_factor: dict = itmd.available

    print('\n\n', '#'*80, '\n', " "*25, "INTERMEDIATE FACTORIZATION\n", '#'*80,
          "\n", sep='')
    print(f"Trying to factor intermediates in expr of length {len(expr)}\n")
    for i, term in enumerate(expr.terms):
        print(f"{i+1}:  {EriOrbenergy(term)}\n")
    print('#'*80)
    # try to factor all requested intermediates
    factored = []
    for name, itmd_cls in itmd_to_factor.items():
        print("\n", ' '*25, f"Factoring {name}\n\n", '#'*80, sep='',
              flush=True)
        start = perf_counter()
        expr = itmd_cls.factor_itmd(expr, factored, max_order)
        factored.append(name)
        print('\n', '-'*80, sep='')
        print(f"Done in {perf_counter()-start:.2f}s. {len(expr)} terms remain")
        print('-'*80, '\n')
        for i, term in enumerate(expr.terms):
            print(f"{i+1: >{len(str(len(expr)+1))}}:  {EriOrbenergy(term)}\n")
        print('#'*80)
    print("\n\n", '#'*80, "\n", " "*25,
          "INTERMEDIATE FACTORIZATION FINISHED\n", '#'*80, sep='', flush=True)
    # make the result pretty by minimizing contracted indices:
    # some contracted indices might be hidden inside some intermediates.
    # -> ensure that the remaining ones are the lowest available
    expr = expr.substitute_contracted()
    print(f"\n{len(expr)} terms in the final result:")
    width = len(str(len(expr)+1))
    for i, term in enumerate(expr.terms):
        print(f"{i+1: >{width}}: {EriOrbenergy(term)}")
    return expr


def _factor_long_intermediate(expr: e.Expr, itmd: list[EriOrbenergy],
                              itmd_data: tuple, itmd_term_map: LazyTermMap,
                              itmd_cls) -> e.Expr:
    """Function for factoring a long intermediate, i.e., a intermediate that
       consists of more than one term."""

    if expr.sympy.is_number:
        return expr

    # does any itmd term has a denominator?
    itmd_has_denom = any(term_data.denom_bracket_lengths is not None
                         for term_data in itmd_data)
    itmd_length = len(itmd)
    # get the default symbols of the intermediate
    itmd_default_symbols = tuple(get_symbols(itmd_cls.default_idx))

    terms: list[EriOrbenergy] = list(expr.terms)

    # class that manages the found itmd variants
    intermediate_variants = LongItmdVariants(itmd_length)
    for term_i, term in enumerate(terms):
        term = EriOrbenergy(term).canonicalize_sign()
        # prescan: check that the term holds the correct tensors and
        #          denominator brackets
        term_data = FactorizationTermData(term)
        # description of all objects in the eri part, exponent implicitly
        # included
        obj_descr = term_data.eri_obj_descriptions
        if itmd_has_denom:
            bracket_lengths = term_data.denom_bracket_lengths

        # compare to all of the itmd terms -> only try to map on a subset of
        # intermediate terms later
        possible_matches = []
        for itmd_i, itmd_term_data in enumerate(itmd_data):
            # do all tensors in the eri part occur at least as often as
            # in the intermediate
            if any(obj_descr[descr] < n for descr, n in
                   itmd_term_data.eri_obj_descriptions.items()):
                continue
            # itmd_term has a denominator?
            itmd_bracket_lengths = itmd_term_data.denom_bracket_lengths
            if itmd_bracket_lengths is not None:
                if bracket_lengths is None:  # term has no denom -> cant match
                    continue
                else:  # term also has a denominator
                    # ensure that bracket of the correct length are available
                    if any(bracket_lengths[length] < n for length, n in
                           itmd_bracket_lengths.items()):
                        continue
            possible_matches.append(itmd_i)
        if not possible_matches:  # did not find any possible matches
            continue

        # extract the target idx names of the term
        target_idx_by_space = {}
        for s in term.eri.target:
            if (sp := s.space) not in target_idx_by_space:
                target_idx_by_space[sp] = set()
            target_idx_by_space[sp].add(s.name)

        # go through all possible matches
        for itmd_i in possible_matches:
            # - compare and obtain data (sub_dict, obj indices, factor)
            #   that makes the itmd_term equal to the defined sub part
            #   of the term.
            variants = _compare_terms(term, itmd[itmd_i], term_data=term_data,
                                      itmd_term_data=itmd_data[itmd_i])
            if variants is None:  # was not possible to map the terms
                continue

            # The term_map allows to spread a term assignement to multiple
            # terms taking the symmetry of the remainder into account, e.g.,
            # for the t2_2 amplitudes:
            #    t2_2 <- (1-P_ij)(1-P_ab) X
            #  - Depending on the symmetry of the remainder these 4 terms
            #    might occur as 4, 2 or 1 terms in the expression to factor:
            #     Rem * (1-P_ij)(1-P_ab) X -> 4 * Rem * X
            #    (if Rem has ij and ab antisymmetry)
            #  - If a term with such a remainder is matched with one of the 4
            #    4 terms he will automatically also be matched with the other
            #    3 terms using the term_map for the intermediate.
            # NOTE: it is not possible to exploit this to reduce the workload
            #       by exploiting the fact that the current term has already
            #       been matched to a itmd_term through the term map, because
            #       more complicated permutations do not provide a back and
            #       forth relation ship between terms:
            #         P_ij P_ik A(ijk) -> B(kij)
            #         P_ij P_ik B(kij) -> C(jki)
            #       comparing the current term to A can also provide a match
            #       with B through the term_map.
            #       comparing the current term to B however can provide a match
            #       with C!
            #       Therefore, the comparison with B can not be skipped, even
            #       if remainder and itmd_indices are identical to a previously
            #       found variant that matched to A and B!
            # What can be done: for matching term 1 -> itmd_term A
            # due to the symmetry of tensors one probably obtains multiple
            # variants for the same itmd_indices and remainder that only
            # differ in contracted indices.
            # -> for each itmd_indices only consider one variant for each
            #    remainder
            found_remainders = {}  # {itmd_indices: [remainder]}
            for variant_data in variants:  # go through all valid variants
                # - extract the remainder of the term (objects, excluding
                #   prefactors that will remain if the current variant is
                #   used to factor the itmd)

                remainder = _get_remainder(term, variant_data['eri_i'],
                                           variant_data['denom_i'])

                # - obtain the indices of the intermediate
                itmd_indices = tuple(variant_data['sub'].get(s, s) for s in
                                     itmd_default_symbols)

                # - minimize the indices of the intermediate to ensure that
                #   the same indices are used in each term of the long itmd
                #   (use the lowest non target indices)
                itmd_indices, minimization_perms = minimize_tensor_indices(
                    itmd_indices, target_idx_by_space
                )

                # - apply the substitutions to the remainder
                remainder = remainder.permute(*minimization_perms)
                # if this ever triggers probably switch to a continue
                assert remainder.sympy is not S.Zero

                # - Further minimize the tensor indices taking the tensor
                #   symmetry of the itmd into account by building a tensor
                #   using the minimized tensor indices
                #   -> returns the tensor with completely minimized indices
                #      and possibly a factor of -1
                tensor_obj = itmd_cls.tensor(indices=itmd_indices).terms[0]
                if len(tensor_obj) > 2:
                    raise ValueError("Expected the term to be at most of "
                                     f"length 2. Got: {tensor_obj}.")
                for obj in tensor_obj.objects:
                    if 'tensor' in (o_type := obj.type):
                        itmd_indices = obj.idx
                    elif o_type == 'prefactor':
                        variant_data['factor'] *= obj.sympy
                    else:
                        raise TypeError("Only expected tensor and prefactor."
                                        f"Found {obj} in {tensor_obj}")

                # ensure that there are no indices in the numerator or the
                # denominator of the remainder, that do not occur in the eri
                # part or the itmd indices. This avoids factoring an
                # itmd using invalid contracted itmd indices that are only
                # partially removed from the term, e.g., m - a contracted itmd
                # index occurs in both, denominator and eri part, but is only
                # removed in the eri part, because the itmd has no denominator.
                _validate_indices(remainder, itmd_indices)

                # check if we already found another variant that gives the
                # same itmd_indices and remainder (an identical result that
                # only differs in contracted itmd_indices)
                if itmd_indices not in found_remainders:
                    found_remainders[itmd_indices] = []
                if any(_compare_remainder(remainder, found_rem, itmd_indices)
                       is not None
                       for found_rem in found_remainders[itmd_indices]):
                    continue  # go to the next variant
                else:
                    found_remainders[itmd_indices].append(remainder)

                # - check if the current itmd_term can be mapped onto other
                #   itmd terms
                matching_itmd_terms = _map_on_other_terms(
                    itmd_i, remainder, itmd_term_map, itmd_indices,
                    itmd_default_symbols
                )

                # - calculate the final prefactor of the remainder if the
                #   current variant is applied for factorization
                #   keep the term normalized if spreading to multiple terms!
                #   (factor is +-1)
                prefactor = (term.pref * variant_data['factor'] *
                             Rational(1, len(matching_itmd_terms)) /
                             itmd[itmd_i].pref)

                # - compute the factor that the term should have if we want
                #   to factor the current variant with a prefactor of 1
                #   (required for factoring mixed prefactors)
                unit_factorization_pref = (
                    itmd[itmd_i].pref * variant_data['factor']
                    * len(matching_itmd_terms)
                )

                # - add the match to the pool where intermediate variants
                #   are build from
                intermediate_variants.add(
                    term_i=term_i, itmd_indices=itmd_indices,
                    matching_itmd_terms=matching_itmd_terms,
                    remainder=remainder, prefactor=prefactor,
                    unit_factorization_pref=unit_factorization_pref
                )
    print("\nMATCHED INTERMEDIATE TERMS:")
    print(intermediate_variants)

    result: e.Expr = e.Expr(0, **expr.assumptions)
    factored_terms = set()  # keep track which terms have already been factored
    factored_successfully = False

    # first try to factor all complete intermediate variants
    result, successful = _factor_complete(result, terms, itmd_cls,
                                          factored_terms,
                                          intermediate_variants)
    factored_successfully |= successful

    # go again through the remaining itmd variants and try to build more
    # complete variants by allowing mixed prefactors, i.e.,
    # add a term that belongs to a variant with prefactor 1
    # to the nearly complete variant with prefactor 2. To compensate
    # for this, additional terms are added to the result.
    result, factored_mixed_pref_successfully = _factor_mixed_prefactors(
        result, terms, itmd_cls, factored_terms, intermediate_variants
    )
    factored_successfully |= factored_mixed_pref_successfully

    # TODO:
    # go again through the remaining itmds and see if we can factor another
    # intermediate by filling up some terms, e.g. if we found 5 out of 6 terms
    # it still makes sense to factor the itmd

    # add all terms that were not involved in itmd_factorization to the result
    for term_i, term in enumerate(terms):
        if term_i not in factored_terms:
            factored_terms.add(term_i)
            result += term
    assert len(factored_terms) == len(terms)

    # if we factored the itmd successfully it might be necessary to adjust
    # sym_tensors or antisym_tensors of the returned expression
    if factored_successfully:
        tensor = itmd_cls.tensor(return_sympy=True)
        if isinstance(tensor, AntiSymmetricTensor):
            name = tensor.symbol.name
            if tensor.bra_ket_sym is S.One and \
                    name not in (sym_tensors := result.sym_tensors):
                result.set_sym_tensors(sym_tensors + (name,))
            elif tensor.bra_ket_sym is S.NegativeOne and \
                    name not in (antisym_t := result.antisym_tensors):
                result.set_antisym_tensors(antisym_t + (name,))
    return result


def _factor_short_intermediate(expr: e.Expr, itmd: EriOrbenergy,
                               itmd_data, itmd_cls) -> e.Expr:
    """Tries to factor a short intermediate, i.e., an intermediate that
       consists of a single term."""

    if expr.sympy.is_number:
        return expr

    # get the default symbols of the intermediate
    itmd_default_symbols = tuple(get_symbols(itmd_cls.default_idx))

    terms = expr.terms

    factored: e.Expr = 0  # factored expression that is returned
    factored_sucessfully = False  # bool to indicate whether we factored
    for term in terms:
        term = EriOrbenergy(term).canonicalize_sign()
        data = FactorizationTermData(term)
        # check if the current term and the itmd are compatible:
        #  - check if all necessary objects occur in the eri part
        obj_descr = data.eri_obj_descriptions
        if any(obj_descr[descr] < n for descr, n in
               itmd_data.eri_obj_descriptions.items()):
            factored += term.expr
            continue
        # - check if brackets of the correct length occur in the denominator
        if itmd_data.denom_bracket_lengths is not None:  # itmd has a denom
            bracket_lengths = data.denom_bracket_lengths
            if bracket_lengths is None:  # term has no denom
                factored += term.expr
                continue
            else:  # term also has a denom
                if any(bracket_lengths[length] < n for length, n in
                       itmd_data.denom_bracket_lengths.items()):
                    factored += term.expr
                    continue
        # ok, the term seems to be a possible match -> try to factor

        # compare the term and the itmd term
        variants = _compare_terms(term, itmd, data, itmd_data)

        if variants is None:
            factored += term.expr
            continue

        # choose the variant with the lowest overlap to other variants
        #  - find all unique obj indices (eri and denom)
        #  - and determine all itmd_indices
        unique_obj_i = {}
        for var_idx, var in enumerate(variants):
            key = (tuple(sorted(set(var['eri_i']))),
                   tuple(sorted(set(var['denom_i']))))
            if key not in unique_obj_i:
                unique_obj_i[key] = []
            unique_obj_i[key].append(var_idx)

        if len(unique_obj_i) == 1:  # always the same objects in each variant
            _, rel_variant_indices = unique_obj_i.popitem()
            min_overlap = []
        else:
            # multiple different objects -> try to find the one with the
            # lowest overlap to the other variants (so that we can possibly
            # factor the itmd more than once)
            unique_obj_i = list(unique_obj_i.items())
            overlaps = []
            for i, (key, _) in enumerate(unique_obj_i):
                eri_i, denom_i = set(key[0]), set(key[1])
                # determine the intersection of the objects
                overlaps.append(sorted(
                    [len(eri_i & set(other_key[0])) +
                     len(denom_i & set(other_key[1]))
                     for other_i, (other_key, _) in enumerate(unique_obj_i)
                     if i != other_i]
                ))
            # get the idx of the unique_obj_i with minimal intersections,
            # get the variant_data of the first element in the variant_idx_list
            min_overlap = min(overlaps)
            # collect all variant indices that have this overlap
            rel_variant_indices = []
            for overlap, (_, var_idx_list) in zip(overlaps, unique_obj_i):
                if overlap == min_overlap:
                    rel_variant_indices.extend(var_idx_list)
        # choose the variant with the minimal itmd_indices
        variant_data = min(
            [variants[var_idx] for var_idx in rel_variant_indices],
            key=lambda var: [var['sub'].get(s, s).name for s in
                             itmd_default_symbols]
        )

        # now start with factoring
        # - extract the remainder that survives the factorization (excluding
        #   the prefactor)
        remainder: e.Expr = _get_remainder(term, variant_data['eri_i'],
                                           variant_data['denom_i'])
        # - find the itmd indices:
        #   for short itmds it is not necessary to minimize the itmd indices
        #   just use whatever is found
        itmd_indices = tuple(variant_data['sub'].get(s, s) for s in
                             get_symbols(itmd_cls.default_idx))

        # ensure that there are no indices in the numerator or the
        # denominator of the remainder, that do not also occur in the eri
        # part or the itmd indices. This avoids factoring an
        # itmd using invalid contracted itmd indices that are only
        # partially removed from the term, e.g., m, a contracted itmd
        # index occurs in both, denominator and eri part, but is only
        # removed in the eri part, because the itmd has no denominator.
        _validate_indices(remainder, itmd_indices)

        # - determine the prefactor of the factored term
        pref = term.pref * variant_data['factor'] / itmd.pref
        # - check if it is possible to factor the itmd another time:
        #   should be possible if there is a 0 in the min_overlap list:
        #   -> Currently factoring a variant that has 0 overlap with another
        #      variant
        #   -> It should be possible to factor the intermediate in the
        #      remainder again!
        if 0 in min_overlap:
            # factor again and ensure that the factored result has the
            # the current assumptions
            remainder = e.Expr(
                _factor_short_intermediate(remainder, itmd, itmd_data,
                                           itmd_cls).sympy,
                **remainder.assumptions
            )
        # - build the new term including the itmd
        factored_term = _build_factored_term(remainder, pref, itmd_cls,
                                             itmd_indices)

        factored_sucessfully = True
        print(f"\nFactoring {itmd_cls.name} in:\n{term}\n"
              f"result:\n{EriOrbenergy(factored_term)}")
        factored += factored_term
    # if we factored the itmd sucessfully it might be necessary to add
    # the itmd tensor to the sym or antisym tensors
    if factored_sucessfully:
        tensor = itmd_cls.tensor(return_sympy=True)
        if isinstance(tensor, AntiSymmetricTensor):
            name = tensor.symbol.name
            if tensor.bra_ket_sym is S.One and \
                    name not in (sym_tensors := factored.sym_tensors):
                factored.set_sym_tensors(sym_tensors + (name,))
            elif tensor.bra_ket_sym is S.NegativeOne and \
                    name not in (antisym_t := factored.antisym_tensors):
                factored.set_antisym_tensors(antisym_t + (name,))
    return factored


def _factor_complete(result: e.Expr, terms: list[e.Term], itmd_cls,
                     factored_terms: set,
                     intermediate_variants: 'LongItmdVariants'
                     ) -> tuple[e.Expr, bool]:
    """Factors all intermediate variants which are complete and share
       common prefactor, i.e., no terms have to be added to the expression
       to compensate for the factorization of the intermediate.
    """
    factored_successfully = False
    for itmd_indices, remainders in intermediate_variants.items():
        for rem in remainders:
            complete_variant = intermediate_variants.get_complete_variant(
                itmd_indices, rem
            )
            while complete_variant is not None:
                # Found a complete intermediate with matching prefactors!!
                pref, term_list = complete_variant

                print(f"\nFactoring {itmd_cls.name} in terms:")
                for term_i in term_list:
                    print(EriOrbenergy(terms[term_i]))

                new_term = _build_factored_term(rem, pref, itmd_cls,
                                                itmd_indices)
                print(f"result:\n{EriOrbenergy(new_term)}", flush=True)
                result += new_term

                # remove the used terms from the pool of available terms
                # and add the terms to the already factored terms
                intermediate_variants.remove_used_terms(term_list)
                factored_terms.update(term_list)
                factored_successfully = True

                # try to find the next complete variant
                complete_variant = intermediate_variants.get_complete_variant(
                    itmd_indices, rem
                )
    # remove empty itmd_indices and remainders
    if factored_successfully:
        intermediate_variants.clean_empty()

    return result, factored_successfully


def _factor_mixed_prefactors(result: e.Expr, terms: list[e.Term], itmd_cls,
                             factored_terms: set,
                             intermediate_variants: 'LongItmdVariants'
                             ) -> tuple[e.Expr, bool]:
    """Factors intermediate variants allowing terms to have mixed prefactors.
       To compensate for the mixed prefactors additional terms are added
       to the result. Only factors intermediates if at least 60% of the
       terms have a common prefactor."""
    factored_successfully = False
    for itmd_indices, remainders in intermediate_variants.items():
        for rem in remainders:
            mixed_variant = intermediate_variants.get_mixed_pref_variant(
                itmd_indices, rem
            )
            while mixed_variant is not None:
                prefs, term_list, unit_factors, pref_counter = mixed_variant

                # determine the most common prefactor and which terms needs
                # to be added (have a different prefactor)
                most_common_pref = max(pref_counter.items(),
                                       key=lambda tpl: tpl[1])[0]
                terms_to_add = {}
                for p, term_i in zip(prefs, term_list):
                    if p == most_common_pref or term_i in terms_to_add:
                        continue
                    terms_to_add[term_i] = p

                # for all terms that don't have the most common prefactor:
                # determine the 'extension' that needs to be added to the
                # result to factor the intermediate using the most common pref
                print("\nAdding terms:")
                for term_i, p in terms_to_add.items():
                    desired_pref = most_common_pref * unit_factors[term_i]
                    term = EriOrbenergy(terms[term_i]).canonicalize_sign()
                    extension_pref = term.pref - desired_pref
                    term = extension_pref * term.num * term.eri / term.denom
                    print(EriOrbenergy(term))
                    result += term

                print(f"\nFactoring {itmd_cls.name} with mixed prefactors in:")
                for term_i in term_list:
                    print(EriOrbenergy(terms[term_i]))

                new_term = _build_factored_term(rem, most_common_pref,
                                                itmd_cls, itmd_indices)
                print(f"result:\n{EriOrbenergy(new_term)}", flush=True)
                result += new_term

                # remove the used terms from the pool of available terms
                # and add the terms to the already factored terms
                intermediate_variants.remove_used_terms(term_list)
                factored_terms.update(term_list)
                factored_successfully = True

                # try to find the next mixed intermediate
                mixed_variant = intermediate_variants.get_mixed_pref_variant(
                    itmd_indices, rem
                )
    # remove empty itmd_indices and remainders
    if factored_successfully:
        intermediate_variants.clean_empty()

    return result, factored_successfully


def _build_factored_term(remainder: e.Expr, pref, itmd_cls,
                         itmd_indices) -> e.Expr:
    tensor = itmd_cls.tensor(indices=itmd_indices, return_sympy=True)
    # resolve the Zero placeholder for residuals
    if tensor.symbol.name == "Zero":
        return e.Expr(0, **remainder.assumptions)
    return remainder * pref * tensor


def _get_remainder(term: EriOrbenergy, obj_i: list[int],
                   denom_i: list[int]) -> e.Expr:
    """Returns the remainding part of the provided term that survives the
       factorization of the itmd, excluding the prefactor!
       Note that the returned remainder can still hold a prefactor of -1,
       because sympy is not maintaining the canonical sign in the denominator.
       """
    eri: e.Expr = term.cancel_eri_objects(obj_i)
    denom: e.Expr = term.cancel_denom_brackets(denom_i)
    rem = term.num * eri / denom
    # explicitly set the target indices, because the remainder not necessarily
    # has to contain all of them.
    if rem.provided_target_idx is None:  # no target indices set
        rem.set_target_idx(term.eri.target)
    return rem


def _validate_indices(remainder: e.Expr, itmd_indices: tuple):
    """Ensure that the variant generates a valid remainder by checking that
       that all indices that occur in the numerator or denominator of the
       remainder also occur in the ERI part of the remainder.
       Say we want to factor t2sq = t_ik^ac t_jk^bc in some term.
       Because the has no denominator, the denominator of the original term
       where we try to factor the intermediate is ignored. However, one needs
       to be careful which indices are chosen as contracted intermediate
       indices k and c. They are not allowed to occur anywhere else in the
       term, i.e., also not in the denominator or numerator.
       """
    remainder = EriOrbenergy(remainder)
    required_frac_idx = set(remainder.num.idx) | set(remainder.denom.idx)
    missing_idx = (
        required_frac_idx - (set(remainder.eri.idx) | set(itmd_indices))
    )
    # maybe swith to return True/False and continue in the calling function
    # if False is Returned?
    # -> raise error for now and have a look at the cases
    if missing_idx:
        raise NotImplementedError(
            "All indices that occur in the term have to be present in the "
            "ERI part or the itmd_indices, i.e., no indices are allowed to "
            "only occur in the denominator or numerator of the remainder. "
            "This avoids only partially removing contracted itmd_indices from "
            f"a term.\n{remainder}"
        )


def _map_on_other_terms(itmd_i: int, remainder: e.Expr,
                        itmd_term_map, itmd_indices: tuple,
                        itmd_default_idx: tuple[str]):
    """Checks on which other itmd_terms the current itmd_term can be mapped if
       the symmetry of the remainder is taken into account. A set of all
       terms, the current term contributes to is returned."""
    from .symmetry import Permutation, PermutationProduct

    # find the itmd indices that are no target indices of the overall term
    # -> those are available for permutations
    target_indices = remainder.terms[0].target
    idx_to_permute = {s for s in itmd_indices if s not in target_indices}
    # copy the remainder and set the previously determined
    # indices as target indices
    rem: e.Expr = remainder.copy()
    rem.set_target_idx(idx_to_permute)
    # create a substitution dict to map the minimal indices to the
    # default indices of the intermediate
    minimal_to_default = {o: n for o, n in zip(itmd_indices, itmd_default_idx)}
    # iterate over the subset of remainder symmetry that only involves
    # non-target intermediate indices
    matching_itmd_terms: set[int] = {itmd_i}
    for perms, perm_factor in rem.terms[0].symmetry(only_target=True).items():
        # translate the permutations to the default indices
        perms = PermutationProduct(
            Permutation(minimal_to_default[p], minimal_to_default[q])
            for p, q in perms
        )
        # look up the translated symmetry in the term map
        term_map: dict = itmd_term_map[(perms, perm_factor)]
        if itmd_i in term_map:
            matching_itmd_terms.add(term_map[itmd_i])
    return matching_itmd_terms


def _compare_eri_parts(term: EriOrbenergy, itmd_term: EriOrbenergy,
                       term_data=None, itmd_term_data=None) -> list:
    """Compare the eri parts of two terms and return the substitutions
           that are necessary to transform the itmd_eri."""

    # the eri part of the term to factor has to be at least as long as the
    # eri part of the itmd (prefactors are separated!)
    if len(itmd_term.eri) > len(term.eri):
        return None

    objects = term.eri.objects
    itmd_objects = itmd_term.eri.objects

    # generate term_data if not provided
    if term_data is None:
        term_data = FactorizationTermData(term)
    # generate itmd_data if not provided
    if itmd_term_data is None:
        itmd_term_data = FactorizationTermData(itmd_term)

    relevant_itmd_data = zip(enumerate(itmd_term_data.eri_pattern),
                             itmd_term_data.eri_obj_indices,
                             itmd_term_data.eri_obj_symmetry)

    # compare all objects in the eri parts
    variants = []
    for (itmd_i, (itmd_descr, itmd_coupl)), itmd_indices, itmd_obj_sym in \
            relevant_itmd_data:
        itmd_obj_exponent = itmd_objects[itmd_i].exponent

        relevant_data = zip(enumerate(term_data.eri_pattern),
                            term_data.eri_obj_indices)
        # list to collect all obj that can match the itmd_obj
        # with their corresponding sub variants
        itmd_obj_matches = []
        for (i, (descr, coupl)), indices in relevant_data:
            # tensors have same name and space?
            # is the coupling of the itmd_obj a subset of the obj coupling?
            if descr != itmd_descr or any(coupl[c] < n for c, n in
                                          itmd_coupl.items()):
                continue
            # collect the obj index n-times to indicate how often the
            # object has to be cancelled (possibly multiple times depending
            # on the exponent of the itmd_obj)
            to_cancel = [i for _ in range(itmd_obj_exponent)]
            # create all possibilites to map the indices onto each other
            # by taking the symmetry of the itmd_obj into account
            # store them as tuple: (obj_indices, sub, factor)
            itmd_obj_matches.append((to_cancel,
                                     dict(zip(itmd_indices, indices)),
                                     1))
            for perms, factor in itmd_obj_sym.items():
                perm_itmd_indices = itmd_indices
                for p, q in perms:
                    sub = {p: q, q: p}
                    perm_itmd_indices = [sub.get(s, s) for s in
                                         perm_itmd_indices]
                itmd_obj_matches.append((to_cancel,
                                         dict(zip(perm_itmd_indices, indices)),
                                         factor))
        # was not possible to map the itmd_obj onto any obj in the term
        # -> terms can not match
        if not itmd_obj_matches:
            return None

        if not variants:  # initialize variants
            variants.extend(itmd_obj_matches)
        else:  # try to add the mapping of the current itmd_obj
            extended_variants = []
            for (i_list, sub, factor), (new_i_list, new_sub, new_factor) in \
                    product(variants, itmd_obj_matches):
                # was the obj already mapped onto another itmd_obj?
                # do we have a contradiction in the sub_dicts?
                #  -> a index in the itmd can only be mapped onto 1 index
                #     in the term simultaneously
                if new_i_list[0] not in i_list and all(
                        o not in sub or sub[o] is n
                        for o, n in new_sub.items()):
                    extended_variants.append((i_list + new_i_list,
                                              sub | new_sub,  # OR combine dict
                                              factor * new_factor))
            if not extended_variants:  # no valid combinations -> cant match
                return None
            variants = extended_variants
    # validate the found variants to map the terms onto each other
    valid = []
    for i_list, sub_dict, factor in variants:
        i_set = set(i_list)
        # did we find a match for all itmd_objects?
        if len(i_set) != len(itmd_objects):
            continue
        # extract the objects of the term
        relevant_obj = Mul(*(objects[i].sympy for i in i_set))
        # apply the substitutions to the itmd_term, remove the prefactor
        # (the substitutions might introduce a factor of -1 that we don't need)
        # and check if the substituted itmd_term is identical to the subset
        # of objects
        sub_list = order_substitutions(sub_dict)
        sub_itmd_eri = itmd_term.eri.subs(sub_list)

        if sub_itmd_eri.sympy is S.Zero:  # invalid substitution list
            continue
        pref = sub_itmd_eri.terms[0].prefactor  # +-1

        if relevant_obj - sub_itmd_eri.sympy * pref is S.Zero:
            valid.append((i_list, sub_dict, sub_list, factor))
    return valid if valid else None


def _compare_terms(term: EriOrbenergy, itmd_term: EriOrbenergy,
                   term_data=None, itmd_term_data=None) -> None | list:
    """Compare two terms and return a substitution dict that makes the
        itmd_term equal to the term. Also the indices of the objects in the
        eri part and the denominator that match the intermediate's objects
        are returned."""

    eri_variants = _compare_eri_parts(term, itmd_term, term_data,
                                      itmd_term_data)

    if eri_variants is None:
        return None

    # itmd_term has no denominator -> stop here
    if itmd_term.denom.sympy.is_number:
        return [{'eri_i': eri_i, 'denom_i': [],
                 'sub': sub_dict, 'sub_list': sub_list, 'factor': factor}
                for eri_i, sub_dict, sub_list, factor in eri_variants]

    # term and itmd_term should have a denominator at this point
    # -> extract the brackets
    brackets = term.denom_brackets
    itmd_brackets = itmd_term.denom_brackets
    # extract the lengths of all brakets
    bracket_lengths = [len(bk) for bk in brackets]
    # prescan the brackets according to their length to avoid unnecessary
    # substitutions
    compatible_brackets = {}
    for itmd_denom_i, itmd_bk in enumerate(itmd_brackets):
        itmd_bk_length = len(itmd_bk)
        matching_brackets = [denom_i for denom_i, bk_length
                             in enumerate(bracket_lengths)
                             if bk_length == itmd_bk_length]
        if not matching_brackets:  # could not find a match for a itmd bracket
            return None
        compatible_brackets[itmd_denom_i] = matching_brackets

    # check which of the found substitutions are also valid for the denominator
    variants = []
    for eri_i, sub_dict, sub_list, factor in eri_variants:
        # can only map each bracket onto 1 itmd bracket
        # otherwise something should be wrong
        denom_matches = []
        for itmd_denom_i, denom_idx_list in compatible_brackets.items():
            itmd_bk = itmd_brackets[itmd_denom_i]
            # extract base and exponent of the bracket
            if isinstance(itmd_bk, e.Expr):
                itmd_bk_exponent = 1
                itmd_bk = itmd_bk.sympy
            else:  # polynom  -> Pow object
                itmd_bk, itmd_bk_exponent = itmd_bk.base_and_exponent

            # apply the substitutions to the base of the bracket
            sub_itmd_bk = itmd_bk.subs(sub_list)
            if sub_itmd_bk is S.Zero:  # invalid substitution list
                continue

            # try to find a match in the subset of brackets of equal length
            for denom_i in denom_idx_list:
                if denom_i in denom_matches:  # denom bk is already assigned
                    continue
                bk = brackets[denom_i]
                # extract the base of the bracket
                bk = bk.sympy if isinstance(bk, e.Expr) else bk.base
                if sub_itmd_bk - bk is S.Zero:  # brackets are equal?
                    denom_matches.extend(denom_i for _ in
                                         range(itmd_bk_exponent))
                    break
            # did not run into the break:
            # -> could not find a match for the itmd_bracket
            # -> directly skip to next eri_variant
            else:
                break
        # did we find a match for all itmd brackets?
        if len(set(denom_matches)) == len(itmd_brackets):
            variants.append({'eri_i': eri_i, 'denom_i': denom_matches,
                             'sub': sub_dict, 'sub_list': sub_list,
                             'factor': factor})
    return variants if variants else None


def _compare_remainder(remainder: e.Expr, ref_remainder: e.Expr,
                       itmd_indices: tuple) -> int | None:
    """Try to map remainder onto ref_remainder. Return None if it is not
       possible. If the two remainders can be mapped, the required factor (+-1)
       is returned to indicate whether the sign of remainder needs to be
       changed to achieve equality."""
    from .reduce_expr import factor_eri_parts, factor_denom

    # if we have a number as remainder, it should be +-1
    if remainder.sympy is S.Zero or ref_remainder.sympy is S.Zero:
        raise ValueError("It should not be possible for a remainder to "
                         "be equal to 0.")

    # in addition to the target indices, the itmd_indices have to be fixed too.
    # -> set both indices sets as target indices of the expressions
    fixed_indices = remainder.terms[0].target
    assert fixed_indices == ref_remainder.terms[0].target
    fixed_indices += itmd_indices

    # create a copy of the expressions to keep the assumptions of the original
    # expressions valid (assumptions should hold the correct target indices)
    remainder, ref_remainder = remainder.copy(), ref_remainder.copy()
    remainder.set_target_idx(fixed_indices)
    ref_remainder.set_target_idx(fixed_indices)

    # TODO: we have a different situation in this function, because not all
    #       contracted indices have to occur in the eri part of the remainder:
    #         eri indices: jkln. Additionally we have m in the denominator.
    #       the function will only map n->m but not m->n, because it does
    #       not occur in the eri part. This might mess up the denominator
    #       or numerator of the term completely!
    #       -> can neither use find_compatible_terms nor compare_terms!!
    # I think in a usual run this should only occur if previously some
    # intermediate was not found correctly, because for t-amplitudes all
    # removed indices either only occur in the eri part or occur in eri and
    # denom. But if we did not find some t-amplitude and have some denominator
    # left, this problem might occur if a denom idx is a contracted index
    # in the eri part of the itmd.
    # -> but then we can not factor the itmd anyway, because the contracted
    #    idx in the eri part and the denom have to be identical
    # -> need to be solved at another point

    difference = remainder - ref_remainder
    if len(difference) == 1:  # already identical -> 0 or added to 1 term
        return 1 if difference.sympy is S.Zero else -1
    # check if the eri parts of both remainders can be mapped onto each other
    factored = factor_eri_parts(difference)
    if len(factored) > 1:  # eri parts not compatible
        return None

    # check if the denominators are compatible too.
    factored = factor_denom(factored[0])
    if len(factored) > 1:  # denominators are not compatible
        return None
    return 1 if factored[0].sympy is S.Zero else -1


class LongItmdVariants(dict):
    """Class to manage the variants of long intermediates."""

    def __init__(self, n_itmd_terms: int, *args, **kwargs):
        self.n_itmd_terms = n_itmd_terms
        # The number of terms we require to share a common prefactor
        # for mixed prefactor intermediates
        self.n_common_pref_terms = (0.6 * self.n_itmd_terms).__ceil__()
        super().__init__(*args, **kwargs)

    def add(self, term_i: int, itmd_indices: tuple, remainder: e.Expr,
            matching_itmd_terms: tuple[int],
            prefactor, unit_factorization_pref) -> None:
        """Add a matching term-itmd_term pair to the pool for building
           intermediate variants.
        """

        # trivial separation by itmd_indices (the indices of the itmd we
        # try to factor with the current variant)
        if itmd_indices not in self:
            self[itmd_indices] = {}

        is_new_remainder = True
        for rem, found_matches in self[itmd_indices].items():
            # next we can separate the variants by the remainder they will
            # create when the variant is factored
            factor = _compare_remainder(remainder=remainder, ref_remainder=rem,
                                        itmd_indices=itmd_indices)
            if factor is None:  # remainder did not match
                continue

            is_new_remainder = False
            # possibly we got another -1 from matching the remainder
            prefactor *= factor

            # next, we can separate them according to the itmd_positions
            # so we can later build intermediate variants more efficient
            matching_itmd_terms = tuple(sorted(matching_itmd_terms))
            if matching_itmd_terms not in found_matches:
                found_matches[matching_itmd_terms] = []
            # It is possible to obtain entries that have the same
            # term_i and pref, but differ in the sign of the unit factor
            # this is probably a result of the permutation symmetry
            # of some intermediates
            # -> only add the term if term_i and pref have not been found yet
            is_dublicate = any(
                (term_i == other_term_i and prefactor == other_pref
                 and abs(unit_factorization_pref) == abs(other_unit_factor))
                for other_term_i, other_pref, other_unit_factor in
                found_matches[matching_itmd_terms]
            )
            if not is_dublicate:
                found_matches[matching_itmd_terms].append(
                    (term_i, prefactor, unit_factorization_pref)
                )
            break
        if is_new_remainder:
            self[itmd_indices][remainder] = {}
            matching_itmd_terms = tuple(sorted(matching_itmd_terms))
            self[itmd_indices][remainder][matching_itmd_terms] = [
                (term_i, prefactor, unit_factorization_pref)
            ]

    def get_complete_variant(self, itmd_indices, remainder) -> None | tuple:
        """Returns the first complete variant it finds for the given
           itmd_indices and the remainder. Only variants that are complete
           and share a common prefactor are considered here.
           If no variant can be found None is returned.
        """

        def sort_matches(pool: list) -> list:
            term_i_counter = {}
            for positions, matches in pool:
                for term_i in matches:
                    if term_i not in term_i_counter:
                        term_i_counter[term_i] = {}
                    for p in positions:
                        if p not in term_i_counter[term_i]:
                            term_i_counter[term_i][p] = 0
                        term_i_counter[term_i][p] += 1
            term_i_counter = {term_i: (len(positions), sum(positions.values()))
                              for term_i, positions in term_i_counter.items()}
            return [(pos, sorted(matches, key=lambda m: term_i_counter[m]))
                    for pos, matches in pool]

        # itmd_indices and or remainder not found
        if itmd_indices not in self or \
                remainder not in self[itmd_indices]:
            return None
        pool = self[itmd_indices][remainder]
        if not pool:  # empty pool: already factored everything
            return None

        # construct base variants that are likely to form complete variants
        for pref, term_list in self._complete_base_variants(pool):
            # filter the pool:
            # - remove all already occupied positions
            # - remove all matches of already used terms
            # - remove all matches that have a different prefactor
            relevant_pool = {}
            for positions, matches in pool.items():
                if any(term_list[p] is not None for p in positions):
                    continue
                relevant_matches = {
                    term_i for term_i, other_pref, _ in matches
                    if other_pref == pref and term_i not in term_list
                }
                if relevant_matches:
                    relevant_pool[positions] = relevant_matches
            if not relevant_pool:  # nothing relevant left
                continue

            # sort the pool:
            # - start with the positions with the lowest number of matches
            # - prioritize rare indices
            relevant_pool = sorted(relevant_pool.items(),
                                   key=lambda kv: len(kv[1]))
            relevant_pool = sort_matches(relevant_pool)

            # set up masks to avoid creating copies of the pool
            pos_mask = [True for _ in relevant_pool]
            match_masks = [[True for _ in matches]
                           for _, matches in relevant_pool]
            # try to complete the base variant from the relevant pool
            success = self._build_complete_variant(
                term_list, relevant_pool, pos_mask, match_masks
            )
            if success:
                return pref, term_list
            # continue with the next base variant
        # loop completed -> no complete variant found
        return None

    def _complete_base_variants(self, pool: dict):
        """Iterator over the base variants for complete intermediates."""

        def sort_matches(pool: dict, matches_to_sort: list) -> list:
            term_i_counter = {}
            pref_available_pos = {}
            for positions, matches in pool.items():
                for term_i, pref, _, in matches:
                    if term_i not in term_i_counter:
                        term_i_counter[term_i] = {}
                    if pref not in pref_available_pos:
                        pref_available_pos[pref] = [False for _ in
                                                    range(self.n_itmd_terms)]
                    for p in positions:
                        if p not in term_i_counter[term_i]:
                            term_i_counter[term_i][p] = 0
                        term_i_counter[term_i][p] += 1
                        pref_available_pos[pref][p] = True
            term_i_counter = {term_i: (
                    len(positions),
                    sum(positions.values())
                ) for term_i, positions in term_i_counter.items()}
            # remove prefactors where not all positions are available
            matches_to_sort = [m for m in matches_to_sort
                               if all(pref_available_pos[m[1]])]
            return sorted(matches_to_sort, key=lambda m: term_i_counter[m[0]])

        # find the position with the lowest number of matches
        pos, matches = min(pool.items(), key=lambda kv: len(kv[1]))
        # sort the matches so that rare term_i are covered first
        # and remove prefactors where not all positions are available
        if len(matches) > 1:
            matches = sort_matches(pool, matches)

        # ensure we only ever try once per term_i and pref combination
        prev_tried = {}
        for term_i, pref, _ in matches:
            if pref not in prev_tried:
                prev_tried[pref] = set()
            if term_i in prev_tried[pref]:
                continue
            prev_tried[pref].add(term_i)

            yield (pref,
                   [term_i if i in pos else None
                    for i in range(self.n_itmd_terms)])

    def _build_complete_variant(self, term_list: list, pool: list,
                                pos_mask: list, match_masks: list) -> bool:
        """Tries to recursively complete the given variant from the pool
           of matches."""
        # check if the variant can be completed with the available
        # positions
        unique_positions = {p for pos, _ in compress(pool, pos_mask)
                            for p in pos}
        n_missing_terms = term_list.count(None)
        if n_missing_terms > len(unique_positions):
            return False

        for i, (positions, matches) in compress(enumerate(pool), pos_mask):
            # update the mask:
            # mask the positions that will be filled in the following loop
            pos_mask[i] = False

            # since all positions have to be available we can already
            # predict here whether we will be able to complete the variant
            completed = (n_missing_terms == len(positions))
            for term_i in compress(matches, match_masks[i]):
                # don't copy the term_list. Instead revert the changes
                # before continue the iteration
                for p in positions:
                    term_list[p] = term_i

                if completed:  # check if we completed the variant
                    return True

                # update the mask:
                # mask all positions that intersect with the filled
                # positions and mask term_i as not available
                # for now just store the mask changes, but we can
                # also recompute the changes to revert them.
                masked_pos = []
                masked_matches = []
                for other_i, (pos, other_matches) in \
                        compress(enumerate(pool), pos_mask):
                    if any(p in positions for p in pos):
                        # no need to update the match mask here
                        pos_mask[other_i] = False
                        masked_pos.append(other_i)
                        continue
                    for j, other_term_i in \
                            compress(enumerate(other_matches),
                                     match_masks[other_i]):
                        if term_i == other_term_i:
                            match_masks[other_i][j] = False
                            masked_matches.append((other_i, j))
                    if not any(match_masks[other_i]):
                        pos_mask[other_i] = False
                        masked_pos.append(other_i)

                # recurse and try to complete the variant
                success = self._build_complete_variant(
                    term_list, pool, pos_mask, match_masks
                )
                if success:  # found complete variant
                    return True

                # revert the mask changes
                for other_i in masked_pos:
                    pos_mask[other_i] = True
                for other_i, j in masked_matches:
                    match_masks[other_i][j] = True

                # revert the changes to term_list and continue the loop
                for p in positions:
                    term_list[p] = None
            # unmask the position
            pos_mask[i] = True
        return False

    def get_mixed_pref_variant(self, itmd_indices, remainder) -> None | tuple:
        """Finds an intermediate variant allowing mixed prefactors for the
           given itmd_indices and remainder."""

        # itmd_indices or remainder not available
        if itmd_indices not in self or \
                remainder not in self[itmd_indices]:
            return None
        pool = self[itmd_indices][remainder]
        if not pool:  # empty pool: already factored all term_i
            return None

        for (prefs, term_list, unit_factors) in \
                self._mixed_pref_base_variants(pool):
            # filter the pool by removing all positions that are already
            # occupied. Addtionally, remove all term_i that are already
            # in use.
            relevant_pool = {}
            for positions, matches in pool.items():
                if any(term_list[p] is not None for p in positions):
                    continue
                relevant_matches = [
                    data for data in matches if data[0] not in term_list
                ]
                if relevant_matches:
                    relevant_pool[positions] = relevant_matches
            if not relevant_pool:
                continue

            # sort the pool to start with the position with the lowest amount
            # of valid matches and prioritize rare term_i and common prefactors
            relevant_pool = sorted(relevant_pool.items(),
                                   key=lambda kv: len(kv[1]))
            relevant_pool = self._sort_mixed_pref_matches(relevant_pool)

            # set up masks for position and matches to avoid copying data
            pos_mask = [True for _ in relevant_pool]
            match_masks = [[True for _ in matches]
                           for _, matches in relevant_pool]
            pref_counter = Counter([p for p in prefs if p is not None])
            # try to complete the base variant using the relevant pool
            success = self._complete_mixed_variant(
                term_list, prefs, unit_factors, relevant_pool, pref_counter,
                pos_mask, match_masks
            )
            if success:
                return prefs, term_list, unit_factors, pref_counter
            # else continue with the next base variant
        # loop completed -> no mixed variant found
        return None

    def _mixed_pref_base_variants(self, pool: dict):
        """Iterator over the base variants for intermediates with
           mixed prefactors."""
        # find the positions with the lowest number of matches
        pos, matches = min(pool.items(), key=lambda kv: len(kv[1]))
        # sort the matches so that
        # rare indices and common prefactors are preferred
        if len(matches) > 1:
            matches = self._sort_mixed_pref_matches(pool.items(), matches)

        # filter out matches that have the same term_i and pref
        prev_tried = {}
        for term_i, pref, unit_factor in matches:
            if pref not in prev_tried:
                prev_tried[pref] = set()
            if term_i in prev_tried[pref]:
                continue
            prev_tried[pref].add(term_i)

            yield ([pref if i in pos else None
                    for i in range(self.n_itmd_terms)],
                   [term_i if i in pos else None
                    for i in range(self.n_itmd_terms)],
                   {term_i: unit_factor})

    def _sort_mixed_pref_matches(self, pool: list[tuple],
                                 matches_to_sort: list = None) -> list:
        """Sorts all matches in the pool so that rare term_i and common
           prefactors are preferred. If an additional match list is provided
           instead this match list will be sorted instead of all matches in
           the pool."""
        term_i_counter = {}
        pref_counter = {}
        for positions, matches in pool:
            for term_i, pref, _ in matches:
                if term_i not in term_i_counter:
                    term_i_counter[term_i] = {}
                if pref not in pref_counter:
                    pref_counter[pref] = {}
                for pos in positions:
                    if pos not in term_i_counter[term_i]:
                        term_i_counter[term_i][pos] = 0
                    term_i_counter[term_i][pos] += 1
                    if pos not in pref_counter[pref]:
                        pref_counter[pref][pos] = 0
                    pref_counter[pref][pos] += 1
        term_i_counter = {term_i: (
                len(positions),
                sum(positions.values())
            ) for term_i, positions in term_i_counter.items()}
        pref_counter = {pref: (
                -len(positions),
                -sum(positions.values())
            ) for pref, positions in pref_counter.items()}

        if matches_to_sort is None:
            return [
                (pos, sorted(matches, key=lambda m: (*term_i_counter[m[0]],
                                                     *pref_counter[m[1]])))
                for pos, matches in pool
            ]
        else:
            return sorted(matches, key=lambda m: (*term_i_counter[m[0]],
                                                  *pref_counter[m[1]]))

    def _complete_mixed_variant(self, term_list: list, prefactors: list,
                                unit_factors: dict, pool: list,
                                pref_counter: dict, pos_mask: list,
                                match_masks: list) -> bool:
        """Tries to complete the variant from the pool allowing mixed
           mixed prefactors. Only variants where at least 60% of the
           terms share a common prefactor are accepted."""
        # check if the variant can be completed with the available
        # positions
        unique_positions = {p for pos, _ in compress(pool, pos_mask)
                            for p in pos}
        n_missing_terms = term_list.count(None)
        if n_missing_terms > len(unique_positions):
            return False

        for i, (positions, matches) in compress(enumerate(pool), pos_mask):
            # update the poositions mask
            pos_mask[i] = False

            completed = (n_missing_terms == len(positions))
            for term_i, pref, unit_factor in \
                    compress(matches, match_masks[i]):
                # if we add the match: will we still be able to
                # create a valid variant that hast at least 60% common
                # prefactor?
                pref_counter[pref] += len(positions)

                max_terms_common_pref = max(pref_counter.values()) + \
                    n_missing_terms - len(positions)
                if max_terms_common_pref < self.n_common_pref_terms:
                    # we will not be able to complete the variant
                    # with the current addition
                    pref_counter[pref] -= len(positions)
                    continue

                # add the current match to the variant
                for p in positions:
                    term_list[p] = term_i
                    prefactors[p] = pref
                unit_factors[term_i] = unit_factor

                if completed and max(pref_counter.values()) >= \
                        self.n_common_pref_terms:
                    return True

                # update the mask:
                # - mask any position that intersects with the the added
                #   positions
                # - mask all otherm matches of term_i
                masked_pos = []
                masked_matches = []
                for other_i, (pos, other_matches) in \
                        compress(enumerate(pool), pos_mask):
                    if any(p in positions for p in pos):
                        pos_mask[other_i] = False
                        masked_pos.append(other_i)
                        continue
                    for j, (other_term_i, _, _) in \
                            compress(enumerate(other_matches),
                                     match_masks[other_i]):
                        if term_i == other_term_i:
                            match_masks[other_i][j] = False
                            masked_matches.append((other_i, j))
                    if not any(match_masks[other_i]):
                        pos_mask[other_i] = False
                        masked_pos.append(other_i)

                # recurse and try to complete the variant
                success = self._complete_mixed_variant(
                    term_list, prefactors, unit_factors, pool, pref_counter,
                    pos_mask, match_masks
                )
                if success:
                    return True

                # revert the mask changes
                for other_i in masked_pos:
                    pos_mask[other_i] = True
                for other_i, j in masked_matches:
                    match_masks[other_i][j] = True

                # undo the changes to the variant
                for p in positions:
                    term_list[p] = None
                    prefactors[p] = None
                del unit_factors[term_i]

                # undo the prefcounter changes
                pref_counter[pref] -= len(positions)

            # unmaks the position
            pos_mask[i] = True
        return False

    def remove_used_terms(self, used_terms: list[int]) -> None:
        """Removes the provided terms from the pool, so they can not
           be used to build further variants.
        """
        for remainders in self.values():
            for positions in remainders.values():
                empty_pos = []
                for pos, matches in positions.items():
                    to_delete = [i for i, m in enumerate(matches)
                                 if m[0] in used_terms]
                    # need to remove element with highest index first!
                    for i in sorted(to_delete, reverse=True):
                        del matches[i]
                    if not matches:  # removed all matches for the position
                        empty_pos.append(pos)
                for pos in empty_pos:
                    del positions[pos]

    def clean_empty(self) -> None:
        """Removes all empty entries in the nested dictionary.
        """
        empty_indices = []
        for itmd_indices, remainders in self.items():
            empty_rem = [rem for rem, positions in remainders.items()
                         if not positions]
            for rem in empty_rem:
                del remainders[rem]
            if not remainders:
                empty_indices.append(itmd_indices)
        for itmd_indices in empty_indices:
            del self[itmd_indices]


class FactorizationTermData:
    """Class that extracts some data needed for the intermediate factorization.
       """

    def __init__(self, term: EriOrbenergy):
        self._term = term

    @cached_property
    def eri_pattern(self) -> tuple:
        """Returns the pattern of the eri part of the term. In contrast to the
           pattern used in simplify, the pattern is determined for each object
           as tuple that consists of the object description and the
           coupling of the object."""
        coupling = self._term.eri.coupling(include_exponent=False,
                                           include_target_idx=False)
        return tuple(
            (obj.description(include_exponent=False, include_target_idx=False),
             Counter(coupling.get(i, [])))
            for i, obj in enumerate(self._term.eri.objects)
        )

    @cached_property
    def eri_obj_indices(self) -> tuple:
        """Indices hold by each of the objects in the eri part."""
        return tuple(obj.idx for obj in self._term.eri.objects)

    @cached_property
    def eri_obj_symmetry(self) -> tuple:
        """Symmetry of all objects in the eri part."""
        return tuple(obj.symmetry() for obj in self._term.eri.objects)

    @cached_property
    def eri_obj_descriptions(self) -> Counter:
        """Count how often each description occurs in the eri part.
           Exponent of the objects is included implicitly by incrementing
           the description counter."""
        return Counter(
            obj.description(include_exponent=False, include_target_idx=False)
            for obj in self._term.eri.objects for _ in range(obj.exponent)
        )

    @cached_property
    def denom_bracket_lengths(self) -> None | Counter:
        """Determine the length of all brackets in the orbital energy
           denominator and count how often each length occurs in the
           denominator."""
        if self._term.denom.is_number:
            return None
        else:
            return Counter(len(bk) for bk in self._term.denom_brackets)
