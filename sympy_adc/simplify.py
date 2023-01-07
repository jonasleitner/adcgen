from .indices import index_space, get_first_missing_index, get_symbols
from .misc import Inputerror
from . import sort_expr as sort
from . import expr_container as e
from sympy import Add


def filter_tensor(expr, t_strings, strict='low', ignore_amplitudes=True):
    """Filter an expression and only return terms that contain certain tensors.
       The names of the desired tensors need to be provided as list/tuple of
       strings.
       This filtering is implemented in 3 different ways, controlled by strict:
        - 'high': return all terms that ONLY contain the desired tensors in the
                  requested amount. E.g.: ['V', 'V'] returns only
                  terms that contain not other tensors than 'V*V'
                  Setting ignore_amplitudes, ignores all not requested
                  amplitudes for this.
        - 'medium': return all terms that contain the desired tensors in the
                    requested amount, but other tensors may additionally be
                    present in the term. E.g. ['V', 'V'] also returns terms
                    that contain 'V*V*x', where x may be any amount of
                    arbitrary other tensors.
        - 'low': return all terms that contain all of the requested tensors.
                 E.g. ['V', 'V'] return all terms that contain 'V' at least
                 once.
       """
    from collections import Counter

    def check_term(term):
        available = [o.name for o in term.tensors for _ in range(o.exponent)]
        # True if all requested tensors are in the term
        if strict == 'low':
            return all(t in available for t in set(t_strings))
        # True if all requested Tensors occure the correct amount of times
        elif strict == 'medium':
            available = Counter(available)
            desired = Counter(t_strings)
            return desired.items() <= available.items()
        # True if ony the requested Tensors are in the term in the correct
        # amount
        elif strict == 'high':
            if ignore_amplitudes:
                requested_amplitudes = [
                    name for name in t_strings if name[0] in ['X', 'Y'] or
                    (name[0] == 't' and name[1:].replace('c', '').isnumeric())
                ]
                ignored_amplitudes = {name for name in available
                                      if name[0] in ['t', 'X', 'Y'] and
                                      name not in requested_amplitudes}
                available = Counter([t for t in available
                                     if t not in ignored_amplitudes])
            else:
                available = Counter(available)
            desired = Counter(t_strings)
            return desired == available

    if not all(isinstance(t, str) for t in t_strings):
        raise Inputerror("Tensor names need to be provided as str.")
    if strict not in ['low', 'medium', 'high']:
        raise Inputerror(f"{strict} is not a valid option for strict. Valid"
                         "options are 'low', 'medium' or 'high'.")

    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)

    filtered = Add(*[term.sympy for term in expr.terms if check_term(term)])
    return e.expr(filtered, **expr.assumptions)


def find_compatible_terms(terms: list[e.term]):
    from itertools import product

    if not all(isinstance(term, e.term) for term in terms):
        raise Inputerror("Expected terms as a list of term Containers.")

    # NOTE: tensors and deltas do not cover deltas and tensors in polynoms
    #       However, this should not be a problem, since a term
    #       (a+b) / (c+d) * (e+f) may be split in the three brakets which
    #       can be treated correctly.

    # extract the target indices
    term_target: list[tuple] = [term.target for term in terms]
    term_pattern: list[dict] = []
    term_tensors: list[list[str]] = []
    term_deltas: list[list[str]] = []
    term_target_obj: list[list[tuple]] = []
    for term, target_idx in zip(terms, term_target):
        # extract the pattern
        term_pattern.append(term.pattern(target=target_idx))
        # extract tensors
        term_tensors.append(sorted(
            [o.name for o in term.tensors for _ in range(o.exponent)]
        ))
        # extract deltas
        term_deltas.append(sorted(
            [o.space for o in term.deltas for _ in range(o.exponent)]
        ))
        # extract objects that hold the target indices and also how many occ
        # virt target indices the object holds, e.g. f_cc Y_ij^ac
        # Y holds 2o, 1v
        temp = []
        for o in term.target_idx_objects:
            obj_target_sp = "".join(
                [index_space(s.name)[0] for s in o.idx if s in target_idx]
            )
            temp.extend([
                (o.description(), obj_target_sp.count('o'), obj_target_sp.count('v'))  # noqa E501
                for _ in range(o.exponent)
            ])
        term_target_obj.append(sorted(temp))
        del temp

    # collect terms that are equal according to their pattern
    equal_terms: dict[int, dict[int, dict]] = {}
    matched: set = set()
    for i, pattern in enumerate(term_pattern):
        if i in matched:
            continue
        term = terms[i]
        target = term_target[i]
        tensors = term_tensors[i]
        deltas = term_deltas[i]
        target_obj = term_target_obj[i]
        for other_i in range(i+1, len(term_pattern)):
            other_term = terms[other_i]
            other_pattern = term_pattern[other_i]
            other_target = term_target[other_i]
            # check if the terms are compatible:
            # - have the same length (number of objects) up to a prefactor
            # - the same number and type of indices
            # - the same target indices
            # - the same tensors and deltas
            # - the target indices are placed on the same objects
            if other_i in matched or abs(len(term) - len(other_term)) > 1 or \
                    pattern.keys() != other_pattern.keys() or \
                    any(len(idx_pat) != len(other_pattern[sp])
                        for sp, idx_pat in pattern.items()) or \
                    target != other_target or \
                    tensors != term_tensors[other_i] or \
                    deltas != term_deltas[other_i] or \
                    target_obj != term_target_obj[other_i]:
                continue

            # try to map the indices onto each other according to the pattern
            # if a match for all indices can be found, it should be possible
            # to combine both terms into a single term (or cancel them to 0)
            # by applying index permutations of contracted indices
            match: bool = True
            sub_list: list[dict] = []
            for ov, idx_pat in pattern.items():
                other_idx_pat = other_pattern[ov]
                ov_maps: list[dict] = []
                for idx, pat in idx_pat.items():
                    # find all possible matches for the given index
                    # if its a target index -> can only map onto other
                    # target indices
                    is_target = idx in target
                    matching_idx = []  # list to collect all possible matches
                    for other_idx, other_pat in other_idx_pat.items():
                        other_is_target = other_idx in other_target
                        if is_target != other_is_target:
                            continue  # only one of them is a target idx
                        # pattern is sorted list of strings -> directly compare
                        if pat == other_pat:
                            if is_target and other_is_target and \
                                    idx is not other_idx:
                                continue  # can't substitute target indices
                            matching_idx.append(other_idx)
                    if not matching_idx:
                        break  # could not find a match for the idx
                    if not ov_maps:
                        # initialize all sub dicts
                        ov_maps.extend([{s: idx} for s in matching_idx])
                    else:  # sub dicts already initialized -> add when possible
                        for sub, other_idx in product(ov_maps, matching_idx):
                            if other_idx in sub or idx in sub.values():
                                continue
                            sub[other_idx] = idx

                # filter incomplete sub dicts and remove redundant entries
                valid: list[dict] = []
                for sub in ov_maps:
                    if sub.keys() != other_idx_pat.keys():
                        continue  # did not find a match for all idx
                    valid.append({old: new for old, new in sub.items()
                                  if old is not new})
                if not valid:
                    match = False
                    break
                if not sub_list:
                    # initialize all final sub dicts
                    sub_list.extend(valid)
                else:  # all sub dicts already initialized
                    # spaces can not overlap -> just combine the dicts
                    sub_list = [other_sp_sub | sub for other_sp_sub, sub
                                in product(sub_list, valid)]
            if not match:  # could not map the terms onto each other
                continue

            for sub in sub_list:
                if not sub:  # can a sub dict be empty?
                    continue
                # test the sub_dict: terms should add up or cancel to 0
                sub_other = other_term.subs(sub, simultaneous=True)
                if len(term - sub_other) == 1:
                    if i not in equal_terms:
                        equal_terms[i] = {}
                    equal_terms[i][other_i] = sub
                    matched.add(other_i)
                    break
    return equal_terms


def simplify(expr, real=False):
    """Simplify an expression by renaming indices. The new index names are
       determined by establishing a mapping between the indices in different
       terms. If all indices in two terms share the same pattern (essentially
       have the same occurences), but have different names. The function will
       rename the indices in one of the two terms.
       If real is set, all 'c' are removed in the tensor names in order to make
       the amplitudes real. Additionally, make_real is called that tries to
       further simplify the expression by swapping bra and ket of symmetric
       tensors. By default the Fock matrix 'f' and the ERI 'V' are added to
       the provided symmetric tensors.
       """

    # start = time.time()

    expr = expr.expand()
    # adjust symmetric tensors of the container
    if not isinstance(expr, e.expr):
        expr = e.expr(expr, real)
    if real and not expr.real:
        expr = expr.make_real()

    # create terms and hand try to find comaptible terms that may be
    # simplified by substituting indices
    terms = expr.terms
    equal_terms = find_compatible_terms(terms)

    # substitue the indices in other_n and keep n as is
    res = e.compatible_int(0)
    matched = set()
    for n, sub_dict in equal_terms.items():
        matched.add(n)
        res += terms[n]
        for other_n, sub in sub_dict.items():
            matched.add(other_n)
            res += terms[other_n].subs(sub, simultaneous=True)
    # Add the unmatched remainder
    res += e.expr(Add(*[terms[n].sympy for n in range(len(terms))
                  if n not in matched]), **expr.assumptions)
    del terms  # not valid anymore (expr changed)
    # print(f"simplify took {time.time()- start} seconds")
    return res


def extract_dm(expr, bra_ket_sym: int = None):
    """Function that extracts the density matrix from the expectation
       value expression. Thereby, a dict is returned that contains the
       defintions of all canonical blocks of the density matrix."""
    from sympy.physics.secondquant import KroneckerDelta
    from sympy import Rational
    from .sympy_objects import AntiSymmetricTensor

    def minimize_d_indices(term: e.expr, d_tensor: dict,
                           target: tuple) -> tuple[e.expr, dict]:
        from collections import defaultdict

        used_indices = defaultdict(list)
        for s in target:  # add all target indices to used indices
            ov = index_space(s.name)
            if s not in used_indices[ov]:
                used_indices[ov].append(s)
        # start with upper
        d_tensor = sorted(d_tensor.items(), key=lambda tpl: tpl[0],
                          reverse=True)
        minimization_sub = {}
        for _, idx_list in d_tensor:
            for idx in idx_list:
                if idx in target:  # skip target indices
                    continue
                # find the lowest unused index
                name = idx.name
                sp = index_space(name)
                new_idx = get_first_missing_index(used_indices[sp], sp)
                used_indices[sp].append(new_idx)
                if name == new_idx:  # is already the lowest idx
                    continue
                # found a lower index -> permute indices in the term
                new_idx = get_symbols(new_idx)[0]
                sub = {idx: new_idx, new_idx: idx}
                # immediately permute the d_tensor indices
                for i, (_, other_idx_list) in enumerate(d_tensor):
                    for other_i, s in enumerate(other_idx_list):
                        d_tensor[i][1][other_i] = sub.get(s, s)
                # and build a minimization sub dict to minimize
                # the indices in the term at once
                if not minimization_sub:
                    minimization_sub = sub
                else:
                    for old, new in minimization_sub.items():
                        if new is new_idx:
                            minimization_sub[old] = idx
                            del sub[new_idx]
                        elif new is idx:
                            minimization_sub[old] = new_idx
                            del sub[idx]
                    if sub:
                        minimization_sub.update(sub)
        return term.subs(minimization_sub, simultaneous=True), dict(d_tensor)

    def symmetrize_keep_pref(term, symmetry):
        symmetrized = term.copy()
        for perms, factor in symmetry.items():
            symmetrized += term.copy().permute(*perms) * factor
        return symmetrized

    if bra_ket_sym is not None and bra_ket_sym not in [0, 1, -1]:
        raise Inputerror(f"Invalid bra_ket symmetry {bra_ket_sym}. 0, 1 and -1"
                         "are valid values.")

    # assume no polynoms are present in the term
    expr = expr.expand()
    # ensure that the expression is in a container
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)

    # sort the expression according to the blocks of the density matrix
    blocks = sort.by_tensor_block(expr, 'd', bra_ket_sym)

    for block, block_expr in blocks.items():
        if block_expr.sympy.is_number:
            continue

        removed = e.compatible_int(0)
        # - remove d in each term
        for term in block_expr.terms:
            new_term = e.expr(1, **term.assumptions)
            d_tensor = None
            for t in term.objects:
                # found the d tensor
                if t.type == 'antisym_tensor' and t.name == 'd':
                    if d_tensor is not None:
                        raise NotImplementedError("Found two d tensors in the "
                                                  f"term {term}.")
                    d_tensor = t.extract_pow
                    current_bra_ket_sym = d_tensor.bra_ket_sym
                    d_tensor: dict[str, list] = {'upper': list(d_tensor.upper),
                                                 'lower': list(d_tensor.lower)}
                else:  # anything else than the d tensor
                    new_term *= t
            if d_tensor is None:
                raise RuntimeError("Could not find a d tensor in the term "
                                   f"{term}.")
            # if indices repeat on the d tensor -> introduce a delta.
            # I think this should only be possible in the occ space, because it
            # should originate from a p+ q contraction
            repeated_idx = [idx for idx in d_tensor['upper']
                            if idx in d_tensor['lower']]
            if repeated_idx:  # extract the indices of new_term
                term_idx: dict[str, set] = {}
                for s in new_term.idx:
                    if (ov := index_space(s.name)) not in term_idx:
                        term_idx[ov] = set()
                    term_idx[ov].add(s.name)
            for idx in repeated_idx:
                # indices that are currently present in the term that belong
                # to the same space as the repeated index
                sp = index_space(idx.name)
                current_idx = {s.name for idx_list in d_tensor.values()
                               for s in idx_list if index_space(s.name) == sp}
                if sp in term_idx:
                    current_idx.update(term_idx[sp])
                # generate a second index of the same space for the delta, e.g.
                # X_ij d^ki_kj -> X_ij delta_kl
                new_idx = get_symbols(get_first_missing_index(current_idx, sp))[0]  # noqa E501
                new_term *= KroneckerDelta(idx, new_idx)
                # also replace the idx in the d_tensor. Does not matter if in
                # upper or lower -> just arbitrarily use lower
                # the index can only appear once in lower. Otherwise the tensor
                # has to be 0 according to the antisymmetry.
                d_tensor['lower'][d_tensor['lower'].index(idx)] = new_idx

            # Now minimize the indices on the d_tensor - the target indices of
            # the term after the d tensor is removed.
            # This ensures that all terms that contribute to the ovov block,
            # for instance, have the same target indices iajb.
            new_term, d_tensor = minimize_d_indices(new_term, d_tensor,
                                                    term.target)
            # now symmetrize the term.
            # Here only Permutations P_ij / P_ab are taken into account,
            # because permutations P_ia will produce a term that contributes to
            # a different, non-canonical, block of the density matrix, e.g.,
            # P_ka d_ijka -> d_ijak, while P_ij d_ijab -> d_jiab is still part
            # of the canonical oovv block.
            # As a result, this function only returns terms that contribute
            # to canonical blocks of the DM.
            # Therefore, when computing an expectation value, e.g., the
            # ovov block needs to be multiplied by a factor of 4, because it
            # also represents the voov, ovvo, vovo blocks, which are not
            # present in memory - due to the antisym of the density and the
            # operator matrix:
            #   D_ovov d_ovov = D_voov d_voov.
            # Symmetric DM's have additional symmetry on diagonal blocks,
            # e.g., D_ijkl = D_klij.
            # Therefore, terms that contribute to a diagonal block are
            # multiplied by 1/2 and the bra/ket permutation
            # is applied in addition to the anti-symmetry permutations.
            # Regarding the prefactor:
            #  - we have to multiply with the inverse of the prefactor that is
            #    introduced from the sum D_pqrs d_pqrs when computing the
            #    expectation value:
            #       mutliply by 4 for 2p-DM.
            #  - However, another prefactor of 1/4 has to be introduced to
            #    account for the anti-symmetry of the operator matrix:
            #       X d_ijab = 1/4 X (1 - P_ij)(1 - P_ab) d_jiab
            #     -> both prefactors cancel each other exactly. No need to
            #        adjust the prefactor
            #  - if a symmetric DM is computed for a symmetric operator matrix,
            #    terms of non-canonical blocks are added to canonical blocks,
            #    because in this case
            #       d_ovoo = d_ooov and D_ovoo = D_oovo.
            #    As a consequence if the pure ooov block is desired for a
            #    symmetric tensor
            #   -> need to multiply all tems by 1/2 if the bra/ket swap
            #      gives a non-canonical block. But only if bra/ket are of the
            #      same length. Otherwise the operator can not be symmetric
            #      anyway.
            #   -> diagonal blocks also need to be multiplied by 1/2, because
            #      here we additionally have to account for the
            #      bra_ket symmetry!
            if current_bra_ket_sym:  # possible values: +-1 (or 0)
                new_term *= Rational(1, 2)
            # bra_ket symmetry is automaticall accounted for!
            d_tensor = e.expr(AntiSymmetricTensor('d', d_tensor['upper'],
                                                  d_tensor['lower'],
                                                  current_bra_ket_sym))
            sym = d_tensor.terms[0].symmetry()
            symmetrized = symmetrize_keep_pref(new_term, sym)
            symmetrized = simplify(symmetrized)
            removed += symmetrized
        blocks[block] = removed
    return blocks
