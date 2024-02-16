from .expr_container import Expr
from .misc import Inputerror
from .indices import idx_sort_key, get_symbols, order_substitutions
from .simplify import simplify

from itertools import product


def integrate_spin(expr: Expr, target_idx: str, target_spin: str) -> Expr:
    """Transform an expression from spin to spatial orbitals by integrating
       over the spin, i.e., a spin is attached to all indices in the
       expression."""

    if not isinstance(expr, Expr):
        raise Inputerror(f"Expr needs to be provided as {Expr}. Found {expr}")
    target_idx = get_symbols(target_idx)
    if len(target_idx) != len(target_spin):
        raise Inputerror(f"Spin {target_spin} and indices {target_idx} are "
                         "not compatible.")
    sorted_target = tuple(sorted(target_idx, key=idx_sort_key))
    # - assign the target indices to a spin
    target_idx_spin_map = {}
    for idx, spin in zip(target_idx, target_spin):
        if idx in target_idx_spin_map and target_idx_spin_map[idx] != spin:
            raise ValueError(f"The index {idx} can not be assigned to alpha "
                             "and beta spin simultaneously.")
        target_idx_spin_map[idx] = spin

    result = Expr(0, **expr.assumptions)
    for term in expr.terms:
        # - ensure that the term has matching target indices
        if term.target != sorted_target:
            raise ValueError(f"Target indices of {term} {term.target} dont "
                             f"match the desired target indices {target_idx}")
        # - ensure that no index in the term is holding a spin
        term_indices = set(term.idx)
        if any(s.spin for s in term_indices):
            raise ValueError("The function assumes that the input expression "
                             "is expressed in terms of spin orbitals. Found "
                             f"a spatial orbital in term {term}.")
        # - go through all objects in the term and get the allowed spin
        #   blocks of all tensors and deltas contained in the term
        #   filtering out those spin blocks that are in contradiction to the
        #   desired spin of the target indices
        term_vanishes = False
        term_spin_idx_maps = []
        for obj in term.objects:
            allowed_blocks = obj.allowed_spin_blocks
            if allowed_blocks is None:  # hit a Polynom or Prefactor
                continue
            obj_idx = obj.idx
            obj_spin_idx_maps = []
            for block in allowed_blocks:
                valid = True
                idx_map = {"a": set(), "b": set()}
                for spin, idx in zip(block, obj_idx):
                    if idx in target_idx_spin_map and \
                            spin != target_idx_spin_map[idx]:
                        valid = False
                        break
                    else:
                        idx_map[spin].add(idx)
                if not valid:
                    continue
                if idx_map["a"] & idx_map["b"]:
                    raise ValueError("Found invalid allowed spin block "
                                     f"{block} for {obj}.")
                obj_spin_idx_maps.append(idx_map)
            if not obj_spin_idx_maps:
                term_vanishes = True
                break
            term_spin_idx_maps.append(obj_spin_idx_maps)
        if term_vanishes:
            continue

        # - form all unique valid combinations of idx_maps while checking
        #   for contradictions
        combinations = []
        for tensor_spin_idx_maps in term_spin_idx_maps:
            if not combinations:  # initialize combinations
                combinations.extend(tensor_spin_idx_maps)
                continue
            old_combinations = combinations.copy()
            combinations.clear()
            for idx_map, addition in \
                    product(old_combinations, tensor_spin_idx_maps):
                # ensure that there are no contradictions
                if idx_map["a"] & addition["b"] or \
                        idx_map["b"] & addition["a"]:
                    continue
                combined_map = {"a": idx_map["a"] | addition["a"],
                                "b": idx_map["b"] | addition["b"]}
                # we only need unique variants -> remove duplicates
                if any(d == combined_map for d in combinations):
                    continue
                combinations.append(combined_map)
            # it was not possible to find a single valid combination
            # -> the term should vanish for the given target indices
            if not combinations:
                term_vanishes = True
                break
        if term_vanishes:
            continue

        # - iterate over the unique combinations, replace the spin orbitals
        #   by the corresponding spatial orbitals (assign a spin to the
        #   indices) and add the corresponding terms to the result.
        #   Thereby, ensure that all indices have a spin assigned and
        #   try to assign a spin for not yet assigned indices:
        #   if they are are target indices -> use the input spin
        #   if they are contracted -> generate a variant for both spins

        contribution = Expr(0, **expr.assumptions)
        for idx_map in combinations:
            assigned_indices = idx_map["a"] | idx_map["b"]
            missing_indices = [idx for idx in term_indices
                               if idx not in assigned_indices]
            if missing_indices:  # some indices don't have a spin assigned yet!
                missing_contracted = []
                for idx in missing_indices:
                    spin = target_idx_spin_map.get(idx, None)
                    if spin is not None:  # is a target index -> just add
                        idx_map[target_idx_spin_map[idx]].add(idx)
                    else:  # is a contracted index -> need to try both spins
                        missing_contracted.append(idx)
                if missing_contracted:
                    # construct all the different variants where all missing
                    # contracted indices are assigned to either a or b spin
                    variants = []
                    for var in product("ab", repeat=len(missing_contracted)):
                        complete_variant = idx_map.copy()
                        for spin, idx in zip(var, missing_contracted):
                            complete_variant[spin].add(idx)
                        variants.append(complete_variant)
                else:
                    variants = [idx_map]
            else:
                variants = [idx_map]
            for var in variants:
                sub = {}
                for spin in ["a", "b"]:
                    old = list(var[spin])
                    if not old:
                        continue
                    names = "".join(s.name for s in old)
                    spins = "".join(spin for _ in range(len(old)))
                    new = get_symbols(names, spins)
                    sub.update({o: n for o, n in zip(old, new)})
                contribution += term.subs(order_substitutions(sub))
        # TODO: if we simplify the result it will throw an error for any
        # polynoms or denominators. Should we skip the simplification altough
        # we currently don't treat polynoms correctly in this function
        # since their allowed_spin_blocks are not considered.
        result += simplify(contribution)
    return result


def allowed_spin_blocks(expr: Expr, target_idx: str) -> tuple[str]:
    """Determines all allowed spin blocks that can be populated by the
       provided expression. Thereby, it is assumed that all allowed spin
       blocks of tensors in the expression are either known or can be
       determined, i.e., this only works for closed expressions."""

    if not isinstance(expr, Expr):
        raise Inputerror(f"Expr needs to be provided as {Expr}. Found {expr}")

    target_idx = get_symbols(target_idx)
    sorted_target = tuple(sorted(target_idx, key=idx_sort_key))

    # - determine all possible spin blocks
    spin_blocks = ["".join(b) for b in product("ab", repeat=len(target_idx))]
    spin_blocks_to_check = [i for i in range(len(spin_blocks))]

    allowed_blocks = set()
    for term in expr.terms:
        # - ensure that the term has matching target indices
        if term.target != sorted_target:
            raise ValueError(f"Target indices of {term} {term.target} dont "
                             f"match the desired target indices {target_idx}")
        # - extract the allowed blocks for all tensors and initialize
        #   index maps to relate indices to a spin
        term_idx_maps = []
        for obj in term.objects:
            allowed_object_blocks = obj.allowed_spin_blocks
            if allowed_object_blocks is None:  # hit a Polynom or Prefactor
                continue
            obj_indices = obj.idx
            n_target = len([idx for idx in obj_indices if idx in target_idx])
            object_idx_maps = []
            for block in allowed_object_blocks:
                idx_map = {}
                for spin, idx in zip(block, obj_indices):
                    if idx in idx_map and idx_map[idx] != spin:
                        raise ValueError("Found invalid allowed spin block "
                                         f"{block} for {obj}.")
                    idx_map[idx] = spin
                object_idx_maps.append(idx_map)
            term_idx_maps.append((object_idx_maps, n_target))
        # - sort the allowed_tensor_blocks such that tensors with a high
        #   number of target indices are preferred
        term_idx_maps = sorted(term_idx_maps,
                               key=lambda tpl: tpl[1], reverse=True)

        term_indices = set(term.idx)
        blocks_to_remove = set()
        for block_i in spin_blocks_to_check:
            block = spin_blocks[block_i]
            if block in allowed_blocks:
                blocks_to_remove.add(block_i)
                continue
            valid_block = True

            # - assign the target indices to a spin
            target_spin = {}
            for spin, idx in zip(block, target_idx):
                # in case we have target indices iiab only spin blocks
                # aaxx or bbxx are valid
                if idx in target_spin and target_spin[spin] != spin:
                    valid_block = False
                    break
                target_spin[idx] = spin
            if not valid_block:
                continue

            # - remove all object spin blocks that are in contradiction to the
            #   current spin block
            relevant_term_spin_idx_maps = []
            for tensor_idx_maps, _ in term_idx_maps:
                relevant_object_spin_idx_maps = []
                for idx_map in tensor_idx_maps:
                    # are all target idx compatible with the block?
                    if any(spin != idx_map[t_idx]
                           for t_idx, spin in target_spin.items()
                           if t_idx in idx_map):
                        continue
                    spin_idx_map = {"a": set(), "b": set()}
                    for idx, spin in idx_map.items():
                        spin_idx_map[spin].add(idx)
                    relevant_object_spin_idx_maps.append(spin_idx_map)
                # the object has not a single allowed spin block that is
                # compatible to the currently probed block
                if not relevant_object_spin_idx_maps:
                    valid_block = False
                    break
                relevant_term_spin_idx_maps.append(
                    relevant_object_spin_idx_maps
                )
            # at least 1 object has no compatible allowed spin block
            # -> the current term can not contribute to the current block
            if not valid_block:
                continue

            # - try to find a valid combination of the remaining spin blocks
            spin_idx_map = {"a": set(), "b": set()}
            if not _has_valid_combination(relevant_term_spin_idx_maps, 0,
                                          spin_idx_map):
                continue
            # - verify the result:
            #       ensure that all indices are assigned
            #       the target indices have the desired spin
            #       there is no intersection between the different spins
            if spin_idx_map["a"] & spin_idx_map["b"]:
                raise RuntimeError("Indices are assigned to alpha and beta "
                                   f"simultaneously in term {term}: ",
                                   spin_idx_map)
            if term_indices ^ (spin_idx_map["a"] | spin_idx_map["b"]):
                raise RuntimeError("Not all indices were assigned to a spin: "
                                   f"{term_indices} -> {spin_idx_map}")
            if any(idx not in spin_idx_map[spin]
                   for idx, spin in target_spin.items()):
                raise RuntimeError("Target index has wrong spin. Desired: "
                                   f"{target_spin}. Found: {spin_idx_map}.")
            # everything should be fine!
            # also add the 'inverse' block to the allowed blocks
            allowed_blocks.add(block)
            allowed_blocks.add("".join("a" if spin == "b" else "b"
                                       for spin in block))
            blocks_to_remove.add(block_i)
        # blocks that have been found dont need to be checked again
        spin_blocks_to_check = [i for i in spin_blocks_to_check
                                if i not in blocks_to_remove]
    return tuple(sorted(allowed_blocks))


def _has_valid_combination(tensor_idx_maps: list, current_pos: int,
                           variant: dict):
    """Tries to recursively assign all indices to a spin without introducing
       contradictions."""

    for idx_map in tensor_idx_maps[current_pos]:
        # look for any contradictions
        if idx_map["a"] & variant["b"] or idx_map["b"] & variant["a"]:
            continue
        # compute the indices which are added to remove them later again
        # if necessary
        addition = {"a": tuple(idx for idx in idx_map["a"]
                               if idx not in variant["a"]),
                    "b": tuple(idx for idx in idx_map["b"]
                               if idx not in variant["b"])}
        variant["a"].update(idx_map["a"])
        variant["b"].update(idx_map["b"])
        if len(tensor_idx_maps) == current_pos + 1:  # we are done!!
            return True
        # recurse further and try to complete
        if _has_valid_combination(tensor_idx_maps, current_pos+1, variant):
            return True
        # could not complete -> revert the addition and continue looping
        variant["a"].difference_update(addition["a"])
        variant["b"].difference_update(addition["b"])
    # could not add anything to the variant
    return False
