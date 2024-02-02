from .expr_container import Expr
from .indices import get_symbols

from itertools import product


def allowed_spin_blocks(expr: Expr, target_idx: str):
    """Determines all allowed spin blocks that can be populated by the
       provided expression. Thereby, it is assumed that all allowed spin
       blocks of tensors in the expression are either known or can be
       determined, i.e., this only works for closed expressions."""

    target_idx = get_symbols(target_idx)

    # - determine all possible spin blocks
    spin_blocks = ["".join(b) for b in product("ab", repeat=len(target_idx))]
    spin_blocks_to_check = [i for i in range(len(spin_blocks))]

    allowed_blocks = set()
    for term in expr.terms:
        # - extract the allowed blocks for all tensors and initialize
        #   index maps to relate indices to a spin
        allowed_tensor_blocks = []
        for tensor in term.tensors:
            tensor_idx = tensor.idx
            n_target = len([idx for idx in tensor_idx if idx in target_idx])
            tensor_idx_maps = []
            for block in tensor.allowed_spin_blocks:
                idx_map = {}
                for spin, idx in zip(block, tensor_idx):
                    if idx in idx_map and idx_map[idx] != spin:
                        raise ValueError("Found invalid allowed spin block "
                                         f"{block} for tensor {tensor}.")
                    idx_map[idx] = spin
                tensor_idx_maps.append(idx_map)
            allowed_tensor_blocks.append((tensor_idx_maps, n_target))
        # - sort the allowed_tensor_blocks such that tensors with a high
        #   number of target indices are preferred
        allowed_tensor_blocks = sorted(allowed_tensor_blocks,
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
                target_spin[idx] = spin
            if not valid_block:
                continue

            # - remove all tensor blocks that are in contradiction to the
            #   current spin block
            relevant_tensor_blocks = []
            for tensor_idx_maps, _ in allowed_tensor_blocks:
                relevant_tensor_data = []
                for idx_map in tensor_idx_maps:
                    # are all target idx compatible with the block?
                    if any(spin != idx_map[t_idx]
                           for t_idx, spin in target_spin.items()
                           if t_idx in idx_map):
                        continue
                    inverted_idx_map = {"a": set(), "b": set()}
                    for idx, spin in idx_map.items():
                        inverted_idx_map[spin].add(idx)
                    relevant_tensor_data.append(inverted_idx_map)
                # the tensor does not have a single allowed spin block that is
                # compatible to the currently tested block
                if not relevant_tensor_data:
                    valid_block = False
                relevant_tensor_blocks.append(relevant_tensor_data)
            # at least 1 tensor has no compatible allowed spin block
            # -> the current term can not contribute to the current block
            if not valid_block:
                continue

            # - try to find a valid combination of the remaining spin blocks
            spin_idx_map = {"a": set(), "b": set()}
            if not _has_valid_combination(relevant_tensor_blocks, 0,
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
    return sorted(allowed_blocks)


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
