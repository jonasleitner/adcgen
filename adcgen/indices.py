from collections.abc import Sequence, Collection, Mapping
from typing import Any, TypeGuard, TYPE_CHECKING

from sympy import Dummy, Tuple

from .misc import Inputerror, Singleton

if TYPE_CHECKING:
    from .symmetry import Permutation


class Index(Dummy):
    """
    Represents an Index. Wrapper implementation around the 'sympy.Dummy'
    class, which means Index("x") != Index("x").
    Important assumptions:
    - below_fermi: The index represents an occupied orbital.
    - above_fermi: The index represents a virtual orbital.
    - core: The index represents a core orbital.
    - alpha: The index represents an alpha (spatial) orbital.
    - beta: The index represents a beta (spatial) orbital.
    """

    @property
    def spin(self) -> str:
        """
        The spin of the index. An empty string is returned if no spin is
        defined.
        """
        if self.assumptions0.get("alpha"):
            return "a"
        elif self.assumptions0.get("beta"):
            return "b"
        else:
            return ""

    @property
    def space(self) -> str:
        """
        The space to which the index belongs (core/occupied/virtual/general).
        """
        if self.assumptions0.get("below_fermi"):
            return "occ"
        elif self.assumptions0.get("above_fermi"):
            return "virt"
        elif self.assumptions0.get("core"):
            return "core"
        elif self.assumptions0.get("aux"):
            return "aux"
        else:
            return "general"

    @property
    def space_and_spin(self) -> tuple[str, str]:
        """Returns space and spin of the Index."""
        return self.space, self.spin

    def __str__(self):
        spin = self.spin
        return f"{self.name}_{spin}" if spin else self.name

    def __repr__(self) -> str:
        return self.__str__()

    def _sympystr(self, printer):
        _ = printer
        return self.__str__()

    def _latex(self, printer) -> str:
        _ = printer
        ret = self.name
        if (spin := self.spin):
            spin = "alpha" if spin == "a" else "beta"
            ret += "_{\\" + spin + "}"
        return ret


class Indices(metaclass=Singleton):
    """
    Manages the indices used thoughout the package and ensures that
    only a single class instance exists for each index.
    This is necessary because the 'equality' operator is essentially replaced
    by the 'is' operator for the indices: Index("x") != Index("x").
    """
    # the valid spaces with their corresponding associated index names
    base = {
        "occ": "ijklmno", "virt": "abcdefgh", "general": "pqrstuvw",
        "core": "IJKLMNO", "aux": "PQRST"
    }
    # the valid spins
    spins = ("", "a", "b")
    # the generation of generic indices starts with e.g., "i3" for occupied
    # indices. Therefore, the indices "i", "i1" and "i2" are only available
    # through a specific request to get_indices
    _initial_counter = 3

    def __init__(self) -> None:
        # dict that holds all symbols that have been created previously.
        # structure: {space: {spin: {name: symbol}}}
        self._symbols: dict[str, dict[str, dict[str, Index]]] = {}
        # dict that holds the automatically generated generic index names
        # (strings) that have not been used yet.
        # structure: {space: {spin: [names]}}
        self._generic_indices: dict[str, dict[str, list[str]]] = {}
        # dict holding the counter (the number appended to the index names)
        # for the generation of generic indices.
        self._counter: dict[str, dict[str, int]] = {}
        # initialize the data structures
        for space in self.base:
            self._symbols[space] = {}
            self._generic_indices[space] = {}
            self._counter[space] = {}
            for spin in self.spins:
                self._symbols[space][spin] = {}
                self._generic_indices[space][spin] = []
                self._counter[space][spin] = self._initial_counter

    def is_cached_index(self, index: Index) -> bool:
        """
        Whether an index was generated with the 'Indices' class and is thus
        cached in the class.
        """
        cached_symbol = (
            self._symbols[index.space][index.spin].get(index.name, None)
        )
        return cached_symbol is index

    def _gen_generic_idx(self, space: str, spin: str = ""):
        """
        Generated the next 'generation' of generic indices, i.e. extends
        _generic_indices by incrementing the _counter attached to the index
        base.
        """
        # generate the not used indices of the new generation
        counter = str(self._counter[space][spin])
        used_names = self._symbols[space][spin]
        new_idx = [idx + counter for idx in self.base[space]
                   if idx + counter not in used_names]
        # extend the available generic indices and increment the counter
        self._generic_indices[space][spin].extend(new_idx)
        self._counter[space][spin] += 1

    def get_indices(self, indices: Sequence[str],
                    spins: Sequence[str] | None = None
                    ) -> dict[tuple[str, str], list[Index]]:
        """
        Obtain the Indices for the provided string of names.

        Parameters
        ----------
        indices : Sequence[str]
            The names of the indices as a single string that is split
            automatically as "ij21kl3" -> i, j21, k, l3.
            They can also provided as list/tuple of index names.
        spins : Sequence[str] | None, optional
            The spins of the indices as a single string, e.g., "aaba"
            to obtain four indices with spin: alpha, alpha, beta, alpha.
            Since no spin is represented by the empty string, it is only
            possible to obtain indices with and without spin when the spins
            are provided as list/tuple.

        Returns
        -------
        dict
            key: tuple containing the space and spin of the indices.
            value: list containing the indices in the order the indices are
                   provided in the input.
        """
        # split the string of indices
        if isinstance(indices, str):
            indices = split_idx_string(indices)
        if spins is None:
            spins = ["" for _ in range(len(indices))]
        if len(indices) != len(spins):
            raise Inputerror(f"Indices {indices} and spins {spins} do not "
                             "match.")

        ret: dict[tuple[str, str], list[Index]] = {}
        for idx, spin in zip(indices, spins):
            space = index_space(idx)
            key = (space, spin)
            if key not in ret:
                ret[key] = []
            # check the cache for the index
            symbol = self._symbols[space][spin].get(idx, None)
            if symbol is not None:
                ret[key].append(symbol)
                continue
            # not found in cache
            # -> create new symbol and cache it
            symbol = self._new_symbol(idx, space, spin)
            self._symbols[space][spin][idx] = symbol
            ret[key].append(symbol)
            # -> also remove it from the available generic indices
            try:
                self._generic_indices[space][spin].remove(idx)
            except ValueError:
                continue
        return ret

    def get_generic_indices(self, **kwargs
                            ) -> dict[tuple[str, str], list[Index]]:
        """
        Request indices with unique names that have not been used in the
        current run of the program. Easy way to ensure that contracted indices
        do not appear anywhere else in a term.
        Indices can be requested using the syntax "{space}_{spin}",
        where spin is optional. For instance, occupied indices without
        spin can be obtained with "occ=5", or "occ_a=5" for occupied indices
        with alpha spin.

        Returns
        -------
        dict
            key: tuple of space and spin of the indices.
            value: list containing the indices.
        """

        ret = {}
        for key, n in kwargs.items():
            if n == 0:
                continue
            key = key.split("_")
            if len(key) == 2:
                space, spin = key
            elif len(key) == 1:
                space, spin = key[0], ""
            else:
                raise Inputerror(f"{'_'.join(key)} is not valid for "
                                 "requesting generic indices.")
            # generate generic index names until we have enough
            while n > len(self._generic_indices[space][spin]):
                self._gen_generic_idx(space, spin)
            # get the indices
            idx = self._generic_indices[space][spin][:n]
            spins = tuple(spin for _ in range(n))
            ret.update(self.get_indices(idx, spins))
        return ret

    def _new_symbol(self, name: str, space: str, spin: str) -> Index:
        """Creates a new symbol with the defined name, space and spin."""
        assumptions = {}
        if space == "occ":
            assumptions["below_fermi"] = True
        elif space == "virt":
            assumptions["above_fermi"] = True
        elif space == "core":
            assumptions["core"] = True
        elif space == "aux":
            assumptions["aux"] = True
        elif space != "general":
            raise ValueError(f"Invalid space {space}")
        if spin:
            if spin == "a":
                assumptions["alpha"] = True
            elif spin == "b":
                assumptions["beta"] = True
            else:
                raise ValueError(f"Invalid spin {spin}.")
        return Index(name, **assumptions)


def index_space(idx: str) -> str:
    """Returns the space an index belongs to (occ/virt/general)."""
    for sp, idx_string in Indices.base.items():
        if idx[0] in idx_string:
            return sp
    raise Inputerror(f"Could not assign the index {idx} to a space.")


def sort_idx_canonical(idx: Index | Any):
    """Use as sort key to to bring indices in canonical order."""
    if isinstance(idx, Index):
        # - also add the hash here for wicks, where multiple i are around
        # - we have to map the spaces onto numbers, since in adcman and adcc
        # the ordering o < c < v is used for the definition of canonical blocks
        space_keys = {"g": 0, "o": 1, "c": 2, "v": 3, "a": 4}
        return (space_keys[idx.space[0]],
                idx.spin,
                int(idx.name[1:]) if idx.name[1:] else 0,
                idx.name[0],
                hash(idx))
    else:  # necessary for subs to work correctly with simultaneous=True
        return (-1, "", 0, str(idx), hash(idx))


def split_idx_string(str_tosplit: str) -> list[str]:
    """
    Splits an index string of the form 'ij12a3b' in a list ['i','j12','a3','b']
    """
    splitted = []
    temp = []
    for i, idx in enumerate(str_tosplit):
        temp.append(idx)
        try:
            if str_tosplit[i+1].isdigit():
                continue
            else:
                splitted.append("".join(temp))
                temp.clear()
        except IndexError:
            splitted.append("".join(temp))
    return splitted


def n_ov_from_space(space_str: str):
    """
    Number of required occupied and virtual indices required for the given
    exictation space, e.g., "pph" -> 2 virtual and 1 occupied index.
    """
    return {"occ": space_str.count("h"), "virt": space_str.count("p")}


def generic_indices_from_space(space_str: str) -> list[Index]:
    """
    Constructs generic indices from a given space string (e.g. 'pphh').
    Thereby, occupied indices are listed before virtual indices!
    """
    generic_idx = Indices().get_generic_indices(**n_ov_from_space(space_str))
    assert len(generic_idx) <= 2  # only occ and virt
    occ = generic_idx.get(("occ", ""), [])
    occ.extend(generic_idx.get(("virt", ""), []))
    occ.extend(generic_idx.get(("core", ""), []))
    occ.extend(generic_idx.get(("aux", ""), []))
    return occ


def repeated_indices(idx_a: str, idx_b: str) -> bool:
    """Checks whether both index strings share an index."""
    split_a = split_idx_string(idx_a)
    split_b = split_idx_string(idx_b)
    return any(i in split_b for i in split_a)


def get_lowest_avail_indices(n: int, used: Collection[str], space: str
                             ) -> list[str]:
    """
    Return the names of the n lowest available indices belonging to the desired
    space.

    Parameters
    ----------
    n : int
        The number of available indices.
    used : Collection[str]
        The names of the indices that are already in use.
    space : str
        The space (occ/virt/general) to which the indices belong.
    """
    # generate idx pool to pick the lowest indices from
    base = Indices.base[space]
    idx = list(base)
    required = len(used) + n  # the number of indices present in the term
    suffix = 1
    while len(idx) < required:
        idx.extend(s + str(suffix) for s in base)
        suffix += 1
    # remove the already used indices (that are not available anymore)
    # and return the first n elements of the resulting list
    return [s for s in idx if s not in used][:n]


def get_symbols(indices: Sequence[str] | Index | Sequence[Index],
                spins: Sequence[str] | None = None) -> list[Index]:
    """
    Wrapper around the Indices class to initialize 'Index' instances with the
    provided names and spin.

    Parameters
    ----------
    indices : Index | Sequence[str] | Sequence[Index]
        The names of the indices to generate. If they are already instances
        of the 'Index' class we do nothing.
    spins : Sequence[str] | None, optional
        The spin of the indices, e.g., "aab" to obtain 3 indices with
        alpha, alpha and beta spin.
    """

    if not indices:  # empty string/list
        return []
    elif isinstance(indices, Index):  # a single symbol is not iterable
        return [indices]
    elif _is_index_sequence(indices):
        return indices if isinstance(indices, list) else list(indices)
    # we actually have to do something
    # -> prepare indices and spin
    if isinstance(indices, str):
        indices = split_idx_string(indices)
    if spins is None:
        spins = ["" for _ in range(len(indices))]
    # at this point we should have a list/tuple of strings
    # construct the indices
    assert _is_str_sequence(indices)
    symbols = Indices().get_indices(indices, spins)
    # and return them in the order they were provided in the input
    for val in symbols.values():
        val.reverse()
    ret = [symbols[(index_space(idx), spin)].pop()
           for idx, spin in zip(indices, spins)]
    assert not any(symbols.values())  # ensure we consumed all indices
    return ret


def order_substitutions(subsdict: dict[Index, Index]
                        ) -> list[tuple[Index, Index]]:
    """
    Order substitutions such that only a minial amount of intermediate
    indices is required when the substitutions are executed one after another
    and the usage of the 'simultaneous=True' option of sympys 'subs' method.
    Adapted from the 'substitute_dummies' function defined in
    'sympy.physics.secondquant'.
    """

    subs = []
    final_subs = []
    for o, n in subsdict.items():
        if o is n:  # indices are identical -> nothing to do
            continue
        # the new index is substituted by another index
        if (other_n := subsdict.get(n, None)) is not None:
            if other_n in subsdict:
                # i -> j / j -> i
                # temporary variable is needed
                p = Index('p')
                subs.append((o, p))
                final_subs.append((p, n))
            else:
                # i -> j / j -> k
                # in this case it is sufficient to do the i -> j substitution
                # after the j -> k substitution, but before temporary variables
                # are resubstituted again.
                final_subs.insert(0, (o, n))
        else:
            subs.append((o, n))
    subs.extend(final_subs)
    return subs


def minimize_tensor_indices(
        tensor_indices: Sequence[Index],
        target_idx_names: Mapping[tuple[str, str], Collection[str]]
        ) -> "tuple[tuple[Index, ...], tuple[Permutation, ...]]":
    """
    Minimizes the indices on a tensor using the lowest available indices that
    are no target indices.

    Parameters
    ----------
    tensor_indices : Sequence[Index]
        The indices of the tensor.
    target_idx : dict[tuple[str, str], Collection[str]]
        The names of target indices sorted by their space and spin with
        key = (space, spin).

    Returns
    -------
    tuple
        The minimized indices and the list of index permutations that was
        applied to reach this minimized state.
    """
    from .symmetry import Permutation, PermutationProduct

    for target in target_idx_names.values():
        if not all(isinstance(s, str) for s in target):
            raise TypeError("Target indices need to be provided as string.")

    tensor_idx: list[Index] = list(tensor_indices)
    n_unique_indices: int = len(set(tensor_idx))
    minimal_indices: dict[tuple[str, str], list[Index]] = {}
    permutations: list[Permutation] = []
    minimized = set()
    for s in tensor_idx:
        if s in minimized:
            continue
        idx_key = s.space_and_spin
        # target indices of the corresponding space
        space_target = target_idx_names.get(idx_key, [])
        # index is a target idx -> keep as is
        if s.name in space_target:
            minimized.add(s)
            continue
        # generate minimal indices for the corresponding space and spin
        if idx_key not in minimal_indices:
            space, spin = idx_key
            min_names = get_lowest_avail_indices(n_unique_indices,
                                                 space_target, space)
            if spin:
                spins = spin * n_unique_indices
            else:
                spins = None
            min_symbols = get_symbols(min_names, spins)
            min_symbols.reverse()
            minimal_indices[idx_key] = min_symbols

        # get the lowest available index for the corresponding space
        min_s = minimal_indices[idx_key].pop()
        minimized.add(min_s)
        if s is min_s:  # s is already the lowest available index
            continue
        # found a lower index
        # -> permute tensor indices and append permutation to permutations
        #    list
        perm = {s: min_s, min_s: s}
        for i, other_s in enumerate(tensor_idx):
            tensor_idx[i] = perm.get(other_s, other_s)
        permutations.append(Permutation(s, min_s))
    return tuple(tensor_idx), PermutationProduct(*permutations)


################################################
# Some TypeGuards to make the type checker happy
###############################################
def _is_index_sequence(sequence: Sequence) -> TypeGuard[Sequence[Index]]:
    return all(isinstance(s, Index) for s in sequence)


def _is_index_tuple(sequence: tuple | Tuple) -> TypeGuard[tuple[Index, ...]]:
    return all(isinstance(s, Index) for s in sequence)


def _is_str_sequence(sequence: Sequence) -> TypeGuard[Sequence[str]]:
    return (
        isinstance(sequence, str) or all(isinstance(s, str) for s in sequence)
    )


_ = _is_index_tuple
