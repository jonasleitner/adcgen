from collections.abc import Sequence

from .index_metadata import IndexName
from .utils import split_idx_string


class IndexSpace:
    """
    Tree-like structure to represent index spaces and their connections.

    Parameters
    ----------
    name: str
        The name of the index space, e.g., 'occ'.
    idx_names: Sequence[str | IndexName]
        The names of indices that belong to the space. A single
        name has to consist of a single non-digit character.
        Strings 'ijk' are splitted automatically into ['i', 'j', 'k'].
        The provided names will be sorted lexicographically.
    sort_key: int
        The priority of the current :py:class:`IndexSpace` instance. Defines
        the order in which particular :py:class:`Index` instances are
        sorted.
    parent: IndexSpace, optional
        The parent index space (or its name) the newly created space
        is a subspace of, e.g., the general MO space of which the
        occupied space is a subset.
    """
    __slots__ = ("_name", "_idx_names", "_subspaces", "_sort_key")
    _name: str
    _idx_names: tuple[IndexName, ...]
    _subspaces: tuple["IndexSpace", ...]
    _sort_key: int

    # Cache where IndexSpace instances are stored upon creation.
    # Since we prevent name collisions for space names and their
    # contained index names, the default __eq__ implementation
    # (is comparison) is fine, since the same instance will be
    # used throughout.
    _available_spaces: tuple["IndexSpace", ...] = tuple()

    def __init__(self, name: str, idx_names: Sequence[str | IndexName],
                 sort_key: int, parent: "str | IndexSpace | None" = None
                 ) -> None:
        self._name = name

        if isinstance(idx_names, str):
            idx_names = split_idx_string(idx_names)
        idx_names = tuple(
            IndexName.from_str(idx) if isinstance(idx, str) else idx
            for idx in idx_names
        )
        if any(idx.suffix for idx in idx_names):
            raise ValueError("Index names with a number suffix are not "
                             "allowed as base index names for an IndexSpace. "
                             "Each name is expected to consist of a single "
                             "non-digit character. Imported the index names: "
                             f"{idx_names}.")
        if len(idx_names) != len(set(idx_names)):
            raise ValueError(f"Duplicate found in idx_names: {idx_names}")
        self._idx_names = tuple(sorted(idx_names, key=IndexName.sort_key))

        self._sort_key = sort_key

        self._subspaces = tuple()
        if parent is not None:
            if isinstance(parent, str):
                parent = self.get(parent)
            parent._subspaces += (self,)

        self._register_index_space(self)

    @classmethod
    def _register_index_space(cls, index_space: "IndexSpace") -> None:
        for space in cls._available_spaces:
            if space.name == index_space.name:
                raise ValueError("The name of the new index space "
                                 f"{index_space} is not unique. The index "
                                 f"space {space} shares the same name.")
            if any(name in space.idx_names for name in index_space.idx_names):
                raise ValueError("The index names of the new index space "
                                 f"{index_space} are not unique. At least 1 "
                                 "name is also used in index space "
                                 f"{space}.")
        cls._available_spaces += (index_space,)

    @classmethod
    def get(cls, name: str) -> "IndexSpace":
        """Obtain an IndexSpace instance according to its name."""
        for space in cls._available_spaces:
            if space.name == name:
                return space
        raise ValueError(f"An IndexSpace with name {name} is not available.")

    @classmethod
    def get_by_index_name(cls, idx_name: str | IndexName) -> "IndexSpace":
        """
        Obtain the :py:class:`IndexSpace` to which the provided index name
        belongs.
        """
        # NOTE: this assumes that index names are unique and not shared
        # between spaces. This is ensured in _register_index_space.
        if isinstance(idx_name, str):
            idx_name = IndexName.from_str(idx_name)
        for space in cls._available_spaces:
            if space.is_valid_index_name(idx_name):
                return space
        raise ValueError(f"An IndexSpace for the index name {idx_name} "
                         "is not available.")

    @classmethod
    def remove_from_cache(cls, space: "IndexSpace") -> None:
        """
        Removes the given IndexSpace instance from the instance cache
        allowing e.g. the construction of a new index space with the
        name of the removed space.
        """
        # Should be mostly for testing purposes
        remaining_spaces = tuple(
            sp for sp in cls._available_spaces if sp is not space
        )
        if len(remaining_spaces) == len(cls._available_spaces):
            raise ValueError("Can't remove the given index space, since "
                             "it is not available in the cache anymore. Maybe"
                             f" it was already invalidated?\nSpace: {space}.")
        elif len(remaining_spaces) != len(cls._available_spaces) - 1:
            raise RuntimeError("More than one index space was removed from "
                               "the cache. Somehow the cache was in a bad "
                               f"state.\nCache: {cls._available_spaces}\n"
                               f"Removing: {space}\n"
                               f"Remaining: {remaining_spaces}")
        cls._available_spaces = remaining_spaces

    @classmethod
    def erase_cache(cls) -> None:
        """
        Clears the index space cache removing all previously created
        :py:class:`IndexSpace`s. Note that even if the spaces
        where recreated afterwards they will not be identified
        to be equal to the previous ones.
        """
        cls._available_spaces = tuple()

    @property
    def name(self) -> str:
        return self._name

    @property
    def idx_names(self) -> tuple[IndexName, ...]:
        return self._idx_names

    @property
    def subspaces(self) -> tuple["IndexSpace", ...]:
        return self._subspaces

    def sort_key(self) -> int:
        return self._sort_key

    def is_subspace_of(self, other: "IndexSpace") -> bool:
        """
        Returns True when the current :py:class:`IndexSpace` is a subspace
        of the other :py:class:`IndexSpace` - or of one of its subspaces.
        """
        return (
            self in other.subspaces or
            any(self.is_subspace_of(space) for space in other.subspaces)
        )

    def is_valid_index_name(self, name: IndexName | str) -> bool:
        """
        Verifies that the given index is a valid name for the
        :py:class:`IndexSpace`.
        """
        if isinstance(name, str):
            name = IndexName.from_str(name)
        # the suffix is not relevant for assigning index names to a
        # space, i.e., i and i42 both belong to the same space.
        # -> only compare the base
        return any(name.base == idx.base for idx in self.idx_names)

    def __str__(self) -> str:
        return (
            f"IndexSpace(name={self.name}, idx_names={self.idx_names}, "
            f"subspaces={tuple(s.name for s in self.subspaces)})"
        )

    def __repr__(self) -> str:
        return self.__str__()
