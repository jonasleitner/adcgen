from collections.abc import Sequence

from ..misc import Singleton
from .index import Index
from .index_metadata import IndexMetaData, IndexName, Spin
from .index_space import IndexSpace
from .utils import split_idx_string


class Indices(metaclass=Singleton):
    """
    Manages and caches Index instances.
    Since the equality operator is implemented as the 'is' operator, i.e.,
    Index("i", "occ") != Index("i", "occ"),
    the caching is required to generate visibly interpretable
    expressions, while still offering the possibility to easily
    generate temporary unique indices with arbitrary name.
    """
    __slots__ = ("_index_cache", "_generic_counter")
    # cache for the Index instances
    #                  (space, spin)     name       subscripts
    _cache_key = tuple[IndexMetaData, IndexName, tuple[Index, ...]]
    _index_cache: dict[_cache_key, Index]
    # storage for the counters for each space and spin.
    _generic_counter: dict[IndexMetaData, int]

    # the generation of generic indices starts with an suffix of 42
    # rendering all indices with a lower suffix (0-42) only available
    # through a specific request to get_indices
    _initial_suffix: int = 42

    def __init__(self) -> None:
        self._index_cache = {}
        self._generic_counter = {}

    def reset(self) -> None:
        """
        Clears the indices cache and resets the counter for generic indices.
        Note that indices created after clearing the cache are not
        comparable to indices that were created before.

        Examples
        --------
        i = Indices().get_indices("i", ["occ"]).pop()
        Indices().reset()
        i = Indices().get_indices("i", ["occ"]).pop()
        i != i
        """
        self._index_cache = {}
        self._generic_counter = {}

    def get_indices(self, names: Sequence[str | IndexName],
                    spaces: Sequence[str | IndexSpace] | None = None,
                    spins: Sequence[str | Spin] | None = None,
                    subscripts: Sequence[Sequence[Index]] | None = None
                    ) -> list[Index]:
        """
        Build or load indices from the cache.

        Parameters
        ----------
        names: Sequence[str | IndexName]
            The names of the desired indices. Note that due to the assignment
            of index names to an :py:class:`IndexSpace` this implicitly
            defines the space of the index. The names can also be provided as
            a single string that is then split automatically.
        spaces: Sequence[str | IndexSpace], optional
            The :py:class:`IndexSpace`s of the desired indices. It is also
            possible to provide the names of the spaces as strings. The
            corresponding index names have to be valid for the corresponding
            :py:class:`IndexSpace`s! By default, the space will be determined
            automatically according to the index names.
        spins: Sequence[str | Spin], optional
            The spins of the desired indices. Valid string representations
            of e.g. the alpha spin are either 'a' or 'alpha'. It is also
            possible to provide the spin as single string of the form
            'abn', which will be interpreted as alpha, beta, none (no spin).
            By default, indices without spin will be created.
        subscripts: Sequence[Sequence[str | Index]], optional
            The sequence of subscripts for the indices to generate.
            If only the names of the subscripts are given as str, the
            subscript indices will be constructed first before creating
            the actual indices. Note that in this case it is not possible to
            create more sophisticated subscripts with e.g. spin.

        Returns
        -------
        list of Index
            The list of indices in the order they were requested.
        """
        # prepare all the input arguments
        if isinstance(names, str):
            names = split_idx_string(names)

        if spaces is None:
            # determine the spaces of the indices according to their names
            spaces = [IndexSpace.get_by_index_name(name) for name in names]
        elif len(names) != len(spaces):
            raise ValueError(f"The number of provided spaces ({len(spaces)}) "
                             "does not match the number of requested indices "
                             f"({len(names)}): names = {names}, "
                             f"spaces = {spaces}.")

        if spins is None:
            spins = [Spin.NONE for _ in range(len(names))]
        elif len(names) != len(spins):
            raise ValueError(f"The number of provided spins ({len(spins)}) "
                             f"does not match the number of requested indices "
                             f"({len(names)}): spins = {spins}, "
                             f"names = {names}.")

        if subscripts is None:
            subscripts = [tuple() for _ in range(len(names))]
        elif len(subscripts) != len(names):
            raise ValueError("Subscripts only provided for "
                             f"{len(subscripts)} indices, while {len(names)}"
                             f" have requested: names = {names}, "
                             f"subscripts = {subscripts}.")

        ret: list[Index] = []
        for name, space, spin, subscript in \
                zip(names, spaces, spins, subscripts, strict=True):
            if isinstance(name, str):
                name = IndexName.from_str(name)
            if isinstance(space, str):
                space = IndexSpace.get(space)
            if isinstance(spin, str):
                spin = Spin.from_str(spin)
            if not isinstance(subscript, tuple):
                subscript = tuple(subscript)
            # construct the key for the cache lookup
            cache_key = (
                IndexMetaData(space=space, spin=spin),
                name, subscript
            )
            index = self._index_cache.get(cache_key, None)
            if index is None:
                # could not find the index in the cache
                # -> construct a new index
                index = Index(
                    name=name, space=space, spin=spin, subscripts=subscript
                )
                self._index_cache[cache_key] = index
            ret.append(index)
        return ret

    def get_generic_indices(self, count: int, space: str | IndexSpace,
                            spin: str | Spin | None = None) -> list[Index]:
        """
        Constructs new, unused indices that have not yet been used in the
        current run of the program.

        Parameters
        ----------
        count: int
            The number of generic indices to construct.
        space: str | IndexSpace
            The index space (or its name) the indices should belong to.
        spin: str | Spin, optional
            The spin of the newly constructed indices.

        Returns
        -------
        list of Index
            The list of newly generated generic indices
        """
        # NOTE: it is technically still possible to accidentally recreat
        # indices that have been constructed with get_generic_indices
        # by later making a request for indices with the appropriate name,
        # space and spin to get_indices.
        # To avoid this we add a counter suffix to generic indices, e.g.,
        # the lowest generic index might be something like 'a42'. So there
        # should be a sufficient number of indices with lower number suffixes
        # to pick from for target indices of expressions.
        if count < 1:
            return []
        # deal with the input args
        if isinstance(space, str):
            space = IndexSpace.get(space)
        if spin is None:
            spin = Spin.NONE
        elif isinstance(spin, str):
            spin = Spin.from_str(spin)
        metadata = IndexMetaData(space=space, spin=spin)
        # generate unused names
        generic_names = []
        counter = self._generic_counter.get(metadata, self._initial_suffix)
        base_names = [name.base for name in space.idx_names]
        while len(generic_names) < count:
            candidates = [
                IndexName(base=base, suffix=counter) for base in base_names
            ]
            # remove all names that have already been constructed before
            generic_names.extend(
                candidate for candidate in candidates
                if (metadata, candidate, tuple()) not in self._index_cache
            )
            counter += 1
        self._generic_counter[metadata] = counter
        return self.get_indices(
            names=generic_names[:count], spaces=[space for _ in range(count)],
            spins=[spin for _ in range(count)]
        )
