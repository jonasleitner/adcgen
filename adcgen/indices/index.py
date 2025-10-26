from collections.abc import Sequence
from typing import Any

from sympy import AtomicExpr, Basic, Pow, S

from .index_metadata import IndexMetaData, IndexName, Spin
from .index_space import IndexSpace


class Index(AtomicExpr):
    """
    Represents an Index.
    Comparison of indices is handled via the 'is' operator, i.e.,
    Index("i", "occ") != Index("i", "occ")
    assuming a space with name "occ" has been created beforehand.

    Parameters
    ----------
    name: str | IndexName
        The name of the index.
    space: str | IndexSpace
        The name of the space the index belongs to.
    spin: str | Spin, optional
        The spin of the index. Valid string representations of e.g. the
        alpha spin are 'a' or 'alpha'. By default the index has no spin.
    subscripts: Sequence[Index]
        Indices that are used as subscripts for the current index.
    """
    __slots__ = ("_name", "_metadata", "_subscripts")
    _name: IndexName
    _metadata: IndexMetaData
    _subscripts: tuple["Index", ...]

    # enables derivates wrt Indices
    _diff_wrt: bool = True  # type: ignore
    # an Index can not be converted to a real number if I understood the
    # docstring on Basic correctly (exactly like Symbol and Dummy)
    is_comparable: bool = False  # type: ignore

    def __new__(cls, name: IndexName | str, space: IndexSpace | str,
                spin: Spin | str = Spin.NONE,
                subscripts: Sequence["Index"] = tuple()) -> "Index":
        instance = super().__new__(cls)

        if isinstance(name, str):
            name = IndexName.from_str(name)
        instance._name = name

        if isinstance(space, str):
            # get the IndexSpace according to its name
            space = IndexSpace.get(space)
        if not space.is_valid_index_name(instance._name):
            raise ValueError(f"The index name {instance._name} is not valid "
                             f"for {space}.")
        if isinstance(spin, str):
            spin = Spin.from_str(spin)
        instance._metadata = IndexMetaData(space=space, spin=spin)

        if not isinstance(subscripts, tuple):
            subscripts = tuple(subscripts)
        instance._subscripts = subscripts
        return instance

    @property
    def name(self) -> IndexName:
        return self._name

    @property
    def space(self) -> IndexSpace:
        return self._metadata.space

    @property
    def spin(self) -> Spin:
        return self._metadata.spin

    @property
    def subscripts(self) -> tuple["Index", ...]:
        return self._subscripts

    @property
    def metadata(self) -> tuple[IndexMetaData, ...]:
        """
        The metadata of the index, e.g., space and spin.
        The index name is explicitly excluded.
        The metadata of subscripts is recursively appended using depth
        first tree traversal.
        """
        if not self.subscripts:
            return (self._metadata,)

        ret = [self._metadata]
        for idx in self.subscripts:
            ret.extend(idx.metadata)
        return tuple(ret)

    def sort_key(self, order=None):  # type: ignore
        """Used as sort key to bring indices in canonical order."""
        # structure taken from the sort_key implementations on
        # Symbol/Dummy/Atom
        _ = order
        return (
            self.class_key(), (1, (
                self._metadata.sort_key(),  # space, spin, ...
                self.name.sort_key(),  # suffix, base
                tuple(idx.sort_key(order=order) for idx in self.subscripts),
                hash(self),  # in case the data and name are identical
            )),
            S.One.sort_key(), S.One
        )

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __str__(self) -> str:
        spin = self.spin
        subscripts = self.subscripts
        ret = str(self.name)
        if spin:
            ret += f"^{{{spin.name.lower()}}}"
        if subscripts:
            ret += f"_{{{"".join(str(idx) for idx in subscripts)}}}"
        return ret

    def __repr__(self) -> str:
        return self.__str__()

    def _latex(self, printer) -> str:
        _ = printer
        spin = self.spin
        subscripts = self.subscripts
        ret = str(self.name)
        if spin:
            ret += f"^{{\\{spin.name.lower()}}}"
        if subscripts:
            ret += f"_{{{"".join(str(idx) for idx in subscripts)}}}"
        return ret

    def _sympystr(self, printer):
        _ = printer
        return self.__str__()

    def _hashable_content(self):  # type: ignore
        return (self._metadata, self.name, self.subscripts)

    def _eval_subs(self, old, new) -> Basic | None:
        # taken from Symbol
        # TODO: what is the expected behaviour for the subsitution of
        # subscript indices? Also perform subsitution or treat
        # Index as Atom and only perform subsitution of the full index
        # including subscripts, i.e., (i_{j}).subs(j, k) will not work
        # but (i_{j}).subs(i_{j}, i_{k}) will work.
        # If we want this to work correctly, all arguments should go in
        # _args, so sympy can work with them.
        if old.is_Pow:
            return Pow(self, S.One, evaluate=False)._eval_subs(old, new)
        return None

    @property
    def free_symbols(self) -> set[Basic]:
        ret: set[Basic] = {self}
        for idx in self.subscripts:
            ret |= idx.free_symbols
        return ret
