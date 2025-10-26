from enum import Enum
from typing import Any
import dataclasses

from sympy import S, Symbol
from sympy.core.cache import cacheit


class Spin(Enum):
    ALPHA = "alpha"
    BETA = "beta"
    NONE = "none"

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return self.name != "NONE"


@dataclasses.dataclass(slots=True, frozen=True)
class IndexMetaData:
    space: str
    spin: Spin = Spin.NONE


# Inherit from symbol or from Dummy?
# aka should we stay with the 'is' comparison?
#  Pro: easily create Dummy indices at places like wicks, RI and derivative
#  Con: equations might be hard to read and we still need to cache all indices
# -> 'is' comparison might be easier for consistency
# Also we can rewrite the adc factories to use substitute_contracted
# at the end of a function/method
# -> fewer calls to get_generic_indices

# Take the IndexSpace or only it's name upon creation?
# I guess the name, since IndexSpace should not directly be coupled to Index
# but instead only indirectly through the Indices class.

# how to handle the optional sub indices?
# If I cache them on indices but allow to modify the subscript after index
# creation, this will corrupt the cache...
# -> Indices have to know about their subindices when asking the Indices class
#    for indices
#
# There are 2 options regarding teh indices
# option1: cache the instances on the Index class
# -> Index("i") == Index("i")
# -> need to use another class of Indices for wicks, RI, ...
# -> During building of the expression we need to use get_generic_indices
#    or define other logic to identify available indices
# option2: cache the instances on the Indices class
# -> upon creation (and caching) of an Index the subscript must be
#    defined, which requires a change in the Indices API
#    ... not clear how the API should look like in this case
# In both cases an Index has to be immutable!
# -> overwrite setattr does not work since we inherit from symbol
# Since Symbol is a very well known sympy quantity it might be very good
# to inherit from it due to .is_Symbol checks throughout the code

class Index(Symbol):
    """
    Represents an Index.
    Comparison of indices is handled via the 'is' operator, i.e.,
    Index("x", "space") != Index("x", "space").

    Parameters
    ----------
    name: str
        The name of the index.
    space: str
        The name of the space the index belongs to.
    spin: Spin, optional
        The spin of the index. By default the index has no spin.
    subscript: Index, optional
        Another index that is used as subscript for the current index.
        Can be used to connect two indices and allows for an recursive
        definition.
    """
    __slots__ = ("_metadata", "_subscript")
    _metadata: IndexMetaData
    _subscript: "Index | None"

    def __new__(cls, name: str, space: str, spin: Spin = Spin.NONE,
                subscript: "Index | None" = None) -> "Index":
        # don't call __new__ but __xnew__ instead to avoid caching
        # the instances
        obj = super().__xnew__(cls, name=name)
        obj._metadata = IndexMetaData(space=space, spin=spin)
        # subscript has to be a different object since we just created
        # the current index instance
        obj._subscript = subscript
        return obj

    @property
    def space(self) -> str:
        return self._metadata.space

    @property
    def spin(self) -> Spin:
        return self._metadata.spin

    @property
    def subscript(self) -> "Index | None":
        return self._subscript

    def metadata(self, recurse: bool = True) -> tuple[IndexMetaData, ...]:
        """
        Metadata like space and spin of the index. The name is excplicitly not
        included!
        If recurse is set, the metadata of subscripts is also collected.
        """
        if not recurse or self.subscript is None:
            return (self._metadata,)

        ret = [self._metadata]
        subscript = self.subscript
        while subscript is not None:
            ret.append(subscript._metadata)
            subscript = subscript.subscript
        return tuple(ret)

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    #################
    # print methods #
    #################
    def __str__(self) -> str:
        ret = self.name
        spin = self.spin
        subscript = self.subscript
        if spin:
            ret += f"^{{{spin.value}}}"
        if subscript is not None:
            ret += f"_{{{str(subscript)}}}"
        return ret

    def _sympystr(self, printer) -> str:
        _ = printer
        return self.__str__()

    def _latex(self, printer) -> str:
        _ = printer
        ret = self.name
        spin = self.spin
        subscript = self.subscript
        if spin:
            ret += f"^{{\\{spin.value}}}"
        if subscript is not None:
            ret += f"_{{{subscript._latex(printer)}}}"
        return ret

    #################################
    # Overwrite methods from Symbol #
    #################################
    def __getnewargs_ex__(self):  # type: ignore
        # in order to pickle an Index
        return ((self.name, self.space, self.spin, self.subscript), dict())

    def _hashable_content(self):
        # Since we modified __eq__ to compare the memory address
        # we should also modify the hashable_content to also depend on the
        # memory address since it might be used for more internal comparisons.
        return super()._hashable_content() + (str(id(self)),)

    @cacheit
    def sort_key(self, order: Any = None):  # type: ignore
        _ = order
        # I guess this is for internal sorting of expressions in sympy
        if self.subscript is None:
            subscript_key = None
        else:
            subscript_key = self.subscript.sort_key()
        # adapted from the Symbol class...
        # not rly sure why the S.Ones are there.
        return (
            self.class_key(),
            (2, (self.space, self.spin, self.name, subscript_key, id(self))),
            S.One.sort_key(), S.One
        )
