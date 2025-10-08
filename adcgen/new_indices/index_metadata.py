from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .index_space import IndexSpace


@dataclass(slots=True, frozen=True)
class IndexName:
    """
    The name of an index split in base (single non-digit character) and
    a optional suffix (an integer).
    """
    base: str
    suffix: int

    @classmethod
    def from_str(cls, name: str) -> "IndexName":
        """Constructs an IndexName from a string representing the name."""
        base, suffix = name[0], name[1:]
        if base.isdigit():
            raise ValueError(f"Invalid base {base} for index name. Digits are "
                             f"not allowed. Base extracted from {name}")
        if not all(c.isdigit() for c in suffix):
            raise ValueError(f"Invalid index name suffix: {suffix}. Non-digit "
                             f"character found. Extracted from name {name}.")
        return cls(base=base, suffix=int(suffix) if suffix else 0)

    def sort_key(self) -> tuple[int, str]:
        return self.suffix, self.base

    def __str__(self) -> str:
        return f"{self.base}{self.suffix}" if self.suffix else self.base


class Spin(Enum):
    # the values define the ordering of the different enum variants
    NONE = 0
    ALPHA = 1
    BETA = 2

    @classmethod
    def from_str(cls, spin: str) -> "Spin":
        """
        Constructs a Spin from string. Valid strings are
        'a' or 'alpha' -> Spin.ALPHA
        'b' or 'beta'  -> Spin.BETA
        'n' or 'none'  -> Spin.NONE
        """
        spin = spin.lower()
        if spin in ("n", "none"):
            return cls.NONE
        elif spin in ("a", "alpha"):
            return cls.ALPHA
        elif spin in ("b", "beta"):
            return cls.BETA
        raise ValueError("Invalid spin string. Failed to "
                         f"construct Spin from string {spin}.")

    def sort_key(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return self is not Spin.NONE


@dataclass(slots=True, frozen=True)
class IndexMetaData:
    """
    The metadata of an Index, e.g., space and spin.
    """
    space: "IndexSpace"
    spin: Spin

    def sort_key(self) -> tuple[int, int]:
        return self.space.sort_key(), self.spin.sort_key()
