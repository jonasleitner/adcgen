from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import Any
import itertools

from sympy.physics.secondquant import F, Fd, NO
from sympy import Expr, Mul

from ..indices import Index
from ..misc import cached_member
from .container import Container
from .object_container import ObjectContainer


class NormalOrderedContainer(ObjectContainer):
    """
    Wrapper for a normal ordered operator string.

    Parameters
    ----------
    inner:
        The NO object to wrap
    real : bool, optional
        Whether the expression is represented in a real orbital basis.
    sym_tensors: Iterable[str] | None, optional
        Names of tensors with bra-ket-symmetry, i.e.,
        d^{pq}_{rs} = d^{rs}_{pq}. Adjusts the corresponding tensors to
        correctly represent this additional symmetry if they are not aware
        of it yet.
    antisym_tensors: Iterable[str] | None, optional
        Names of tensors with bra-ket-antisymmetry, i.e.,
        d^{pq}_{rs} = - d^{rs}_{pq}. Adjusts the corresponding tensors to
        correctly represent this additional antisymmetry if they are not
        aware of it yet.
    target_idx: Iterable[Index] | None, optional
        Target indices of the expression. By default the Einstein sum
        convention will be used to identify target and contracted indices,
        which is not always sufficient.
    """
    def __init__(self, inner: Expr | Container | Any,
                 real: bool = False,
                 sym_tensors: Iterable[str] = tuple(),
                 antisym_tensors: Iterable[str] = tuple(),
                 target_idx: Iterable[Index] | None = None) -> None:
        # call init from ObjectContainers parent class
        super(ObjectContainer, self).__init__(
            inner=inner, real=real, sym_tensors=sym_tensors,
            antisym_tensors=antisym_tensors, target_idx=target_idx
        )
        assert isinstance(self._inner, NO)

    def __len__(self) -> int:
        return len(self._extract_operators.args)

    ####################################
    # Some helpers for accessing inner #
    ####################################
    @property
    def _extract_operators(self) -> Expr:
        operators = self._inner.args[0]
        assert isinstance(operators, Mul)
        return operators

    @cached_property
    def objects(self) -> tuple["ObjectContainer", ...]:
        return tuple(
            ObjectContainer(op, **self.assumptions)
            for op in self._extract_operators.args
        )

    @property
    def exponent(self) -> Expr:
        # actually sympy should throw an error if a NO object contains a Pow
        # obj or anything else than a*b*c
        exp = set(o.exponent for o in self.objects)
        if len(exp) == 1:
            return exp.pop()
        else:
            raise NotImplementedError(
                'Exponent only implemented for NO objects, where all '
                f'operators share the same exponent. {self}'
            )

    @cached_property
    def idx(self) -> tuple[Index, ...]:
        """
        Indices of the normal ordered operator string. Indices that appear
        multiple times will be listed multiple times.
        """
        objects = self.objects
        exp = self.exponent
        assert exp.is_Integer
        exp = int(exp)
        ret = tuple(s for o in objects for s in o.idx for _ in range(exp))
        if len(objects) != len(ret):
            raise NotImplementedError('Expected a NO object only to contain'
                                      "second quantized operators with an "
                                      f"exponent of 1. {self}")
        return ret

    ################################################
    # compute additional properties for the object #
    ################################################
    @property
    def type_as_str(self) -> str:
        return 'NormalOrdered'

    @cached_member
    def description(self, target_idx: Sequence[Index] | None = None,
                    include_exponent: bool = True) -> str:
        """
        Generates a string that describes the operators.

        Parameters
        ----------
        target_idx: Sequence[Index] | None, optional
            The target indices of the term the operators are a part of.
            If given, the explicit names of target indices will be
            included in the description.
        include_exponent: bool, optional
            If set the exponent of the object will be included in the
            description. (default: True)
        """
        # exponent has to be 1 for all contained operators
        assert self.exponent == 1
        _ = include_exponent

        obj_contribs = []
        for o in self.objects:
            # add either index space or target idx name
            idx = o.idx
            assert len(idx) == 1
            idx = idx[0]
            if target_idx is not None and idx in target_idx:
                op_str = f"{idx.name}_{idx.space[0]}{idx.spin}"
            else:
                op_str = idx.space[0] + idx.spin
            # add a plus for creation operators
            base = o.base
            if isinstance(base, Fd):
                op_str += '+'
            elif not isinstance(base, F):  # has to be annihilate here
                raise TypeError("Unexpected content for "
                                f"NormalOrderedContainer: {o}, {type(o)}.")
            obj_contribs.append(op_str)
        return f"{self.type_as_str}-{'-'.join(sorted(obj_contribs))}"

    @cached_member
    def crude_pos(self, target_idx: Sequence[Index] | None = None,
                  include_exponent: bool = True) -> dict[Index, list[str]]:
        """
        Returns the 'crude' position of the indices in the operator string.

        Parameters
        ----------
        target_idx: Sequence[Index] | None, optional
            The target indices of the term the operators are a part of.
            If given, the names of target indices will be included in
            the positions.
        include_exponent: bool, optional
            If set the exponent of the object will be considered in the
            positions. (default: True)
        """

        descr = self.description(
            target_idx=target_idx, include_exponent=include_exponent
        )
        ret = {}
        for o in self.objects:
            o_descr = o.description(
                target_idx=target_idx, include_exponent=include_exponent
            )
            idx = o.idx
            assert len(idx) == 1
            idx = idx[0]
            if idx not in ret:
                ret[idx] = []
            ret[idx].append(f"{descr}_{o_descr}")
        return ret

    @property
    def allowed_spin_blocks(self) -> tuple[str, ...]:
        """
        Returns the valid spin blocks of the operator string.
        """
        allowed_blocks = []
        for obj in self.objects:
            blocks = obj.allowed_spin_blocks
            if blocks is not None:
                allowed_blocks.append(blocks)
        return tuple("".join(b) for b in itertools.product(*allowed_blocks))

    def to_latex_str(self, only_pull_out_pref: bool = False,
                     spin_as_overbar: bool = False) -> str:
        """Returns a latex string for the object."""
        return " ".join(
            o.to_latex_str(only_pull_out_pref, spin_as_overbar)
            for o in self.objects
        )
