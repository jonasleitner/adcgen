from collections.abc import Iterable
from functools import cached_property
from typing import Any, Sequence, TYPE_CHECKING
import itertools

from sympy.physics.secondquant import F, Fd, FermionicOperator, NO
from sympy import Add, Expr, Mul, Number, Pow, S, Symbol, latex, sympify

from ..indices import Index, _is_index_tuple
from ..logger import logger
from ..misc import cached_member
from ..sympy_objects import (
    Amplitude, AntiSymmetricTensor, KroneckerDelta, NonSymmetricTensor,
    SymbolicTensor, SymmetricTensor
)
from ..tensor_names import (
    is_adc_amplitude, is_t_amplitude, is_gs_density, split_gs_density_name,
    split_t_amplitude_name, tensor_names
)
from .container import Container
# imports only required for type checking (avoid circular imports)
if TYPE_CHECKING:
    from .expr_container import ExprContainer


class ObjectContainer(Container):
    """
    Wrapper for a single object, e.g., a tensor that is part of a term.

    Parameters
    ----------
    inner:
        The object to wrap, e.g., an AntiSymmetricTensor
    target_idx: Iterable[Index] | None, optional
        Target indices of the expression. By default the Einstein sum
        convention will be used to identify target and contracted indices,
        which is not always sufficient.
    """
    def __init__(self, inner: Expr | Container | Any,
                 target_idx: Iterable[Index] | None = None) -> None:
        super().__init__(inner=inner, target_idx=target_idx)
        # we can not wrap an Add object: should be wrapped by ExprContainer
        # we can not wrap an Mul object: should be wrapped by TermContainer
        # we can not wrap an NO object: should be wrapped by
        # NormalOrderedContainer
        # we can not wrap a polynom: should be wrapped by PolynomContainer
        # But everything else should be fine (single objects)
        assert not isinstance(self._inner, (Add, Mul, NO))
        if isinstance(self._inner, Pow):  # polynom
            assert not isinstance(self._inner.args[0], Add)

    ####################################
    # Some helpers for accessing inner #
    ####################################
    @property
    def base(self) -> Expr:
        """
        Returns the base of an :py:class:`Pow` object (base^exp) - if we have
        a Pow object. Otherwise the object itself is returned.
        """
        if isinstance(self.inner, Pow):
            return self.inner.args[0]
        else:
            return self.inner

    @property
    def exponent(self) -> Expr:
        """
        Returns the exponent of an :py:class:`Pow` object (base^exp).
        """
        if isinstance(self.inner, Pow):
            return self.inner.args[1]
        else:
            return sympify(1)

    @property
    def base_and_exponent(self) -> tuple[Expr, Expr]:
        """Return base and exponent of the object."""
        base = self.inner
        if isinstance(base, Pow):
            return base.args
        else:
            return base, sympify(1)

    @property
    def name(self) -> str | None:
        """Extract the name of tensor objects."""
        base = self.base
        if isinstance(base, SymbolicTensor):
            return base.name
        return None

    @property
    def is_t_amplitude(self) -> bool:
        """Whether the object is a ground state t-amplitude."""
        name = self.name
        return False if name is None else is_t_amplitude(name)

    @property
    def is_gs_density(self) -> bool:
        """Check whether the object is a ground state density tensor."""
        name = self.name
        return False if name is None else is_gs_density(name)

    @property
    def is_orbital_energy(self) -> bool:
        """Whether the object is a orbital energy tensor."""
        # all orb energies should be nonsym_tensors actually
        return self.name == tensor_names.orb_energy and len(self.idx) == 1

    @property
    def contains_only_orb_energies(self) -> bool:
        """Whether the object is a orbital energy tensor."""
        # To have a common interface with e.g. Polynoms
        return self.is_orbital_energy

    @cached_property
    def idx(self) -> tuple[Index, ...]:
        """Return the indices of the object."""
        if self.inner.is_number:  # prefactor
            return tuple()
        obj = self.base
        # Antisym-, Sym-, Nonsymtensor, Amplitude, Kroneckerdelta
        if isinstance(obj, (SymbolicTensor, KroneckerDelta)):
            return obj.idx
        elif isinstance(obj, FermionicOperator):  # F and Fd
            idx = obj.args
            assert _is_index_tuple(idx)
            return idx
        elif isinstance(obj, Symbol):  # a symbol without indices
            return tuple()
        else:
            raise TypeError("Can not determine the indices for an obj of type"
                            f"{type(obj)}: {self}.")

    @property
    def space(self) -> str:
        """Returns the index space (tensor block) of the object."""
        return "".join(s.space[0] for s in self.idx)

    @property
    def spin(self) -> str:
        """Returns the spin block of the current object."""
        return "".join(s.spin if s.spin else "n" for s in self.idx)

    ################################################
    # compute additional properties for the object #
    ################################################
    @property
    def type_as_str(self) -> str:
        """Returns a string that describes the type of the object."""
        if self.inner.is_number:
            return "prefactor"
        obj = self.base
        if isinstance(obj, Amplitude):
            return "amplitude"
        elif isinstance(obj, SymmetricTensor):
            return "symtensor"
        elif isinstance(obj, AntiSymmetricTensor):
            return "antisymtensor"
        elif isinstance(obj, NonSymmetricTensor):
            return "nonsymtensor"
        elif isinstance(obj, KroneckerDelta):
            return "delta"
        elif isinstance(obj, F):
            return "annihilate"
        elif isinstance(obj, Fd):
            return "create"
        elif isinstance(obj, Symbol):
            return "symbol"
        else:
            raise TypeError(f"Unknown object {self} of type {type(obj)}.")

    def longname(self, use_default_names: bool = False) -> str | None:
        """
        Returns a more exhaustive name of the object. Used for intermediates
        and transformation to code.

        Parameters
        ----------
        use_default_names: bool, optional
            If set, the default names are used to generate the longname.
            This is necessary to e.g., map a tensor name to an intermediate
            name, since they are defined using the default names.
            (default: False)
        """
        if any(s.spin for s in self.idx):
            logger.warning("Longname only covers the space of indices. The "
                           "spin is omitted.")
        name = None
        base = self.base
        if isinstance(base, SymbolicTensor):
            name = base.name
            # t-amplitudes
            if is_t_amplitude(name):
                assert isinstance(base, Amplitude)
                if len(base.upper) != len(base.lower):
                    raise RuntimeError("Number of upper and lower indices not "
                                       f"equal for t-amplitude {self}.")
                base_name, ext = split_t_amplitude_name(name)
                if use_default_names:
                    base_name = tensor_names.defaults().get("gs_amplitude")
                    assert base_name is not None
                if ext:
                    name = f"{base_name}{len(base.upper)}_{ext}"
                else:  # name for t-amplitudes without a order
                    name = f"{base_name}{len(base.upper)}"
            elif is_adc_amplitude(name):  # adc amplitudes
                assert isinstance(base, Amplitude)
                # need to determine the excitation space as int
                space = self.space
                assert all(sp in ["o", "v", "c"] for sp in space)
                n_o = space.count("o") + space.count("c")
                n_v = space.count("v")
                if n_o == n_v:  # pp-ADC
                    n = n_o  # p-h -> 1 // 2p-2h -> 2 etc.
                else:  # ip-/ea-/dip-/dea-ADC
                    n = min([n_o, n_v]) + 1  # h -> 1 / 2h -> 1 / p-2h -> 2...
                lr = "l" if name == tensor_names.left_adc_amplitude else 'r'
                name = f"u{lr}{n}"
            elif is_gs_density(name):  # mp densities
                assert isinstance(base, AntiSymmetricTensor)
                if len(base.upper) != len(base.lower):
                    raise RuntimeError("Number of upper and lower indices not "
                                       f"equal for mp density {self}.")
                base_name, ext = split_gs_density_name(name)
                if use_default_names:
                    base_name = tensor_names.defaults().get("gs_density")
                    assert base_name is not None
                if ext:
                    name = f"{base_name}0_{ext}_{self.space}"
                else:  # name for gs-dentity without a order
                    name = f"{base_name}0_{self.space}"
            elif name.startswith('t2eri'):  # t2eri
                name = f"t2eri_{name[5:]}"
            elif name == 't2sq':
                pass
            else:  # arbitrary other tensor
                name += f"_{self.space}"
        elif isinstance(base, KroneckerDelta):  # deltas -> d_oo / d_vv
            name = f"d_{self.space}"
        return name

    @cached_property
    def order(self) -> int:
        """
        Returns the perturbation theoretical order of the object (tensor).
        """
        from ..intermediates import Intermediates

        if isinstance(self.base, SymbolicTensor):
            name = self.name
            assert name is not None
            if name == tensor_names.eri:  # eri
                return 1
            elif is_t_amplitude(name):
                _, ext = split_t_amplitude_name(name)
                return int(ext.replace('c', ''))
            elif is_gs_density(name):
                # we might have p / p2 / p3 / ...
                _, ext = split_gs_density_name(name)
                if ext:
                    return int(ext)
            # all intermediates
            longname = self.longname(True)
            assert longname is not None
            itmd_cls = Intermediates().available.get(longname, None)
            if itmd_cls is not None:
                return itmd_cls.order
        return 0

    @cached_member
    def description(self, target_idx: Sequence[Index] | None = None,
                    include_exponent: bool = True) -> str:
        """
        Generates a string that describes the object.

        Parameters
        ----------
        target_idx: Sequence[Index] | None, optional
            The target indices of the term the object is a part of.
            If given, the explicit names of target indices will be
            included in the description.
        include_exponent: bool, optional
            If set the exponent of the object will be included in the
            description. (default: True)
        """

        descr = [self.type_as_str]
        if descr[0] in ["prefactor", "symbol"]:
            return descr[0]

        if descr[0] in ["antisymtensor", "amplitude", "symtensor"]:
            base, exponent = self.base_and_exponent
            assert isinstance(base, AntiSymmetricTensor)
            # - space separated in upper and lower part
            upper, lower = base.upper, base.lower
            assert _is_index_tuple(upper) and _is_index_tuple(lower)
            data_u = "".join(s.space[0] + s.spin for s in upper)
            data_l = "".join(s.space[0] + s.spin for s in lower)
            descr.append(f"{base.name}-{data_u}-{data_l}")
            # names of target indices, also separated in upper and lower part
            # indices in upper and lower have been sorted upon tensor creation!
            if target_idx is not None:
                target_u = "".join(s.name for s in upper if s in target_idx)
                target_l = "".join(s.name for s in lower if s in target_idx)
                if target_l or target_u:  # we have at least 1 target idx
                    if base.bra_ket_sym is S.Zero:  # no bra ket symmetry
                        if not target_u:
                            target_u = "none"
                        if not target_l:
                            target_l = "none"
                        descr.append(f"{target_u}-{target_l}")
                    else:  # bra ket sym or antisym
                        # target indices in both spaces
                        if target_u and target_l:
                            descr.extend(sorted([target_u, target_l]))
                        else:  # only in 1 space at least 1 target idx
                            descr.append(target_u + target_l)
            if include_exponent:  # add exponent to description
                descr.append(str(exponent))
        elif descr[0] == "nonsymtensor":
            data = "".join(s.space[0] + s.spin for s in self.idx)
            descr.append(f"{self.name}-{data}")
            if target_idx is not None:
                target_str = "".join(
                    s.name + str(i) for i, s in
                    enumerate(self.idx) if s in target_idx
                )
                if target_str:
                    descr.append(target_str)
            if include_exponent:
                descr.append(str(self.exponent))
        elif descr[0] in ["delta", "annihilate", "create"]:
            data = "".join(s.space[0] + s.spin for s in self.idx)
            descr.append(data)
            if target_idx is not None:
                target_str = "".join(
                    s.name for s in self.idx if s in target_idx
                )
                if target_str:
                    descr.append(target_str)
            if include_exponent:
                descr.append(str(self.exponent))
        else:
            raise ValueError(f"Unknown object {self} of type {descr[0]}")
        return "-".join(descr)

    @cached_member
    def crude_pos(self, target_idx: Sequence[Index] | None = None,
                  include_exponent: bool = True) -> dict[Index, list[str]]:
        """
        Returns the 'crude' position of the indices in the object.
        (e.g. only if they are located in bra/ket, not the exact position).

        Parameters
        ----------
        target_idx: Sequence[Index] | None, optional
            The target indices of the term the object is a part of.
            If given, the names of target indices will be included in
            the positions.
        include_exponent: bool, optional
            If set the exponent of the object will be considered in the
            positions. (default: True)
        """
        if not self.idx:  # just a number (prefactor or symbol)
            return {}

        ret = {}
        description = self.description(
            include_exponent=include_exponent, target_idx=target_idx
        )
        obj = self.base
        # antisym-, symtensor and amplitude
        if isinstance(obj, AntiSymmetricTensor):
            for uplo, idx_tpl in (("u", obj.upper), ("l", obj.lower)):
                assert _is_index_tuple(idx_tpl)
                for s in idx_tpl:
                    # space (upper/lower) in which the tensor occurs
                    pos = [description]
                    if obj.bra_ket_sym is S.Zero:
                        pos.append(uplo)
                    # space (occ/virt) of neighbour indices
                    neighbours = [i for i in idx_tpl if i is not s]
                    if neighbours:
                        neighbour_data = "".join(
                            i.space[0] + i.spin for i in neighbours
                        )
                        pos.append(neighbour_data)
                    # names of neighbour target indices
                    if target_idx is not None:
                        neighbour_target = [
                            i.name for i in neighbours if i in target_idx
                        ]
                        if neighbour_target:
                            pos.append("".join(neighbour_target))
                    if s not in ret:
                        ret[s] = []
                    ret[s].append("-".join(pos))
        elif isinstance(obj, NonSymmetricTensor):  # nonsymtensor
            # target idx position is already in the description
            idx = self.idx
            for i, s in enumerate(idx):
                if s not in ret:
                    ret[s] = []
                ret[s].append(f"{description}_{i}")
        # delta, create, annihilate
        elif isinstance(obj, (KroneckerDelta, F, Fd)):
            for s in self.idx:
                if s not in ret:
                    ret[s] = []
                ret[s].append(description)
        else:
            raise ValueError(f"Unknown object {self} of type {type(obj)}")
        return ret

    @property
    def allowed_spin_blocks(self) -> tuple[str, ...] | None:
        """
        Returns the valid spin blocks of the object.
        """
        from ..intermediates import Intermediates, RegisteredIntermediate

        # prefactor or symbol have no indices -> no allowed spin blocks
        if not self.idx:
            return None

        obj = self.base
        # antisym-, sym-, nonsymtensor and amplitude
        if isinstance(obj, SymbolicTensor):
            name = obj.name
            if name == tensor_names.eri:  # hardcode the ERI spin blocks
                return ("aaaa", "abab", "abba", "baab", "baba", "bbbb")
            # t-amplitudes: all spin conserving spin blocks are allowed, i.e.,
            # all blocks with the same amount of alpha and beta indices
            # in upper and lower
            elif is_t_amplitude(name):
                idx = obj.idx
                assert not len(idx) % 2
                n = len(idx)//2
                return tuple(sorted([
                    "".join(block) for block in
                    itertools.product("ab", repeat=len(idx))
                    if block[:n].count("a") == block[n:].count("a")
                ]))
            elif name == tensor_names.coulomb:  # ERI in chemist notation
                return ("aaaa", "aabb", "bbaa", "bbbb")
            elif name in (tensor_names.ri_sym, tensor_names.ri_asym_eri,
                          tensor_names.ri_asym_factor):
                return ("aaa", "abb")
            elif name == tensor_names.fock:
                return ("aa", "bb")
        elif isinstance(obj, KroneckerDelta):  # delta
            # spins have to be equal
            return ("aa", "bb")
        elif isinstance(obj, FermionicOperator):  # create / annihilate
            # both spins allowed!
            return ("a", "b")
        # the known allowed spin blocks of eri, t-amplitudes and deltas
        # may be used to generate the spin blocks of other intermediates
        longname = self.longname(True)
        assert longname is not None
        itmd = Intermediates().available.get(longname, None)
        if itmd is None:
            logger.warning(
                f"Could not determine valid spin blocks for {self}."
            )
            return None
        assert isinstance(itmd, RegisteredIntermediate)
        return itmd.allowed_spin_blocks

    def to_latex_str(self, only_pull_out_pref: bool = False,
                     spin_as_overbar: bool = False) -> str:
        """Returns a latex string for the object."""

        def format_indices(indices) -> str:
            if spin_as_overbar:
                spins = [s.spin for s in indices]
                if any(spins) and not all(spins):
                    raise ValueError("All indices have to have a spin "
                                     "assigned in order to differentiate "
                                     "indices without spin from indices with "
                                     f"alpha spin: {self}")
                return "".join(
                    f"\\overline{{{i.name}}}" if s == "b" else i.name
                    for i, s in zip(indices, spins)
                )
            else:
                return "".join(latex(i) for i in indices)

        if only_pull_out_pref:  # use sympy latex print
            return self.__str__()

        name = self.name
        obj, exp = self.base_and_exponent
        if isinstance(obj, SymbolicTensor):
            assert name is not None
            special_tensors = {
                tensor_names.eri: (  # antisym ERI physicist
                    lambda up, lo: f"\\langle {up}\\vert\\vert {lo}\\rangle"
                ),
                tensor_names.fock: (  # fock matrix
                    lambda up, lo: f"{tensor_names.fock}_{{{up}{lo}}}"
                ),
                # coulomb integral chemist notation
                tensor_names.coulomb: lambda up, lo: f"({up}\\vert {lo})",
                # 2e3c integral in asymmetric RI
                tensor_names.ri_asym_eri: lambda up, lo: f"({up}\\vert {lo})",
                # orbital energy
                tensor_names.orb_energy: lambda _, lo: f"\\varepsilon_{{{lo}}}"
            }
            # convert the indices to string
            if isinstance(obj, AntiSymmetricTensor):
                upper = format_indices(obj.upper)
                lower = format_indices(obj.lower)
            elif isinstance(obj, NonSymmetricTensor):
                upper, lower = None, format_indices(obj.indices)
            else:
                raise TypeError(f"Unknown tensor object {obj} of type "
                                f"{type(obj)}")

            if name in special_tensors:
                tex_str = special_tensors[name](upper, lower)
            else:
                order_str = None
                if is_t_amplitude(name):  # mp t-amplitudes
                    base_name, ext = split_t_amplitude_name(name)
                    if "c" in ext:
                        order_str = f"({ext.replace('c', '')})\\ast"
                    else:
                        order_str = f"({ext})"
                    order_str = f"}}^{{{order_str}}}"
                    name = f"{{{base_name}"
                elif is_gs_density(name):  # mp densities
                    _, ext = split_gs_density_name(name)
                    order_str = f"}}^{{({ext})}}"
                    name = "{\\rho"

                tex_str = name
                if upper is not None:
                    tex_str += f"^{{{upper}}}"
                tex_str += f"_{{{lower}}}"

                # append pt order for amplitude and mp densities
                if order_str is not None:
                    tex_str += order_str
        elif isinstance(obj, KroneckerDelta):
            tex_str = f"\\delta_{{{format_indices(obj.idx)}}}"
        elif isinstance(obj, F):  # annihilate
            tex_str = f"a_{{{format_indices(obj.args)}}}"
        elif isinstance(obj, Fd):  # create
            tex_str = f"a^\\dagger_{{{format_indices(obj.args)}}}"
        else:
            return self.__str__()

        if exp != 1:
            # special case for ERI and coulomb
            if name in [tensor_names.eri, tensor_names.coulomb]:
                tex_str += f"^{{{exp}}}"
            else:
                tex_str = f"\\bigl({tex_str}\\bigr)^{{{exp}}}"
        return tex_str

    ###################################
    # methods manipulating the object #
    ###################################
    def _apply_tensor_braket_sym(
            self, braket_sym_tensors: Sequence[str] = tuple(),
            braket_antisym_tensors: Sequence[str] = tuple(),
            wrap_result: bool = True) -> "ExprContainer | Expr":
        """
        Applies the bra-ket symmetry defined in braket_sym_tensors and
        braket_antisym_tensors to the current object.
        If wrap_result is set, the new object will be
        wrapped by :py:class:`ExprContainer`.
        """
        from .expr_container import ExprContainer

        obj = self.inner
        base, exponent = self.base_and_exponent
        if isinstance(base, AntiSymmetricTensor):
            name = base.name
            braketsym: None | Number = None
            if name in braket_sym_tensors and base.bra_ket_sym is not S.One:
                braketsym = S.One
            elif name in braket_antisym_tensors and \
                    base.bra_ket_sym is not S.NegativeOne:
                braketsym = S.NegativeOne
            if braketsym is not None:
                obj = Pow(
                    base.add_bra_ket_sym(braketsym),
                    exponent
                )
        if wrap_result:
            obj = ExprContainer(inner=obj, **self.assumptions)
        return obj

    def _rename_complex_tensors(self, wrap_result: bool = True
                                ) -> "ExprContainer | Expr":
        """
        Renames complex tensors to reflect that the expression is
        represented in a real orbital basis, e.g., complex t-amplitudes
        are renamed t1cc -> t1.

        Parameters
        ----------
        wrap_result: bool, optional
            If set the result will be wrapped with
            :py:class:`ExprContainer`. Otherwise the unwrapped
            object is returned. (default: True)
        """
        from .expr_container import ExprContainer

        real_obj = self.inner
        if self.is_t_amplitude:
            old = self.name
            assert old is not None
            base_name, ext = split_t_amplitude_name(old)
            new = f"{base_name}{ext.replace('c', '')}"
            if old != new:  # only rename when name changes
                base, exponent = self.base_and_exponent
                assert isinstance(base, Amplitude)
                real_obj = Pow(
                    Amplitude(new, base.upper, base.lower, base.bra_ket_sym),
                    exponent
                )
        if wrap_result:
            real_obj = ExprContainer(real_obj, **self.assumptions)
        return real_obj

    def block_diagonalize_fock(self, wrap_result: bool = True
                               ) -> "ExprContainer | Expr":
        """
        Block diagonalize the Fock matrix, i.e. if the object is part of an
        off-diagonal fock matrix block, it is set to 0.

        Parameters
        ----------
        wrap_result: bool, optional
            If this is set the result will be wrapped with an
            :py:class:`ExprContainer`. (default: True)
        """
        from .expr_container import ExprContainer

        bl_diag = self.inner
        if self.name == tensor_names.fock:
            sp1, sp2 = self.space
            if sp1 != sp2:
                bl_diag = S.Zero
        if wrap_result:
            bl_diag = ExprContainer(bl_diag, **self.assumptions)
        return bl_diag

    def diagonalize_fock(self, target: Sequence[Index],
                         wrap_result: bool = False
                         ) -> tuple["ExprContainer | Expr", dict[Index, Index]]:  # noqa E501
        """
        Diagonalize the fock matrix, i.e., if the object is a fock matrix
        element it is replaced by an orbital energy - but only if no
        information is lost.
        If the result is wrapped, the target indices will be set in the
        resulting expression, because it might not be possible to
        determine them according to the einstein sum convention
        (f_ij X_j -> e_i X_i).
        """
        from ..func import evaluate_deltas

        def pack_result(diag, sub, target):
            if wrap_result:
                assumptions = self.assumptions
                assumptions["target_idx"] = target
                diag = Expr(diag, **assumptions)
            return diag, sub

        if self.name != tensor_names.fock:  # no fock matrix
            return pack_result(self.inner, {}, target)
        # build a delta with the fock indices
        p, q = self.idx
        delta = KroneckerDelta(p, q)
        if delta is S.Zero:  # off diagonal block
            assert isinstance(delta, Number)
            return pack_result(delta, {}, target)
        elif delta is S.One:
            # diagonal fock element: if we evaluate it, we might loose a
            # contracted index.
            return pack_result(self.inner, {}, target)
        # try to evaluate the delta
        result = evaluate_deltas(Mul(self.inner, delta), target_idx=target)
        if isinstance(result, Mul):  # could not evaluate
            return pack_result(self.inner, {}, target)
        # check which of the indices survived
        remaining_idx = result.atoms(Index)
        assert len(remaining_idx) == 1  # only one of the indices can survive
        remaining_idx = remaining_idx.pop()
        # dict holding the necessary index substitution
        sub = {}
        if p is remaining_idx:  # p survived
            sub[q] = p
        else:  # q surived
            assert q is remaining_idx
            sub[p] = q
        diag = Pow(
            NonSymmetricTensor(tensor_names.orb_energy, (remaining_idx,)),
            self.exponent
        )
        return pack_result(diag, sub, target)

    def rename_tensor(self, current: str, new: str,
                      wrap_result: bool = True) -> "ExprContainer | Expr":
        """
        Renames a tensor object with name 'current' to 'new'. If wrap_result
        is set, the result will be wrapped with an :py:class:`ExprContainer`.
        """
        from .expr_container import ExprContainer

        obj = self.inner
        base, exponent = self.base_and_exponent
        if isinstance(base, SymbolicTensor) and base.name == current:
            if isinstance(base, AntiSymmetricTensor):
                # antisym, amplitude, symmetric
                base = base.__class__(
                    new, base.upper, base.lower, base.bra_ket_sym
                )
            elif isinstance(base, NonSymmetricTensor):
                # nonsymmetric
                base = base.__class__(new, base.indices)
            else:
                raise TypeError(f"Unknown tensor type {type(base)}.")
            obj = Pow(base, exponent)
        if wrap_result:
            obj = ExprContainer(obj, **self.assumptions)
        return obj

    def expand_coulomb_ri(self, factorisation: str = 'sym',
                          wrap_result: bool = True) -> "ExprContainer | Expr":
        """
        Expands the Coulomb operators (pq | rs) into RI format

        Parameters
        ----------
        factorisation : str, optional
            The type of factorisation ('sym' or 'asym'), by default 'sym'
        wrap_result : bool, optional
            Whether to wrap the result in an ExprContainer, by default True

        Returns
        -------
        ExprContainer | Expr
            The factorised expression.
        """
        from .expr_container import ExprContainer

        if factorisation not in ("sym", "asym"):
            raise NotImplementedError("Only symmetric ('sym') and asymmetric "
                                      "('asym') factorisation of the Coulomb "
                                      "integral is implemented")

        res = self.inner
        base, exponent = self.base_and_exponent
        if isinstance(base, SymmetricTensor) and \
                base.name == tensor_names.coulomb:
            if base.bra_ket_sym != 1:
                raise NotImplementedError("Can only apply RI approximation to "
                                          "coulomb integrals with "
                                          "bra-ket symmetry.")
            # we dont expand negative exponents, because the result
            # (ab)^-n will evaluate to a^-n b^-n, which is
            # only correct if the product ab has no contracted
            # indices
            if not exponent.is_Integer or exponent < S.Zero:
                raise NotImplementedError("Can only apply RI approximation to "
                                          "coulomb integrals "
                                          "with positive integer exponents. "
                                          f"{self} has an invalid exponent.")
            # setup the assumptions for the aux index:
            # assign alpha spin if represented in spatial orbitals
            idx = self.idx
            has_spin = bool(idx[0].spin)
            if any(bool(s.spin) != has_spin for s in idx):
                raise NotImplementedError(f"The coulomb integral {self} has "
                                          "to be represented either in spatial"
                                          " or spin orbitals. A mixture is not"
                                          " valid.")
            assumptions = {"aux": True}
            if has_spin:
                assumptions["alpha"] = True
            # actually do the expansion
            p, q, r, s = idx
            res = S.One
            for _ in range(int(exponent)):  # exponent has to be positive
                aux_idx = Index("P", **assumptions)
                if factorisation == "sym":
                    res *= SymmetricTensor(
                        tensor_names.ri_sym, (aux_idx,), (p, q), 0
                    )
                    res *= SymmetricTensor(
                        tensor_names.ri_sym, (aux_idx,), (r, s), 0
                    )
                else:
                    assert factorisation == "asym"
                    res *= SymmetricTensor(
                        tensor_names.ri_asym_factor, (aux_idx,), (p, q), 0
                    )
                    res *= SymmetricTensor(
                        tensor_names.ri_asym_eri, (aux_idx,), (r, s), 0
                    )
        if wrap_result:
            kwargs = self.assumptions
            res = ExprContainer(res, **kwargs)
        return res

    def expand_antisym_eri(self, wrap_result: bool = True
                           ) -> "ExprContainer | Expr":
        """
        Expands the antisymmetric ERI using chemists notation
        <pq||rs> = (pr|qs) - (ps|qr).
        ERI's in chemists notation are by default denoted as 'v'.
        Currently this only works for real orbitals, i.e.,
        for symmetric ERI's <pq||rs> = <rs||pq>.
        """
        from .expr_container import ExprContainer

        res = self.inner
        base, exponent = self.base_and_exponent
        if isinstance(base, AntiSymmetricTensor) and \
                base.name == tensor_names.eri:
            # ensure that the eri is Symmetric. Otherwise we would introduce
            # additional unwanted symmetry in the result
            if base.bra_ket_sym != 1:
                raise NotImplementedError("Can only expand antisymmetric ERI "
                                          "with bra-ket symmetry "
                                          "(real orbitals).")
            p, q, r, s = self.idx  # <pq||rs>
            res = S.Zero
            if p.spin == r.spin and q.spin == s.spin:
                res += SymmetricTensor(tensor_names.coulomb, (p, r), (q, s), 1)
            if p.spin == s.spin and q.spin == r.spin:
                res -= SymmetricTensor(tensor_names.coulomb, (p, s), (q, r), 1)
            res = Pow(res, exponent)

        if wrap_result:
            res = ExprContainer(res, **self.assumptions)
        return res

    def expand_intermediates(self, target: Sequence[Index],
                             wrap_result: bool = True,
                             fully_expand: bool = True,
                             braket_sym_tensors: Sequence[str] = tuple(),
                             braket_antisym_tensors: Sequence[str] = tuple()
                             ) -> "ExprContainer | Expr":
        """
        Expand the object if it is a known intermediate.

        Parameters
        ----------
        target: tuple[Index]
            The target indices of the term the object is a part of.
        wrap_result: bool, optional
            If set, the result will be wrapped with an
            :py:class:`ExprContainer`. Note that the target indices will
            be set in the resturned container, since the einstein
            sum convention is often not valid after intermediate
            expansion. (default: True)
        fully_expand: bool, optional
            True (default): The intermediate is recursively expanded
              into orbital energies and ERI (if possible)
            False: The intermediate is only expanded once, e.g., n'th
              order MP t-amplitudes are expressed by means of (n-1)'th order
              MP t-amplitudes and ERI.
        braket_sym_tensors: Sequence[str], optional
            Add bra-ket-symmetry to the given tensors of the expanded
            expression (after expansion of the intermediates).
        braket_antisym_tensors: Sequence[str], optional
            Add bra-ket-antisymmetry to the given tensors of the expanded
            expression (after expansion of the intermediates).
        """
        from ..intermediates import Intermediates
        from .expr_container import ExprContainer

        # intermediates only defined for tensors
        if not isinstance(self.base, SymbolicTensor):
            ret = self.inner
            if wrap_result:
                assumptions = self.assumptions
                assumptions["target_idx"] = target
                ret = ExprContainer(ret, **assumptions)
            return ret

        longname = self.longname(use_default_names=True)
        assert longname is not None
        itmd = Intermediates().available.get(longname, None)
        expanded = self.inner
        if itmd is not None:
            # for negative exponents we would have to ensure that the
            # intermediate is a "long" intermediate that consists of
            # multiple terms. Or if it consists of a single term
            # that it does not have any contracted indices
            # However, this can only be checked by calling ".expand()"
            # on the contributions in the for loop below, which seems bad.
            # A short intermediates will be expanded as
            # X^-2 = (ab * cd)^-1 -> a^-1 b^-1 c^-1 d^-1
            # where the last step is not correct if the intermediate
            # has contracted indices.
            exponent = self.exponent
            if not exponent.is_Integer or exponent < S.Zero:
                raise NotImplementedError(
                    "Can only expand intermediates with positive "
                    f"integer exponents. {self} has an invalid exponent."
                )
            # Use a for loop to obtain different contracted itmd indices
            # for each x in: x * x * ...
            expanded = S.One
            assert exponent.is_Integer
            for _ in range(abs(int(exponent))):
                expanded *= itmd.expand_itmd(
                    indices=self.idx, wrap_result=False,
                    fully_expand=fully_expand
                )
        # apply assumptions to the expanded object
        if braket_sym_tensors or braket_antisym_tensors:
            expanded = ExprContainer(expanded).add_bra_ket_sym(
                braket_sym_tensors=braket_sym_tensors,
                braket_antisym_tensors=braket_antisym_tensors
            ).inner

        if wrap_result:
            assumptions = self.assumptions
            assumptions["target_idx"] = target
            ret = ExprContainer(expanded, **assumptions)
        return expanded

    def use_explicit_denominators(self, wrap_result: bool = True
                                  ) -> "ExprContainer | Expr":
        """
        Switch to an explicit representation of orbital energy denominators by
        replacing all symbolic denominators by their explicit counter part,
        i.e., D^{ij}_{ab} -> (e_i + e_j - e_a - e_b)^{-1}.+
        """
        from .expr_container import ExprContainer

        explicit_denom = self.inner
        if self.name == tensor_names.sym_orb_denom:
            tensor, exponent = self.base_and_exponent
            assert isinstance(tensor, AntiSymmetricTensor)
            # upper indices are added, lower indices subtracted
            explicit_denom = S.Zero
            for s in tensor.upper:
                assert isinstance(s, Index)
                explicit_denom += NonSymmetricTensor(
                    tensor_names.orb_energy, (s,)
                )
            for s in tensor.lower:
                assert isinstance(s, Index)
                explicit_denom -= NonSymmetricTensor(
                    tensor_names.orb_energy, (s,)
                )
            explicit_denom = Pow(explicit_denom, -exponent)
        if wrap_result:
            explicit_denom = ExprContainer(explicit_denom, **self.assumptions)
        return explicit_denom
