from collections.abc import Sequence, Callable
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
import itertools

from sympy import Add, Expr, Min, Mul, Pow, Rational, S

from .expression import ExprContainer, ObjectContainer
from .core_valence_separation import allowed_cvs_blocks
from .indices import (
    Indices, Index,
    get_symbols, order_substitutions, sort_idx_canonical,
)
from .logger import logger
from .misc import Inputerror, Singleton, cached_member
from .eri_orbenergy import EriOrbenergy
from .sympy_objects import NonSymmetricTensor, AntiSymmetricTensor, Amplitude
from .symmetry import LazyTermMap, Permutation
from .spatial_orbitals import allowed_spin_blocks
from .tensor_names import tensor_names


@dataclass
class ItmdExpr:
    expr: Expr
    target: tuple[Index, ...]
    contracted: tuple[Index, ...] | None


class Intermediates(metaclass=Singleton):
    """
    Manages all defined intermediates.
    New intermediates can be defined by inheriting from
    'RegisteredIntermediate'.
    """

    def __init__(self):
        self._registered: dict[str, dict[str, RegisteredIntermediate]] = (
            RegisteredIntermediate()._registry
        )
        self._available: dict[str, RegisteredIntermediate] = {
            name: obj for objects in self._registered.values()
            for name, obj in objects.items()
        }

    @property
    def available(self) -> dict[str, "RegisteredIntermediate"]:
        """
        Returns all available intermediates using their name as dict key.
        """
        return self._available

    @property
    def types(self) -> list[str]:
        """Returns all available types of intermediates."""
        return list(self._registered.keys())

    def __getattr__(self, attr: str) -> dict[str, "RegisteredIntermediate"]:
        if attr in self._registered:  # is the attr an intermediate type?
            return self._registered[attr]
        elif attr in self._available:  # is the attr an intermediate name?
            return {attr: self._available[attr]}
        else:
            raise AttributeError(f"{self} has no attribute {attr}. "
                                 f"The intermediate types: {self.types} "
                                 "and the intermediate names: "
                                 f"{list(self.available.keys())} are "
                                 "available.")


class RegisteredIntermediate:
    """
    Base class for defined intermediates.
    New intermediates can be added by inheriting from this class and require:
    - an itmd type '_itmd_type'
    - an perturbation theoretical order '_order'
    - names of default indices '_default_idx'
    - a method that fully expands the itmd into orbital energies and ERI
      '_build_expanded_itmd'
    - a method that returns the itmd tensor '_build_tensor'
    """
    _registry: dict[str, dict[str, "RegisteredIntermediate"]] = {}
    _itmd_type: str | None = None
    _order: int | None = None
    _default_idx: tuple[str, ...] | None = None

    def __init_subclass__(cls) -> None:
        itmd_type = cls._itmd_type
        assert itmd_type is not None
        if itmd_type not in cls._registry:
            cls._registry[itmd_type] = {}
        if (name := cls.__name__) not in cls._registry[itmd_type]:
            cls._registry[itmd_type][name] = cls()

    @property
    def name(self) -> str:
        """Name of the intermediate (the class name)."""
        return type(self).__name__

    @property
    def order(self) -> int:
        """Perturbation theoretical order of the intermediate."""
        if not hasattr(self, "_order") or self._order is None:
            raise AttributeError(f"No order defined for {self.name}.")
        return self._order

    @property
    def default_idx(self) -> tuple[str, ...]:
        """Names of the default indices of the intermediate."""
        if not hasattr(self, "_default_idx") or self._default_idx is None:
            raise AttributeError(f"No default indices defined for {self.name}")
        return self._default_idx

    @property
    def itmd_type(self) -> str:
        """The type of the intermediate."""
        if not hasattr(self, "_itmd_type") or self._itmd_type is None:
            raise AttributeError(f"No itmd_type defined for {self.name}.")
        return self._itmd_type

    def validate_indices(self,
                         indices: Sequence[str] | Sequence[Index] | None = None
                         ) -> list[Index]:
        """
        Ensures that the indices are valid for the intermediate and
        transforms them to 'Index' instances.
        """
        if indices is None:  # no need to validate the default indices
            return get_symbols(self.default_idx)

        indices = get_symbols(indices)
        default = get_symbols(self.default_idx)
        if len(indices) != len(default):
            raise Inputerror("Wrong number of indices for the itmd "
                             f"{self.name}.")
        elif any(s.space != d.space for s, d in zip(indices, default)):
            raise Inputerror(f"The indices {indices} are not valid for the "
                             f"itmd {self.name}")
        return indices

    def expand_itmd(self,
                    indices: Sequence[str] | Sequence[Index] | None = None,
                    wrap_result: bool = True, fully_expand: bool = True
                    ) -> Expr | ExprContainer:
        """
        Expands the intermediate into orbital energies and ERI.

        Parameters
        ----------
        indices : Sequence[str] | Sequence[Index], optional
            The names of the indices of the intermediate. By default the
            default indices (defined on the itmd class) will be used.
        wrap_result : bool, optional
            Whether to wrap the result in an
            :py:class:ExprContainer. (default: True)
        fully_expand : bool, optional
            True (default): The returned intermediate is recursively fully
              expanded into orbital energies and ERI (if possible).
            False: Returns a more readable version which is not recusively
              expanded, e.g., n't-order MP t-amplitudes are expressed by
              means of (n-1)'th-order MP t-amplitudes.
        """
        # check that the provided indices are fine for the itmd
        indices = self.validate_indices(indices)
        # currently all intermediates are only implemented for spin orbitals,
        # because the intermediate definition depends on the spin, i.e.,
        # we would need either multiple definitions per intermediate or
        # incorporate the spin in the intermediate names.
        if any(idx.spin for idx in indices):
            raise NotImplementedError(
                    "Intermediates not implemented for indices with spin "
                    "(spatial orbitals)."
            )

        # build a cached base version of the intermediate where we can just
        # substitute indices in
        expanded_itmd = self._build_expanded_itmd(fully_expand)

        # build the substitution dict
        subs: dict[Index, Index] = {}
        # map target indices onto each other
        if (base_target := expanded_itmd.target) is not None:
            subs.update({o: n for o, n in zip(base_target, indices)})
        # map contracted indices onto each other (replace them by generic idx)
        if (base_contracted := expanded_itmd.contracted) is not None:
            spaces = [s.space_and_spin for s in base_contracted]
            kwargs = Counter(
                f"{sp}_{spin}" if spin else sp for sp, spin in spaces
            )
            contracted = Indices().get_generic_indices(**kwargs)
            for new in contracted.values():
                new.reverse()
            for old, sp in zip(base_contracted, spaces):
                subs[old] = contracted[sp].pop()
            if any(li for li in contracted.values()):
                raise RuntimeError("Generated more contracted indices than "
                                   f"necessary. {contracted} are left.")

        # do some extra work with the substitutions to avoid using the
        # simultantous=True option for subs (very slow)
        itmd = expanded_itmd.expr.subs(order_substitutions(subs))
        assert isinstance(itmd, Expr)

        if itmd is S.Zero and expanded_itmd.expr is not S.Zero:
            raise ValueError(f"The substitutions {subs} are not valid for "
                             f"{expanded_itmd.expr}.")

        if wrap_result:
            itmd = ExprContainer(itmd, target_idx=indices)
        return itmd

    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        """
        Expand the intermediate using the default indices.
        """
        _ = fully_expand
        raise NotImplementedError("Build expanded intermediate not implemented"
                                  f" on {self.name}")

    def tensor(self, indices: Sequence[str] | Sequence[Index] | None = None,
               wrap_result: bool = True):
        """
        Returns the itmd tensor.

        Parameters
        ----------
        indices : str, optional
            The names of the indices of the intermediate. By default the
            default indices (defined on the itmd class) will be used.
        wrap_result : bool, optional
            Whether to wrap the result in an
            :py:class:ExprContainer. (default: True)
        """
        # check that the provided indices are sufficient for the itmd
        indices = self.validate_indices(indices)

        # build the tensor object
        tensor = self._build_tensor(indices=indices)
        if wrap_result:
            kwargs = {}
            if isinstance(tensor, AntiSymmetricTensor):
                if tensor.bra_ket_sym is S.One:  # bra ket symmetry
                    kwargs["sym_tensors"] = (tensor.name,)
                elif tensor.bra_ket_sym is S.NegativeOne:  # bra ket anisym
                    kwargs["antisym_tensors"] = (tensor.name,)
            return ExprContainer(tensor, **kwargs)
        else:
            return tensor

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        """
        Build the tensor representing the intermediate using the given indices.
        """
        _ = indices
        raise NotImplementedError("Build tensor not implemented on "
                                  f"{self.name}")

    @cached_property
    def tensor_symmetry(self) -> dict[tuple[Permutation, ...], int]:
        """
        Determines the symmetry of the itmd tensor object using the
        default indices, e.g., ijk/abc triples symmetry for t3_2.
        """
        tensor = self.tensor(wrap_result=True)
        assert isinstance(tensor, ExprContainer) and len(tensor) == 1
        return tensor.terms[0].symmetry()

    @cached_property
    def allowed_spin_blocks(self) -> tuple[str, ...]:
        """Determines all non-zero spin block of the intermediate."""

        target_idx = self.default_idx
        itmd = self.expand_itmd(
            indices=target_idx, wrap_result=True, fully_expand=False
        )
        assert isinstance(itmd, ExprContainer)
        return allowed_spin_blocks(itmd.expand(), target_idx)

    @cached_member
    def allowed_cvs_blocks(
            self,
            cvs_approximation: Callable[[ObjectContainer, str], bool] | None = None  # noqa E501
            ) -> tuple[str, ...]:
        """
        Splits the occupied orbitals in core and valence orbitals and
        determines the valid blocks if the CVS approximation is applied.

        Parameters
        ----------
        cvs_approximation : callable, optional
            Callable that takes an expr_container.Obj instance and a space
            string (e.g. 'covv'). It returns a bool indicating whether the
            block of the object described by the space string is valid within
            the CVS approximation, i.e., whether the block is neglected or not.
            By default, the "is_allowed_cvs_block" function is used,
            which applies the CVS approximation as described in
            10.1063/1.453424 and as implemented in adcman/adcc.
        """
        target_idx = self.default_idx
        itmd = self.expand_itmd(
            indices=target_idx, wrap_result=True, fully_expand=False
        )
        assert isinstance(itmd, ExprContainer)
        return allowed_cvs_blocks(
            itmd.expand(), target_idx, cvs_approximation=cvs_approximation
        )

    @cached_member
    def itmd_term_map(self, factored_itmds: Sequence[str] = tuple()
                      ) -> LazyTermMap:
        """
        Returns a map that lazily determines permutations of target indices
        that map terms in the intermediate definition onto each other.

        Parameters
        ----------
        factored_itmds : tuple[str], optional
            Names of other intermediates to factor in the fully expanded
            definition of the current intermediate which (if factorization is
            successful) changes the form of the intermediate.
            By default the fully expanded version will be used.
        """
        # - load the appropriate version of the intermediate
        itmd = self._prepare_itmd(factored_itmds)
        return LazyTermMap(itmd)

    @cached_member
    def _prepare_itmd(self, factored_itmds: Sequence[str] = tuple()
                      ) -> ExprContainer:
        """"
        Generates a variant of the intermediate with default indices and
        simplifies it as much as possible.

        Parameters
        ----------
        factored_itmds : tuple[str], optional
            Names of other intermediates to factor in the fully expanded
            definition of the current intermediate. By default the fully
            expanded version will be used.
        """
        from .reduce_expr import factor_eri_parts, factor_denom

        # In a usual run we only need 1 variant of an intermediate:
        #   a  b  c  d  e
        #      a  b  c  d
        #         a  b  c
        #            a  b
        #               a
        # For example, always the version of b where a is factorized
        # -> for b this function will always be called with a as factored_itmds
        # -> caching decorator is sufficient... no need to additionally
        #    cache the simplified base version

        # build the base version of the itmd and simplify it
        # - factor eri and denominator
        itmd = self.expand_itmd(wrap_result=True, fully_expand=True)
        assert isinstance(itmd, ExprContainer)
        itmd.expand().make_real()
        reduced = itertools.chain.from_iterable(
            factor_denom(sub_expr) for sub_expr in factor_eri_parts(itmd)
        )
        itmd = ExprContainer(0, **itmd.assumptions)
        for term in reduced:
            itmd += term.factor()

        logger.info("".join([
            "\n", "-"*80, "\n",
            f"Preparing Intermediate: Factoring {factored_itmds}"
        ]))

        if factored_itmds:
            available = Intermediates().available
            # iterate through factored_itmds and factor them one after another
            # in the simplified base itmd
            for i, it in enumerate(factored_itmds):
                logger.info("\n".join([
                    "-"*80, f"Factoring {it} in {self.name}:"
                ]))
                itmd = available[it].factor_itmd(
                    itmd, factored_itmds=factored_itmds[:i],
                    max_order=self.order
                )
        logger.info("".join([
            "\n", "-"*80, "\n",
            f"Done with factoring {factored_itmds} in {self.name}", "\n",
            "-"*80
        ]))
        return itmd

    def factor_itmd(self, expr: ExprContainer,
                    factored_itmds: Sequence[str] = tuple(),
                    max_order: int | None = None,
                    allow_repeated_itmd_indices: bool = False
                    ) -> ExprContainer:
        """
        Factors the intermediate in an expression assuming a real orbital
        basis.

        Parameters
        ----------
        expr : Expr
            Expression in which to factor intermediates.
        factored_itmds : Sequence[str], optional
            Names of other intermediates that have already been factored in
            the expression. It is necessary to factor those intermediates in
            the current intermediate definition as well, because the
            definition might change. By default the fully expanded version
            of the intermediate will be used.
        max_order : int, optional
            The maximum perturbation theoretical order of intermediates
            to consider.
        allow_repeated_itmd_indices: bool, optional
            If set, the factorization of intermediates of the form I_iij are
            allowed, i.e., indices on the intermediate may appear more than
            once. This corresponds to either a partial trace or a diagonal
            element of the intermediate. Note that this does not consistently
            work for "long" intermediates (at least 2 terms), because the
            number of terms might be reduced which is not correctly handled
            currently.
        """

        from .factor_intermediates import (
            _factor_long_intermediate, _factor_short_intermediate,
            FactorizationTermData
        )

        assert isinstance(expr, ExprContainer)
        if not expr.real:
            raise NotImplementedError("Intermediates only implemented for "
                                      "a real orbital basis.")

        # ensure that the previously factored intermediates
        # are provided as tuple -> can use them as dict key
        if not isinstance(factored_itmds, tuple):
            factored_itmds = tuple(factored_itmds)

        # can not factor if the expr is just a number or the intermediate
        # has already been factored or the order of the pt order of the
        # intermediate is to high.
        # also it does not make sense to factor t4_2 again, because of the
        # used factorized form.
        if expr.inner.is_number or self.name in factored_itmds or \
                self.name == 't4_2' or \
                (max_order is not None and max_order < self.order):
            return expr

        expr = expr.expand()
        terms = expr.terms

        # if want to factor a t_amplitude
        # -> terms to consider need to have a denominator
        # Also the pt order of the term needs to be high enough for the
        # current intermediate
        if self.itmd_type == 't_amplitude' and self.name != 't4_2':
            term_is_relevant = [
                term.order >= self.order and
                any(o.exponent < S.Zero and o.contains_only_orb_energies
                    for o in term.objects)
                for term in terms
            ]
        else:
            term_is_relevant = [term.order >= self.order for term in terms]
        # no term has a denominator or a sufficient pt order
        # -> can't factor the itmd
        if not any(term_is_relevant):
            return expr

        # determine the maximum pt order present in the expr (order is cached)
        max_order = max(term.order for term in terms)

        # build a new expr that only contains the relevant terms
        remainder = S.Zero
        to_factor = ExprContainer(0, **expr.assumptions)
        for term, is_relevant in zip(terms, term_is_relevant):
            if is_relevant:
                to_factor += term
            else:
                remainder += term.inner

        # - prepare the itmd for factorization and extract data to speed
        #   up the later comparison
        itmd_expr = self._prepare_itmd(factored_itmds=factored_itmds)
        itmd: tuple[EriOrbenergy, ...] = tuple(
            EriOrbenergy(term).canonicalize_sign() for term in itmd_expr.terms
        )
        itmd_data: tuple[FactorizationTermData, ...] = tuple(
            FactorizationTermData(term) for term in itmd
        )

        # factor the intermediate in the expr
        if len(itmd) == 1:  # short intermediate that consists of a single term
            factored = _factor_short_intermediate(
                to_factor, itmd[0], itmd_data[0], self,
                allow_repeated_itmd_indices=allow_repeated_itmd_indices
            )
            factored += remainder
        else:  # long intermediate that consists of multiple terms
            itmd_term_map = self.itmd_term_map(factored_itmds)
            for _ in range(max_order // self.order):
                to_factor = _factor_long_intermediate(
                    to_factor, itmd, itmd_data, itmd_term_map, self,
                    allow_repeated_itmd_indices=allow_repeated_itmd_indices
                )
            factored = to_factor + remainder
        return factored


# -----------------------------------------------------------------------------
# INTERMEDIATE DEFINITIONS:


class t2_1(RegisteredIntermediate):
    """First order MP doubles amplitude."""
    _itmd_type = 't_amplitude'  # type has to be a class variable
    _order = 1
    _default_idx = ("i", "j", "a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        _ = fully_expand
        # build a basic version of the intermediate using minimal indices
        # 'like on paper'
        i, j, a, b = get_symbols(self.default_idx)
        denom = Add(
            orb_energy(a), orb_energy(b), -orb_energy(i), -orb_energy(j)
        )
        ampl = eri((a, b, i, j)) * S.One / denom
        assert isinstance(ampl, Expr)
        return ItmdExpr(ampl, (i, j, a, b), None)

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        # guess its not worth caching here. Maybe if used a lot.
        # build the tensor
        return Amplitude(
            f"{tensor_names.gs_amplitude}1", indices[2:], indices[:2]
        )

    def factor_itmd(self, expr: ExprContainer,
                    factored_itmds: Sequence[str] | None = None,
                    max_order: int | None = None,
                    allow_repeated_itmd_indices: bool = False
                    ) -> ExprContainer:
        """
        Factors the t2_1 intermediate in an expression assuming a real
        orbital basis.
        """
        _ = allow_repeated_itmd_indices
        assert isinstance(expr, ExprContainer)
        if not expr.real:
            raise NotImplementedError("Intermediates only implemented for a "
                                      "real orbital basis.")
        # do we have something to factor? did we already factor the itmd?
        if expr.inner.is_number or \
                (factored_itmds and self.name in factored_itmds):
            return expr

        # no need to determine max order for a first order intermediate
        if max_order is not None and max_order < self.order:
            return expr

        # prepare the itmd and extract information
        t2 = self.expand_itmd(wrap_result=True, fully_expand=True)
        assert isinstance(t2, ExprContainer)
        t2.make_real()
        t2 = EriOrbenergy(t2).canonicalize_sign()
        t2_eri: ObjectContainer = t2.eri.objects[0]
        t2_eri_descr: str = t2_eri.description(include_exponent=False,
                                               target_idx=None)
        t2_denom = t2.denom.inner
        t2_eri_idx: tuple[Index, ...] = t2_eri.idx

        expr = expr.expand()

        factored = ExprContainer(0, **expr.assumptions)
        for term in expr.terms:
            term = EriOrbenergy(term)  # split the term

            if term.denom.inner.is_number:  # term needs to have a denominator
                factored += term.expr
                continue
            term = term.canonicalize_sign()  # fix the sign of the denominator

            brackets = term.denom_brackets
            removed_brackets: set[int] = set()
            factored_term = ExprContainer(1, **expr.assumptions)
            eri_obj_to_remove: list[int] = []
            denom_brackets_to_remove: list[int] = []
            for eri_idx, eri in enumerate(term.eri.objects):
                # - compare the eri objects (check if we have a oovv eri)
                #   coupling is not relevant for t2_1 (only a single object)
                descr: str = eri.description(
                    include_exponent=False, target_idx=None
                )
                if descr != t2_eri_descr:
                    continue
                # repeated indices on t2_1 make no sense since
                # t^aa_ij = <ij||aa> / (2a - i - j) = 0
                # due to the permutational antisymmetry of V
                assert all(c == 1 for c in Counter(eri.idx).values())
                # - have a correct eri -> zip indices together and substitute
                #   the itmd denominator
                sub = order_substitutions(dict(zip(t2_eri_idx, eri.idx)))
                sub_t2_denom = t2_denom.subs(sub)
                # consider the exponent!
                # <oo||vv>^2 may be factored twice
                eri_exp = eri.exponent
                # - check if we find a matching denominator
                for bk_idx, bk in enumerate(brackets):
                    # was the braket already removed?
                    if bk_idx in removed_brackets:
                        continue
                    if isinstance(bk, ExprContainer):
                        bk_exponent = S.One
                        bk = bk.inner
                    else:
                        bk, bk_exponent = bk.base_and_exponent
                    # found matching bracket in denominator
                    if bk == sub_t2_denom:
                        # can possibly factor multiple times, depending
                        # on the exponent of the eri and the denominator
                        min_exp = Min(eri_exp, bk_exponent)
                        # are we removing the bracket completely?
                        if min_exp == bk_exponent:
                            removed_brackets.add(bk_idx)
                        # found matching eri and denominator
                        # replace eri and bracket by a t2_1 tensor
                        assert min_exp.is_Integer
                        denom_brackets_to_remove.extend(
                            bk_idx for _ in range(int(min_exp))
                        )
                        eri_obj_to_remove.extend(
                            eri_idx for _ in range(int(min_exp))
                        )
                        # can simply use the indices of the eri as target
                        # indices for the tensor
                        factored_term *= Pow(
                            self.tensor(indices=eri.idx, wrap_result=False) *
                            S.One / t2.pref,
                            min_exp
                        )
            # - remove the matched eri and denominator objects
            denom = term.cancel_denom_brackets(denom_brackets_to_remove)
            eri = term.cancel_eri_objects(eri_obj_to_remove)
            # - collect the remaining objects in the term and add to result
            factored_term *= term.pref * eri * term.num / denom
            logger.info(f"\nFactoring {self.name} in:\n{term}\nresult:\n"
                        f"{EriOrbenergy(factored_term)}")
            factored += factored_term
        return factored


class t1_2(RegisteredIntermediate):
    """Second order MP singles amplitude."""
    _itmd_type = "t_amplitude"
    _order = 2
    _default_idx = ("i", "a")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        # target_indices
        i, a = get_symbols(self.default_idx)
        # additional contracted indices
        j, k, b, c = get_symbols('jkbc')
        # t2_1 class instance
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the amplitude
        denom = Add(orb_energy(i), -orb_energy(a))
        term1 = (Rational(1, 2) *
                 t2(indices=(i, j, b, c), wrap_result=False) *
                 eri([j, a, b, c]))
        term2 = (Rational(1, 2) *
                 t2(indices=(j, k, a, b), wrap_result=False) *
                 eri([j, k, i, b]))
        return ItmdExpr(term1/denom + term2/denom, (i, a), (j, k, b, c))

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return Amplitude(
            f"{tensor_names.gs_amplitude}2", (indices[1],), (indices[0],)
        )


class t2_2(RegisteredIntermediate):
    """Second order MP doubles amplitude."""
    _itmd_type = "t_amplitude"
    _order = 2
    _default_idx = ("i", "j", "a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2o / 2v)
        k, l, c, d = get_symbols('klcd')
        # t2_1 class instance for generating t2_1 amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the t2_2 amplitude
        denom = Add(
            orb_energy(a), orb_energy(b), -orb_energy(i), -orb_energy(j)
        )
        itmd = S.Zero
        # - 0.5 t2eri_3
        itmd += (- Rational(1, 2) * eri((i, j, k, l)) *
                 t2(indices=(k, l, a, b), wrap_result=False))
        # - 0.5 t2eri_5
        itmd += (- Rational(1, 2) * eri((a, b, c, d)) *
                 t2(indices=(i, j, c, d), wrap_result=False))
        # + (1 - P_ij) (1 - P_ab) P_ij t2eri_4
        ampl = t2(indices=(i, k, a, c), wrap_result=True)
        assert isinstance(ampl, ExprContainer)
        base = ampl * eri((k, b, j, c))
        itmd += Add(
            base.inner, -base.copy().permute((i, j)).inner,
            -base.copy().permute((a, b)).inner,
            base.copy().permute((i, j), (a, b)).inner
        )
        return ItmdExpr(itmd * S.One / denom, (i, j, a, b), (k, l, c, d))

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return Amplitude(
            f"{tensor_names.gs_amplitude}2", indices[2:], indices[:2]
        )


class t3_2(RegisteredIntermediate):
    """Second order MP triples amplitude."""
    _itmd_type = "t_amplitude"
    _order = 2
    _default_idx = ("i", "j", "k", "a", "b", "c")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, k, a, b, c = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        l, d = get_symbols('ld')
        # t2_1 class instance for generating t2_1 amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the t3_2 amplitude
        denom = Add(
            orb_energy(i), orb_energy(j), orb_energy(k),
            -orb_energy(a), -orb_energy(b), -orb_energy(c)
        )
        itmd = S.Zero
        # (1 - P_ik - P_jk) (1 - P_ab - P_ac) <kd||bc> t_ij^ad
        ampl = t2(indices=(i, j, a, d), wrap_result=True)
        assert isinstance(ampl, ExprContainer)
        base = ampl * eri((k, d, b, c))
        itmd += Add(
            base.inner,
            -base.copy().permute((i, k)).inner,
            -base.copy().permute((j, k)).inner,
            -base.copy().permute((a, b)).inner,
            -base.copy().permute((a, c)).inner,
            base.copy().permute((i, k), (a, b)).inner,
            base.copy().permute((i, k), (a, c)).inner,
            base.copy().permute((j, k), (a, b)).inner,
            base.copy().permute((j, k), (a, c)).inner
        )
        # (1 - P_ij - P_ik) (1 - P_ac - P_bc) <jk||lc> t_il^ab
        ampl = t2(indices=(i, l, a, b), wrap_result=True)
        assert isinstance(ampl, ExprContainer)
        base = ampl * eri((j, k, l, c))
        itmd += Add(
            base.inner,
            -base.copy().permute((i, j)).inner,
            -base.copy().permute((i, k)).inner,
            -base.copy().permute((a, c)).inner,
            -base.copy().permute((b, c)).inner,
            base.copy().permute((i, j), (a, c)).inner,
            base.copy().permute((i, j), (b, c)).inner,
            base.copy().permute((i, k), (a, c)).inner,
            base.copy().permute((i, k), (b, c)).inner
        )
        return ItmdExpr(itmd/denom, (i, j, k, a, b, c), (l, d))

    def _build_tensor(self, indices) -> Expr:
        return Amplitude(
            f"{tensor_names.gs_amplitude}2", indices[3:], indices[:3]
        )


class t4_2(RegisteredIntermediate):
    """
    Second order MP quadruple amplitudes in a factorized form that avoids
    the construction of the quadruples denominator.
    """
    _itmd_type = "t_amplitude"
    _order = 2
    _default_idx = ("i", "j", "k", "l", "a", "b", "c", "d")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, k, l, a, b, c, d = get_symbols(self.default_idx)
        # t2_1 class instance
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the t4_2 amplitude
        # (1 - P_ac - P_ad - P_bc - P_bd + P_ac P_bd) (1 - P_jk - P_jl)
        #  t_ij^ab t_kl^cd
        ampl = t2(indices=(i, j, a, b))
        assert isinstance(ampl, ExprContainer)
        base = ampl * t2(indices=(k, l, c, d), wrap_result=False)
        v_permutations = {tuple(tuple()): 1, ((a, c),): -1, ((a, d),): -1,
                          ((b, c),): -1, ((b, d),): -1, ((a, c), (b, d)): +1}
        o_permutations = {tuple(tuple()): 1, ((j, k),): -1, ((j, l),): -1}
        t4 = S.Zero
        for (o_perms, o_factor), (v_perms, v_factor) in \
                itertools.product(o_permutations.items(),
                                  v_permutations.items()):
            perms = o_perms + v_perms
            t4 += Mul(o_factor, v_factor, base.copy().permute(*perms).inner)
        return ItmdExpr(t4, (i, j, k, l, a, b, c, d), None)

    def _build_tensor(self, indices) -> Expr:
        return Amplitude(
            f"{tensor_names.gs_amplitude}2", indices[4:], indices[:4]
        )


class t1_3(RegisteredIntermediate):
    """Third order MP single amplitude."""
    _itmd_type = "t_amplitude"
    _order = 3
    _default_idx = ("i", "a")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, a = get_symbols('ia')
        # generate additional contracted indices (2o / 2v)
        j, k, b, c = get_symbols('jkbc')
        # other intermediate class instances
        t1 = self._registry['t_amplitude']['t1_2']
        t2 = self._registry['t_amplitude']['t2_2']
        t3 = self._registry['t_amplitude']['t3_2']
        if fully_expand:
            t1 = t1.expand_itmd
            t2 = t2.expand_itmd
            t3 = t3.expand_itmd
        else:
            t1 = t1.tensor
            t2 = t2.tensor
            t3 = t3.tensor
        # build the amplitude
        denom = Add(orb_energy(i), -orb_energy(a))
        itmd = S.Zero
        itmd += (Rational(1, 2) * eri([j, a, b, c]) *
                 t2(indices=(i, j, b, c), wrap_result=False))
        itmd += (Rational(1, 2) * eri([j, k, i, b]) *
                 t2(indices=(j, k, a, b), wrap_result=False))
        itmd -= Mul(
            t1(indices=(j, b), wrap_result=False), eri([i, b, j, a])
        )
        itmd += (Rational(1, 4) * eri([j, k, b, c]) *
                 t3(indices=(i, j, k, a, b, c), wrap_result=False))
        # need to keep track of all contracted indices... also contracted
        # indices within each of the second order t-amplitudes
        # -> substitute_contracted indices to minimize the number of contracted
        #    indices
        target = (i, a)
        if fully_expand:
            itmd = ExprContainer(itmd, target_idx=target)
            itmd = itmd.substitute_contracted().inner
            contracted = tuple(sorted(
                [s for s in itmd.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (j, k, b, c)
        return ItmdExpr(itmd * S.One / denom, target, contracted)

    def _build_tensor(self, indices) -> Expr:
        return Amplitude(
            f"{tensor_names.gs_amplitude}3", (indices[1],), (indices[0],))


class t2_3(RegisteredIntermediate):
    """Third order MP double amplitude."""
    _itmd_type = "t_amplitude"
    _order = 3
    _default_idx = ("i", "j", "a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2o / 2v)
        k, l, c, d = get_symbols('klcd')
        # other intermediate class instances
        _t2_1 = self._registry['t_amplitude']['t2_1']
        t1 = self._registry['t_amplitude']['t1_2']
        t2 = self._registry['t_amplitude']['t2_2']
        t3 = self._registry['t_amplitude']['t3_2']
        t4 = self._registry['t_amplitude']['t4_2']
        if fully_expand:
            _t2_1 = _t2_1.expand_itmd
            t1 = t1.expand_itmd
            t2 = t2.expand_itmd
            t3 = t3.expand_itmd
            t4 = t4.expand_itmd
        else:
            _t2_1 = _t2_1.tensor
            t1 = t1.tensor
            t2 = t2.tensor
            t3 = t3.tensor
            t4 = t4.tensor
        # build the amplitude
        denom = Add(
            orb_energy(a), orb_energy(b), -orb_energy(i), -orb_energy(j)
        )
        itmd = S.Zero
        # +(1-P_ij) * <ic||ab> t^c_j(2)
        ampl = t1(indices=(j, c))
        assert isinstance(ampl, ExprContainer)
        base = ampl * eri((i, c, a, b))
        itmd += Add(base.inner, -base.permute((i, j)).inner)
        # +(1-P_ab) * <ij||ka> t^b_k(2)
        ampl = t1(indices=(k, b))
        assert isinstance(ampl, ExprContainer)
        base = ampl * eri((i, j, k, a))
        itmd += Add(base.inner, -base.permute((a, b)).inner)
        # - 0.5 * <ab||cd> t^cd_ij(2)
        itmd -= (Rational(1, 2) * eri((a, b, c, d)) *
                 t2(indices=(i, j, c, d), wrap_result=False))
        # - 0.5 * <ij||kl> t^ab_kl(2)
        itmd -= (Rational(1, 2) * eri((i, j, k, l)) *
                 t2(indices=(k, l, a, b), wrap_result=False))
        # + (1-P_ij)*(1-P_ab) * <jc||kb> t^ac_ik(2)
        ampl = t2(indices=(i, k, a, c))
        assert isinstance(ampl, ExprContainer)
        base = ampl * eri((j, c, k, b))
        itmd += Add(
            base.inner,
            -base.copy().permute((i, j)).inner,
            -base.copy().permute((a, b)).inner,
            base.copy().permute((i, j), (a, b)).inner
        )
        # + 0.5 * (1-P_ab) * <ka||cd> t^bcd_ijk(2)
        ampl = t3(indices=(i, j, k, b, c, d))
        assert isinstance(ampl, ExprContainer)
        base = ampl * eri((k, a, c, d))
        itmd += (Rational(1, 2) * base.inner
                 - Rational(1, 2) * base.copy().permute((a, b)).inner)
        # + 0.5 * (1-P_ij) <kl||ic> t^abc_jkl(2)
        ampl = t3(indices=(j, k, l, a, b, c))
        assert isinstance(ampl, ExprContainer)
        base = ampl * eri((k, l, i, c))
        itmd += (Rational(1, 2) * base.inner
                 - Rational(1, 2) * base.copy().permute((i, j)).inner)
        # + 0.25 <kl||cd> t^abcd_ijkl(2)
        itmd += (Rational(1, 4) * eri((k, l, c, d)) *
                 t4(indices=(i, j, k, l, a, b, c, d), wrap_result=False))
        # - 0.25 <kl||cd> t^ab_ij(1) t^kl_cd(1)
        itmd -= (Rational(1, 4) * eri((k, l, c, d)) *
                 _t2_1(indices=(i, j, a, b), wrap_result=False) *
                 _t2_1(indices=(k, l, c, d), wrap_result=False))
        # minimize the number of contracted indices
        target = (i, j, a, b)
        if fully_expand:
            itmd = ExprContainer(itmd, target_idx=target)
            itmd = itmd.substitute_contracted().inner
            contracted = tuple(sorted(
                [s for s in itmd.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (k, l, c, d)
        return ItmdExpr(itmd * S.One / denom, target, contracted)

    def _build_tensor(self, indices) -> Expr:
        return Amplitude(
            f"{tensor_names.gs_amplitude}3", indices[2:], indices[:2]
        )


class t2_1_re_residual(RegisteredIntermediate):
    """
    Residual of the first order RE doubles amplitudes.
    """
    _itmd_type = "re_residual"
    _order = 2  # according to MP the maximum order of the residual is 2
    _default_idx = ("i", "j", "a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        # re intermediates can not be fully expanded, but add the bool
        # anyway for a consistent interface
        _ = fully_expand
        i, j, a, b = get_symbols(self.default_idx)
        # additional contracted indices
        k, l, c, d = get_symbols('klcd')
        # t2_1 class instance
        t2 = self._registry['t_amplitude']['t2_1']

        itmd = S.Zero

        # (1 - P_ij)(1 - P_ab) <ic||ka> t_jk^bc
        ampl = t2.tensor(indices=[j, k, b, c])
        assert isinstance(ampl, ExprContainer)
        base = eri([i, c, k, a]) * ampl
        itmd += Add(
            base.inner,
            -base.copy().permute((i, j)).inner,
            -base.copy().permute((a, b)).inner,
            base.copy().permute((i, j), (a, b)).inner
        )
        # (1 - P_ab) f_ac t_ij^bc
        ampl = t2.tensor(indices=[i, j, b, c])
        assert isinstance(ampl, ExprContainer)
        base = fock([a, c]) * ampl
        itmd += Add(base.inner, -base.copy().permute((a, b)).inner)
        # (1 - P_ij) f_jk t_ik^ab
        ampl = t2.tensor(indices=[i, k, a, b])
        assert isinstance(ampl, ExprContainer)
        base = fock([j, k]) * ampl
        itmd += Add(base.inner, -base.copy().permute((i, j)).inner)
        # - 0.5 * <ab||cd> t_ij^cd
        itmd -= (Rational(1, 2) * eri((a, b, c, d)) *
                 t2.tensor(indices=(i, j, c, d), wrap_result=False))
        # -0.5 * <ij||kl> t_kl^ab
        itmd -= (Rational(1, 2) * eri((i, j, k, l)) *
                 t2.tensor(indices=(k, l, a, b), wrap_result=False))
        # + <ij||ab>
        itmd += eri((i, j, a, b))
        target = (i, j, a, b)
        contracted = (k, l, c, d)
        return ItmdExpr(itmd, target, contracted)

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        # placeholder for 0, will be replaced in factor_intermediate
        return AntiSymmetricTensor("Zero", indices[:2], indices[2:])


class t1_2_re_residual(RegisteredIntermediate):
    """
    Residual of the second order RE singles amplitudes.
    """
    _itmd_type = "re_residual"
    _order = 3  # according to MP the maximum order of the residual is 3
    _default_idx = ("i", "a")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        _ = fully_expand
        i, a = get_symbols(self.default_idx)
        # additional contracted indices
        j, k, b, c = get_symbols('jkbc')

        # t amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        ts2 = self._registry['t_amplitude']['t1_2']

        itmd = S.Zero

        # - {V^{ib}_{ja}} {t2^{b}_{j}}
        itmd += Mul(
            -eri([i, b, j, a]),
            ts2.tensor(indices=[j, b], wrap_result=False)
        )
        # + {f^{a}_{b}} {t2^{b}_{i}}
        itmd += Mul(
            fock([a, b]),
            ts2.tensor(indices=[i, b], wrap_result=False)
        )
        # - {f^{i}_{j}} {t2^{a}_{j}}
        itmd -= Mul(
            fock([i, j]),
            ts2.tensor(indices=[j, a], wrap_result=False)
        )
        # + \frac{{V^{ja}_{bc}} {t1^{bc}_{ij}}}{2}
        itmd += (Rational(1, 2) * eri([j, a, b, c])
                 * t2.tensor(indices=[i, j, b, c], wrap_result=False))
        # + \frac{{V^{jk}_{ib}} {t1^{ab}_{jk}}}{2}
        itmd += (Rational(1, 2) * eri([j, k, i, b])
                 * t2.tensor(indices=[j, k, a, b], wrap_result=False))
        # - {f^{j}_{b}} {t1^{ab}_{ij}}
        itmd -= Mul(
            fock([j, b]),
            t2.tensor(indices=[i, j, a, b], wrap_result=False)
        )
        target = (i, a)
        contracted = (j, k, b, c)
        return ItmdExpr(itmd, target, contracted)

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        # placeholder for 0, will be replaced in factor_intermediate
        return AntiSymmetricTensor("Zero", (indices[0],), (indices[1],))


class t2_2_re_residual(RegisteredIntermediate):
    """
    Residual of the second order RE doubles amplitudes.
    """
    _itmd_type = "re_residual"
    _order = 3  # according to MP the maximum order of the residual is 3
    _default_idx = ("i", "j", "a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        _ = fully_expand
        i, j, a, b = get_symbols(self.default_idx)
        # additional contracted indices
        k, l, c, d = get_symbols('klcd')
        # t2_1 class instance
        t2 = self._registry['t_amplitude']['t2_2']

        itmd = S.Zero

        # (1 - P_ij)(1 - P_ab) <ic||ka> t_jk^bc
        ampl = t2.tensor(indices=[j, k, b, c])
        assert isinstance(ampl, ExprContainer)
        base = eri([i, c, k, a]) * ampl
        itmd += Add(
            base.inner,
            -base.copy().permute((i, j)).inner,
            -base.copy().permute((a, b)).inner,
            base.copy().permute((i, j), (a, b)).inner
        )
        # (1 - P_ab) f_ac t_ij^bc
        ampl = t2.tensor(indices=[i, j, b, c])
        assert isinstance(ampl, ExprContainer)
        base = fock([a, c]) * ampl
        itmd += Add(
            base.inner, -base.copy().permute((a, b)).inner
        )
        # (1 - P_ij) f_jk t_ik^ab
        ampl = t2.tensor(indices=[i, k, a, b])
        assert isinstance(ampl, ExprContainer)
        base = fock([j, k]) * ampl
        itmd += Add(
            base.inner, -base.copy().permute((i, j)).inner
        )
        # - 0.5 * <ab||cd> t_ij^cd
        itmd -= (Rational(1, 2) * eri((a, b, c, d)) *
                 t2.tensor(indices=(i, j, c, d), wrap_result=False))
        # -0.5 * <ij||kl> t_kl^ab
        itmd -= (Rational(1, 2) * eri((i, j, k, l)) *
                 t2.tensor(indices=(k, l, a, b), wrap_result=False))
        target = (i, j, a, b)
        contracted = (k, l, c, d)
        return ItmdExpr(itmd, target, contracted)

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        # placeholder for 0, will be replaced in factor_intermediate
        return AntiSymmetricTensor("Zero", indices[:2], indices[2:])


class p0_2_oo(RegisteredIntermediate):
    """
    Second order contribution to the occupied occupied block of the MP
    one-particle density matrix.
    """
    _itmd_type = "mp_density"
    _order = 2
    _default_idx = ("i", "j")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j = get_symbols(self.default_idx)
        # additional contracted indices (1o / 2v)
        k, a, b = get_symbols('kab')
        # t2_1 class instance
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the density
        p0 = (- Rational(1, 2) *
              t2(indices=(i, k, a, b), wrap_result=False) *
              t2(indices=(j, k, a, b), wrap_result=False))
        return ItmdExpr(p0, (i, j), (k, a, b))

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}2", (indices[0],), (indices[1],), 1
        )


class p0_2_vv(RegisteredIntermediate):
    """
    Second order contribution to the virtual virtual block of the MP
    one-particle density matrix.
    """
    _itmd_type = "mp_density"
    _order = 2
    _default_idx = ("a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        a, b = get_symbols(self.default_idx)
        # additional contracted indices (2o / 1v)
        i, j, c = get_symbols('ijc')
        # t2_1 class instance
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the density
        p0 = (Rational(1, 2) *
              t2(indices=(i, j, a, c), wrap_result=False) *
              t2(indices=(i, j, b, c), wrap_result=False))
        return ItmdExpr(p0, (a, b), (i, j, c))

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}2", (indices[0],), (indices[1],), 1)


class p0_3_oo(RegisteredIntermediate):
    """
    Third order contribution to the occupied occupied block of the MP
    one-particle density matrix.
    """
    _itmd_type = "mp_density"
    _order = 3
    _default_idx = ("i", "j")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 2v)
        k, a, b = get_symbols('kab')
        # t amplitude cls
        t2 = self._registry['t_amplitude']['t2_1']
        td2 = self._registry['t_amplitude']['t2_2']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        td2 = td2.expand_itmd if fully_expand else td2.tensor
        # build the density
        p0 = (- Rational(1, 2) *
              t2(indices=(i, k, a, b), wrap_result=False) *
              td2(indices=(j, k, a, b), wrap_result=False))
        p0 += p0.subs({i: j, j: i}, simultaneous=True)

        target = (i, j)
        if fully_expand:
            p0 = ExprContainer(
                p0, target_idx=target
            ).substitute_contracted().inner
            contracted = tuple(sorted(
                [s for s in p0.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (k, a, b)
        return ItmdExpr(p0, target, contracted)

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}3", (indices[0],), (indices[1],), 1)


class p0_3_ov(RegisteredIntermediate):
    """
    Third order contribution to the occupied virtual block of the MP
    one-particle density matrix.
    """
    _itmd_type = "mp_density"
    _order = 3
    _default_idx = ("i", "a")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, a = get_symbols(self.default_idx)
        # generate additional contracted indices (2o / 2v)
        j, k, b, c = get_symbols('jkbc')
        # t_amplitude cls instances
        t2 = self._registry['t_amplitude']['t2_1']
        ts2 = self._registry['t_amplitude']['t1_2']
        tt2 = self._registry['t_amplitude']['t3_2']
        ts3 = self._registry['t_amplitude']['t1_3']
        if fully_expand:
            t2 = t2.expand_itmd
            ts2 = ts2.expand_itmd
            tt2 = tt2.expand_itmd
            ts3 = ts3.expand_itmd
        else:
            t2 = t2.tensor
            ts2 = ts2.tensor
            tt2 = tt2.tensor
            ts3 = ts3.tensor
        p0 = S.Zero
        # build the density
        # - t^ab_ij(1) t^b_j(2)
        p0 += (
            S.NegativeOne * t2(indices=(i, j, a, b), wrap_result=False) *
            ts2(indices=(j, b), wrap_result=False)
        )
        # - 0.25 * t^bc_jk(1) t^abc_ijk(2)
        p0 -= (Rational(1, 4) *
               t2(indices=(j, k, b, c), wrap_result=False) *
               tt2(indices=(i, j, k, a, b, c), wrap_result=False))
        # + t^a_i(3)
        p0 += ts3(indices=(i, a), wrap_result=False)

        target = (i, a)
        if fully_expand:
            p0 = ExprContainer(
                p0, target_idx=target
            ).substitute_contracted().inner
            contracted = tuple(sorted(
                [s for s in p0.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (j, k, b, c)
        return ItmdExpr(p0, target, contracted)

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}3", (indices[0],), (indices[1],), 1)


class p0_3_vv(RegisteredIntermediate):
    """
    Third order contribution to the virtual virtual block of the MP
    one-particle density matrix.
    """
    _itmd_type = "mp_density"
    _order = 3
    _default_idx = ("a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        a, b = get_symbols(self.default_idx)
        # additional contracted indices (2o / 1v)
        i, j, c = get_symbols('ijc')
        # t_amplitude cls instances
        t2 = self._registry['t_amplitude']['t2_1']
        td2 = self._registry['t_amplitude']['t2_2']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        td2 = td2.expand_itmd if fully_expand else td2.tensor
        # build the density
        p0 = (Rational(1, 2) *
              t2(indices=(i, j, a, c), wrap_result=False) *
              td2(indices=(i, j, b, c), wrap_result=False))
        p0 += p0.subs({a: b, b: a}, simultaneous=True)

        target = (a, b)
        if fully_expand:
            p0 = ExprContainer(
                p0, target_idx=target
            ).substitute_contracted().inner
            contracted = tuple(sorted(
                [s for s in p0.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (i, j, c)
        return ItmdExpr(p0, target, contracted)

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}3", (indices[0],), (indices[1],), 1)


class t2eri_1(RegisteredIntermediate):
    """t2eri1 in adcc / pi1 in libadc."""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "j", "k", "a")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, k, a = get_symbols(self.default_idx)
        # generate additional contracted indices (2v)
        b, c = get_symbols('bc')
        # t2_1 class instance for generating t2_1 amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = Mul(
            t2(indices=(i, j, b, c), wrap_result=False),
            eri((k, a, b, c))
        )
        return ItmdExpr(t2eri, (i, j, k, a), (b, c))

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor('t2eri1', indices[:2], indices[2:])


class t2eri_2(RegisteredIntermediate):
    """t2eri2 in adcc / pi2 in libadc."""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "j", "k", "a")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, k, a = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        b, l = get_symbols('bl')  # noqa E741
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = Mul(
            t2(indices=(i, l, a, b), wrap_result=False),
            eri((l, k, j, b))
        )
        return ItmdExpr(t2eri, (i, j, k, a), (b, l))

    def _build_tensor(self, indices: Sequence[Index]) -> NonSymmetricTensor:
        return NonSymmetricTensor('t2eri2', indices)


class t2eri_3(RegisteredIntermediate):
    """t2eri3 in adcc / pi3 in libadc."""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "j", "a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2o)
        k, l = get_symbols('kl')  # noqa E741
        # t2_1 class instance for generating t2_1 amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = Mul(
            t2(indices=(k, l, a, b), wrap_result=False),
            eri((i, j, k, l))
        )
        return ItmdExpr(t2eri, (i, j, a, b), (k, l))

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor('t2eri3', indices[:2], indices[2:])


class t2eri_4(RegisteredIntermediate):
    """t2eri4 in adcc / pi4 in libadc."""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "j", "a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        k, c = get_symbols('kc')
        # t2_1 class instance for generating t2_1 amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = Mul(
            t2(indices=(j, k, a, c), wrap_result=False),
            eri((k, b, i, c))
        )
        return ItmdExpr(t2eri, (i, j, a, b), (k, c))

    def _build_tensor(self, indices: Sequence[Index]) -> NonSymmetricTensor:
        return NonSymmetricTensor('t2eri4', indices)


class t2eri_5(RegisteredIntermediate):
    """t2eri5 in adcc / pi5 in libadc."""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "j", "a", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2v)
        c, d = get_symbols('cd')
        # t2_1 class instance for generating t2_1 amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = Mul(
            t2(indices=(i, j, c, d), wrap_result=False),
            eri((a, b, c, d))
        )
        return ItmdExpr(t2eri, (i, j, a, b), (c, d))

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor('t2eri5', indices[:2], indices[2:])


class t2eri_6(RegisteredIntermediate):
    """t2eri6 in adcc / pi6 in libadc."""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "a", "b", "c")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, a, b, c = get_symbols(self.default_idx)
        # generate additional contracted indices (2o)
        j, k = get_symbols('jk')
        # t2_1 class instance for generating t2_1 amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = Mul(
            t2(indices=(j, k, b, c), wrap_result=False),
            eri((j, k, i, a))
        )
        return ItmdExpr(t2eri, (i, a, b, c), (j, k))

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor('t2eri6', indices[:2], indices[2:])


class t2eri_7(RegisteredIntermediate):
    """t2eri7 in adcc / pi7 in libadc."""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "a", "b", "c")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, a, b, c = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        j, d = get_symbols('jd')
        # t2_1 class instance for generating t2_1 amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = Mul(
            t2(indices=(i, j, b, d), wrap_result=False),
            eri((j, c, a, d))
        )
        return ItmdExpr(t2eri, (i, a, b, c), (j, d))

    def _build_tensor(self, indices: Sequence[Index]) -> NonSymmetricTensor:
        return NonSymmetricTensor('t2eri7', indices)


class t2eri_A(RegisteredIntermediate):
    """pia intermediate in libadc"""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "j", "k", "a")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, j, k, a = get_symbols(self.default_idx)
        # t2eri cls instances for generating the itmd
        pi1 = self._registry['misc']['t2eri_1']
        pi2 = self._registry['misc']['t2eri_2']
        pi1 = pi1.expand_itmd if fully_expand else pi1.tensor
        pi2 = pi2.expand_itmd if fully_expand else pi2.tensor
        # build the itmd
        pia = (
            Rational(1, 2) * pi1(indices=(i, j, k, a), wrap_result=False)
            + pi2(indices=(i, j, k, a), wrap_result=False)
            + S.NegativeOne * pi2(indices=(j, i, k, a), wrap_result=False)
        )
        target = (i, j, k, a)
        if fully_expand:
            pia = ExprContainer(
                pia, target_idx=target
            ).substitute_contracted().inner
            contracted = tuple(sorted(
                [s for s in pia.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = tuple()
        return ItmdExpr(pia, target, contracted)

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor('t2eriA', indices[:2], indices[2:])


class t2eri_B(RegisteredIntermediate):
    """pib intermediate in libadc"""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "a", "b", "c")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, a, b, c = get_symbols(self.default_idx)
        # t2eri cls instances for generating the itmd
        pi6 = self._registry['misc']['t2eri_6']
        pi7 = self._registry['misc']['t2eri_7']
        pi6 = pi6.expand_itmd if fully_expand else pi6.tensor
        pi7 = pi7.expand_itmd if fully_expand else pi7.tensor
        # build the itmd
        pib = (-Rational(1, 2) * pi6(indices=(i, a, b, c), wrap_result=False)
               + pi7(indices=(i, a, b, c), wrap_result=False)
               + S.NegativeOne * pi7(indices=(i, a, c, b), wrap_result=False))
        target = (i, a, b, c)
        if fully_expand:
            pib = ExprContainer(
                pib, target_idx=target
            ).substitute_contracted().inner
            contracted = tuple(sorted(
                [s for s in pib.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = tuple()
        return ItmdExpr(pib, target, contracted)

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor('t2eriB', indices[:2], indices[2:])


class t2sq(RegisteredIntermediate):
    """t2sq intermediate from adcc and libadc."""
    _itmd_type = "misc"
    _order = 2
    _default_idx = ("i", "a", "j", "b")

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True) -> ItmdExpr:
        i, a, j, b = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        c, k = get_symbols('ck')
        # t2_1 class instance for generating t2_1 amplitudes
        t2 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        itmd = Mul(
            t2(indices=(i, k, a, c), wrap_result=False),
            t2(indices=(j, k, b, c), wrap_result=False)
        )
        return ItmdExpr(itmd, (i, a, j, b), (k, c))

    def _build_tensor(self, indices: Sequence[Index]) -> Expr:
        return AntiSymmetricTensor('t2sq', indices[:2], indices[2:], 1)


def eri(idx: Sequence[str] | Sequence[Index]) -> Expr:
    """
    Builds an antisymmetric electron repulsion integral.
    Indices may be provided as list of sympy symbols or as string.
    """
    idx = get_symbols(idx)
    if len(idx) != 4:
        raise Inputerror(f'4 indices required to build a ERI. Got: {idx}.')
    return AntiSymmetricTensor(tensor_names.eri, idx[:2], idx[2:])


def fock(idx: Sequence[Index] | Sequence[str]) -> Expr:
    """
    Builds a fock matrix element.
    Indices may be provided as list of sympy symbols or as string.
    """
    idx = get_symbols(idx)
    if len(idx) != 2:
        raise Inputerror('2 indices required to build a Fock matrix element.'
                         f'Got: {idx}.')
    return AntiSymmetricTensor(tensor_names.fock, idx[:1], idx[1:])


def orb_energy(idx: Index | Sequence[str] | Sequence[Index]
               ) -> NonSymmetricTensor:
    """
    Builds an orbital energy.
    Indices may be provided as list of sympy symbols or as string.
    """
    idx = get_symbols(idx)
    if len(idx) != 1:
        raise Inputerror("1 index required to build a orbital energy. Got: "
                         f"{idx}.")
    return NonSymmetricTensor(tensor_names.orb_energy, idx)
