from .indices import get_symbols, order_substitutions, sort_idx_canonical
from .indices import Indices
from .indices import Index
from .logger import logger
from .misc import Inputerror, Singleton, cached_property, cached_member
from . import expr_container as e
from .eri_orbenergy import EriOrbenergy
from .sympy_objects import NonSymmetricTensor, AntiSymmetricTensor, Amplitude
from .symmetry import LazyTermMap
from .spatial_orbitals import allowed_spin_blocks
from .tensor_names import tensor_names

from sympy import S, Rational, Pow

from collections import namedtuple, Counter
from itertools import product, chain


base_expr = namedtuple('base_expr', ['expr', 'target', 'contracted'])


class Intermediates(metaclass=Singleton):
    """
    Manages all defined intermediates.
    New intermediates can be defined by inheriting from
    'RegisteredIntermediate'.
    """

    def __init__(self):
        self._registered: dict = RegisteredIntermediate()._registry
        self._available: dict = {
            name: obj for objects in self._registered.values()
            for name, obj in objects.items()
        }

    @property
    def available(self) -> dict:
        """
        Returns all available intermediates using their name as dict key.
        """
        return self._available

    @property
    def types(self) -> list[str]:
        """Returns all available types of intermediates."""
        return list(self._registered.keys())

    def __getattr__(self, attr) -> dict:
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
    _registry: dict[str, dict[str]] = {}

    def __init_subclass__(cls):
        if (itmd_type := cls._itmd_type) not in cls._registry:
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
        if (order := getattr(self, '_order', None)) is None:
            raise AttributeError(f"No order defined for {self.name}.")
        return order

    @property
    def default_idx(self) -> tuple[str]:
        """Names of the default indices of the intermediate."""
        if (idx := getattr(self, '_default_idx', None)) is None:
            raise AttributeError(f"No default indices defined for {self.name}")
        return idx

    @property
    def itmd_type(self) -> str:
        """The type of the intermediate."""
        if (itmd_type := getattr(self, '_itmd_type', None)) is None:
            raise AttributeError(f"No itmd_type defined for {self.name}.")
        return itmd_type

    def validate_indices(self, indices: str = None) -> tuple[Index]:
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

    def expand_itmd(self, indices: str = None, return_sympy: bool = False,
                    fully_expand: bool = True):
        """
        Expands the intermediate into orbital energies and ERI.

        Parameters
        ----------
        indices : str, optional
            The names of the indices of the intermediate. By default the
            default indices (defined on the itmd class) will be used.
        return_sympy : bool, optional
            Whether to return the unwrapped sympy object (default: False).
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
        subs = {}
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
        subs = order_substitutions(subs)
        itmd = expanded_itmd.expr.subs(subs)

        if itmd is S.Zero and expanded_itmd.expr is not S.Zero:
            raise ValueError(f"The substitutions {subs} are not valid for "
                             f"{expanded_itmd.expr}.")

        if not return_sympy:
            itmd = e.Expr(itmd, target_idx=indices)
        return itmd

    def tensor(self, indices: str = None, return_sympy: bool = False):
        """
        Returns the itmd tensor.

        Parameters
        ----------
        indices : str, optional
            The names of the indices of the intermediate. By default the
            default indices (defined on the itmd class) will be used.
        return_sympy : bool, optional
            Whether to return the unwrapped sympy object (default: False).
        """
        # check that the provided indices are sufficient for the itmd
        indices = self.validate_indices(indices)

        # build the tensor object
        tensor = self._build_tensor(indices=indices)
        if return_sympy:
            return tensor
        else:
            if isinstance(tensor, AntiSymmetricTensor):
                if tensor.bra_ket_sym is S.One:  # bra ket symmetry
                    return e.Expr(tensor, sym_tensors=[tensor.name])
                elif tensor.bra_ket_sym is S.NegativeOne:  # bra ket anisym
                    return e.Expr(tensor, antisym_tensors=[tensor.name])
            return e.Expr(tensor)

    @cached_property
    def tensor_symmetry(self) -> dict:
        """
        Determines the symmetry of the itmd tensor object using the
        default indices, e.g., ijk/abc triples symmetry for t3_2.
        """
        return self.tensor().terms[0].symmetry()

    @cached_property
    def allowed_spin_blocks(self) -> tuple[str]:
        """Determines all non-zero spin block of the intermediate."""

        target_idx = self.default_idx
        itmd = self.expand_itmd(indices=target_idx, fully_expand=False)
        return allowed_spin_blocks(itmd.expand(), target_idx)

    @cached_member
    def itmd_term_map(self,
                      factored_itmds: tuple[str] = tuple()) -> LazyTermMap:
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
    def _prepare_itmd(self, factored_itmds: tuple[str] = tuple()) -> e.Expr:
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
        itmd: e.Expr = self.expand_itmd().expand().make_real()
        reduced = chain.from_iterable(
            factor_denom(sub_expr) for sub_expr in factor_eri_parts(itmd)
        )
        itmd = e.Expr(0, **itmd.assumptions)
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

    def factor_itmd(self, expr: e.Expr, factored_itmds: tuple[str] = tuple(),
                    max_order: int = None) -> e.Expr:
        """
        Factors the intermediate in an expression assuming a real orbital
        basis.

        Parameters
        ----------
        expr : Expr
            Expression in which to factor intermediates.
        factored_itmds : tuple[str], optional
            Names of other intermediates that have already been factored in
            the expression. It is necessary to factor those intermediates in
            the current intermediate definition as well, because the
            definition might change. By default the fully expanded version
            of the intermediate will be used.
        max_order : int, optional
            The maximum perturbation theoretical order of intermediates
            to consider.
        """

        from .factor_intermediates import (_factor_long_intermediate,
                                           _factor_short_intermediate,
                                           FactorizationTermData)

        if not isinstance(expr, e.Expr):
            raise TypeError("Expr to factor needs to be provided as "
                            f"{e.Expr} instance.")
        if not expr.real:
            raise NotImplementedError("Intermediates only implemented for "
                                      "a real orbital basis.")

        # ensure that the previously factored intermediates
        # are provided as tuple -> can use them as dict key
        if factored_itmds is None:
            factored_itmds = tuple()
        elif not isinstance(factored_itmds, tuple):
            factored_itmds = tuple(factored_itmds)

        # can not factor if the expr is just a number or the intermediate
        # has already been factored or the order of the pt order of the
        # intermediate is to high.
        # also it does not make sense to factor t4_2 again, because of the
        # used factorized form.
        if expr.sympy.is_number or self.name in factored_itmds or \
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
            term_is_relevant = [term.order >= self.order and
                                any(o.exponent < 0 and
                                    o.contains_only_orb_energies
                                    for o in term.objects)
                                for term in terms]
        else:
            term_is_relevant = [term.order >= self.order for term in terms]
        # no term has a denominator or a sufficient pt order
        # -> can't factor the itmd
        if not any(term_is_relevant):
            return expr

        # determine the maximum pt order present in the expr (order is cached)
        max_order = max(term.order for term in terms)

        # build a new expr that only contains the relevant terms
        remainder = 0
        to_factor = e.Expr(0, **expr.assumptions)
        for term, is_relevant in zip(terms, term_is_relevant):
            if is_relevant:
                to_factor += term
            else:
                remainder += term.sympy

        # - prepare the itmd for factorization and extract data to speed
        #   up the later comparison
        itmd_expr = self._prepare_itmd(factored_itmds=factored_itmds)
        itmd = tuple(EriOrbenergy(term).canonicalize_sign()
                     for term in itmd_expr.terms)
        itmd_data = tuple(FactorizationTermData(term) for term in itmd)

        # factor the intermediate in the expr
        if len(itmd) == 1:  # short intermediate that consists of a single term
            factored = _factor_short_intermediate(
                to_factor, itmd[0], itmd_data[0], self
            )
            factored += remainder
        else:  # long intermediate that consists of multiple terms
            itmd_term_map = self.itmd_term_map(factored_itmds)
            for _ in range(max_order // self.order):
                to_factor = _factor_long_intermediate(
                    to_factor, itmd, itmd_data, itmd_term_map, self
                )
            factored = to_factor + remainder
        return factored


# -----------------------------------------------------------------------------
# INTERMEDIATE DEFINITIONS:


class t2_1(RegisteredIntermediate):
    """First order MP doubles amplitude."""
    _itmd_type: str = 't_amplitude'  # type has to be a class variable
    _order: int = 1
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        # build a basic version of the intermediate using minimal indices
        # 'like on paper'
        i, j, a, b = get_symbols(self.default_idx)
        denom = orb_energy(a) + orb_energy(b) - orb_energy(i) - orb_energy(j)
        return base_expr(eri((a, b, i, j)) / denom, (i, j, a, b), None)

    def _build_tensor(self, indices) -> Amplitude:
        # guess its not worth caching here. Maybe if used a lot.
        # build the tensor
        return Amplitude(
            f"{tensor_names.gs_amplitude}1", indices[2:], indices[:2]
        )

    def factor_itmd(self, expr: e.Expr, factored_itmds: list[str] = None,
                    max_order: int = None):
        """Factors the t2_1 intermediate in an expression assuming a real
           orbital basis."""

        if not isinstance(expr, e.Expr):
            raise Inputerror("Expr to factor needs to be an instance of "
                             f"{e.Expr}.")
        if not expr.real:
            raise NotImplementedError("Intermediates only implemented for a "
                                      "real orbital basis.")
        # do we have something to factor? did we already factor the itmd?
        if expr.sympy.is_number or \
                (factored_itmds and self.name in factored_itmds):
            return expr

        # no need to determine max order for a first order intermediate
        if max_order is not None and max_order < self.order:
            return expr

        # prepare the itmd and extract information
        t2 = self.expand_itmd().make_real()
        t2 = EriOrbenergy(t2).canonicalize_sign()
        t2_eri: e.Obj = t2.eri.objects[0]
        t2_eri_descr: str = t2_eri.description(include_exponent=False,
                                               include_target_idx=False)
        t2_denom = t2.denom.sympy
        t2_eri_idx: tuple = t2_eri.idx

        expr = expr.expand()

        factored = 0
        for term in expr.terms:
            term = EriOrbenergy(term)  # split the term

            if term.denom.sympy.is_number:  # term needs to have a denominator
                factored += term.expr
                continue
            term = term.canonicalize_sign()  # fix the sign of the denominator

            brackets = term.denom_brackets
            removed_brackets = set()
            factored_term = 1
            eri_obj_to_remove = []
            denom_brackets_to_remove = []
            for eri_idx, eri in enumerate(term.eri.objects):
                # - compare the eri objects (check if we have a oovv eri)
                #   coupling is not relevant for t2_1 (only a single object)
                descr = eri.description(include_exponent=False,
                                        include_target_idx=False)
                if descr != t2_eri_descr:
                    continue
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
                    if isinstance(bk, e.Expr):
                        bk_exponent = 1
                        bk = bk.sympy
                    else:
                        bk, bk_exponent = bk.base_and_exponent
                    # found matching bracket in denominator
                    if bk == sub_t2_denom:
                        # can possibly factor multiple times, depending
                        # on the exponent of the eri and the denominator
                        min_exp = min(eri_exp, bk_exponent)
                        # are we removing the bracket completely?
                        if min_exp == bk_exponent:
                            removed_brackets.add(bk_idx)
                        # found matching eri and denominator
                        # replace eri and bracket by a t2_1 tensor
                        denom_brackets_to_remove.extend(
                            bk_idx for _ in range(min_exp)
                        )
                        eri_obj_to_remove.extend(
                            eri_idx for _ in range(min_exp)
                        )
                        # can simply use the indices of the eri as target
                        # indices for the tensor
                        factored_term *= Pow(
                            self.tensor(indices=eri.idx, return_sympy=True) /
                            t2.pref,
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
    _itmd_type: str = "t_amplitude"
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a')
    _min_n_terms = 2

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        # target_indices
        i, a = get_symbols(self.default_idx)
        # additional contracted indices
        j, k, b, c = get_symbols('jkbc')
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the amplitude
        denom = orb_energy(i) - orb_energy(a)
        term1 = (Rational(1, 2) *
                 t2(indices=(i, j, b, c), return_sympy=True) *
                 eri([j, a, b, c]))
        term2 = (Rational(1, 2) *
                 t2(indices=(j, k, a, b), return_sympy=True) *
                 eri([j, k, i, b]))
        return base_expr(term1/denom + term2/denom, (i, a), (j, k, b, c))

    def _build_tensor(self, indices) -> Amplitude:
        return Amplitude(
            f"{tensor_names.gs_amplitude}2", (indices[1],), (indices[0],)
        )


class t2_2(RegisteredIntermediate):
    """Second order MP doubles amplitude."""
    _itmd_type: str = 't_amplitude'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2o / 2v)
        k, l, c, d = get_symbols('klcd')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the t2_2 amplitude
        denom = (orb_energy(a) + orb_energy(b) - orb_energy(i) - orb_energy(j))
        # - 0.5 t2eri_3
        itmd = (- Rational(1, 2) * eri((i, j, k, l)) *
                t2(indices=(k, l, a, b), return_sympy=True))
        # - 0.5 t2eri_5
        itmd += (- Rational(1, 2) * eri((a, b, c, d)) *
                 t2(indices=(i, j, c, d), return_sympy=True))
        # + (1 - P_ij) (1 - P_ab) P_ij t2eri_4
        base = (
            t2(indices=(i, k, a, c)) * eri((k, b, j, c))
        )
        itmd += (base.sympy - base.copy().permute((i, j)).sympy
                 - base.copy().permute((a, b)).sympy
                 + base.copy().permute((i, j), (a, b)).sympy)
        return base_expr(itmd / denom, (i, j, a, b), (k, l, c, d))

    def _build_tensor(self, indices) -> Amplitude:
        return Amplitude(
            f"{tensor_names.gs_amplitude}2", indices[2:], indices[:2]
        )


class t3_2(RegisteredIntermediate):
    """Second order MP triples amplitude."""
    _itmd_type: str = 't_amplitude'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'a', 'b', 'c')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, k, a, b, c = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        l, d = get_symbols('ld')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the t3_2 amplitude
        denom = (orb_energy(i) + orb_energy(j) + orb_energy(k)
                 - orb_energy(a) - orb_energy(b) - orb_energy(c))
        # (1 - P_ik - P_jk) (1 - P_ab - P_ac) <kd||bc> t_ij^ad
        base = t2(indices=(i, j, a, d)) * eri((k, d, b, c))
        itmd = (base.sympy - base.copy().permute((i, k)).sympy
                - base.copy().permute((j, k)).sympy
                - base.copy().permute((a, b)).sympy
                - base.copy().permute((a, c)).sympy
                + base.copy().permute((i, k), (a, b)).sympy
                + base.copy().permute((i, k), (a, c)).sympy
                + base.copy().permute((j, k), (a, b)).sympy
                + base.copy().permute((j, k), (a, c)).sympy)
        # (1 - P_ij - P_ik) (1 - P_ac - P_bc) <jk||lc> t_il^ab
        base = t2(indices=(i, l, a, b)) * eri((j, k, l, c))
        itmd += (base.sympy - base.copy().permute((i, j)).sympy
                 - base.copy().permute((i, k)).sympy
                 - base.copy().permute((a, c)).sympy
                 - base.copy().permute((b, c)).sympy
                 + base.copy().permute((i, j), (a, c)).sympy
                 + base.copy().permute((i, j), (b, c)).sympy
                 + base.copy().permute((i, k), (a, c)).sympy
                 + base.copy().permute((i, k), (b, c)).sympy)
        return base_expr(itmd/denom, (i, j, k, a, b, c), (l, d))

    def _build_tensor(self, indices) -> Amplitude:
        return Amplitude(
            f"{tensor_names.gs_amplitude}2", indices[3:], indices[:3]
        )


class t4_2(RegisteredIntermediate):
    """
    Second order MP quadruple amplitudes in a factorized form that avoids
    the construction of the quadruples denominator.
    """
    _itmd_type: str = 't_amplitude'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'l', 'a', 'b', 'c', 'd')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):

        i, j, k, l, a, b, c, d = get_symbols(self.default_idx)
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the t4_2 amplitude
        # (1 - P_ac - P_ad - P_bc - P_bd + P_ac P_bd) (1 - P_jk - P_jl)
        #  t_ij^ab t_kl^cd
        base: e.Expr = (
            t2(indices=(i, j, a, b)) *
            t2(indices=(k, l, c, d), return_sympy=True)
        )
        v_permutations = {tuple(tuple()): 1, ((a, c),): -1, ((a, d),): -1,
                          ((b, c),): -1, ((b, d),): -1, ((a, c), (b, d)): +1}
        o_permutations = {tuple(tuple()): 1, ((j, k),): -1, ((j, l),): -1}
        t4 = 0
        for (o_perms, o_factor), (v_perms, v_factor) in \
                product(o_permutations.items(), v_permutations.items()):
            perms = o_perms + v_perms
            t4 += o_factor * v_factor * base.copy().permute(*perms).sympy
        return base_expr(t4, (i, j, k, l, a, b, c, d), None)

    def _build_tensor(self, indices) -> Amplitude:
        return Amplitude(
            f"{tensor_names.gs_amplitude}2", indices[4:], indices[:4]
        )


class t1_3(RegisteredIntermediate):
    """Third order MP single amplitude."""
    _itmd_type: str = 't_amplitude'
    _order: int = 3
    _default_idx: tuple[str] = ('i', 'a')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, a = get_symbols('ia')
        # generate additional contracted indices (2o / 2v)
        j, k, b, c = get_symbols('jkbc')
        # other intermediate class instances
        t1: t1_2 = self._registry['t_amplitude']['t1_2']
        t2: t2_2 = self._registry['t_amplitude']['t2_2']
        t3: t3_2 = self._registry['t_amplitude']['t3_2']
        if fully_expand:
            t1 = t1.expand_itmd
            t2 = t2.expand_itmd
            t3 = t3.expand_itmd
        else:
            t1 = t1.tensor
            t2 = t2.tensor
            t3 = t3.tensor
        # build the amplitude
        denom = orb_energy(i) - orb_energy(a)
        itmd = (Rational(1, 2) * eri([j, a, b, c]) *
                t2(indices=(i, j, b, c), return_sympy=True))
        itmd += (Rational(1, 2) * eri([j, k, i, b]) *
                 t2(indices=(j, k, a, b), return_sympy=True))
        itmd -= (t1(indices=(j, b), return_sympy=True) *
                 eri([i, b, j, a]))
        itmd += (Rational(1, 4) * eri([j, k, b, c]) *
                 t3(indices=(i, j, k, a, b, c), return_sympy=True))
        # need to keep track of all contracted indices... also contracted
        # indices within each of the second order t-amplitudes
        # -> substitute_contracted indices to minimize the number of contracted
        #    indices
        target = (i, a)
        if fully_expand:
            itmd = e.Expr(itmd, target_idx=target)
            itmd = itmd.substitute_contracted().sympy
            contracted = tuple(sorted(
                [s for s in itmd.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (j, k, b, c)
        return base_expr(itmd / denom, target, contracted)

    def _build_tensor(self, indices) -> Amplitude:
        return Amplitude(
            f"{tensor_names.gs_amplitude}3", (indices[1],), (indices[0],))


class t2_3(RegisteredIntermediate):
    """Third order MP double amplitude."""
    _itmd_type: str = 't_amplitude'
    _order: int = 3
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2o / 2v)
        k, l, c, d = get_symbols('klcd')
        # other intermediate class instances
        _t2_1: t2_1 = self._registry['t_amplitude']['t2_1']
        t1: t1_2 = self._registry['t_amplitude']['t1_2']
        t2: t2_2 = self._registry['t_amplitude']['t2_2']
        t3: t3_2 = self._registry['t_amplitude']['t3_2']
        t4: t4_2 = self._registry['t_amplitude']['t4_2']
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
        denom = orb_energy(a) + orb_energy(b) - orb_energy(i) - orb_energy(j)
        # +(1-P_ij) * <ic||ab> t^c_j(2)
        base = t1(indices=(j, c)) * eri((i, c, a, b))
        itmd = base.sympy - base.permute((i, j)).sympy
        # +(1-P_ab) * <ij||ka> t^b_k(2)
        base = t1(indices=(k, b)) * eri((i, j, k, a))
        itmd += base.sympy - base.permute((a, b)).sympy
        # - 0.5 * <ab||cd> t^cd_ij(2)
        itmd -= (Rational(1, 2) * eri((a, b, c, d)) *
                 t2(indices=(i, j, c, d), return_sympy=True))
        # - 0.5 * <ij||kl> t^ab_kl(2)
        itmd -= (Rational(1, 2) * eri((i, j, k, l)) *
                 t2(indices=(k, l, a, b), return_sympy=True))
        # + (1-P_ij)*(1-P_ab) * <jc||kb> t^ac_ik(2)
        base = t2(indices=(i, k, a, c)) * eri((j, c, k, b))
        itmd += (base.sympy - base.copy().permute((i, j)).sympy
                 - base.copy().permute((a, b)).sympy
                 + base.copy().permute((i, j), (a, b)).sympy)
        # + 0.5 * (1-P_ab) * <ka||cd> t^bcd_ijk(2)
        base = t3(indices=(i, j, k, b, c, d)) * eri((k, a, c, d))
        itmd += (Rational(1, 2) * base.sympy
                 - Rational(1, 2) * base.copy().permute((a, b)).sympy)
        # + 0.5 * (1-P_ij) <kl||ic> t^abc_jkl(2)
        base = t3(indices=(j, k, l, a, b, c)) * eri((k, l, i, c))
        itmd += (Rational(1, 2) * base.sympy
                 - Rational(1, 2) * base.copy().permute((i, j)).sympy)
        # + 0.25 <kl||cd> t^abcd_ijkl(2)
        itmd += (Rational(1, 4) * eri((k, l, c, d)) *
                 t4(indices=(i, j, k, l, a, b, c, d), return_sympy=True))
        # - 0.25 <kl||cd> t^ab_ij(1) t^kl_cd(1)
        itmd -= (Rational(1, 4) * eri((k, l, c, d)) *
                 _t2_1(indices=(i, j, a, b), return_sympy=True) *
                 _t2_1(indices=(k, l, c, d), return_sympy=True))
        # minimize the number of contracted indices
        target = (i, j, a, b)
        if fully_expand:
            itmd = e.Expr(itmd, target_idx=target)
            itmd = itmd.substitute_contracted().sympy
            contracted = contracted = tuple(sorted(
                [s for s in itmd.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (k, l, c, d)
        return base_expr(itmd / denom, target, contracted)

    def _build_tensor(self, indices) -> Amplitude:
        return Amplitude(
            f"{tensor_names.gs_amplitude}3", indices[2:], indices[:2]
        )


class t2_1_re_residual(RegisteredIntermediate):
    """
    Residual of the first order RE doubles amplitudes.
    """
    _itmd_type: str = 're_residual'
    _order: int = 2  # according to MP the maximum order of the residual is 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        # re intermediates can not be fully expanded, but add the bool
        # anyway for a consistent interface
        i, j, a, b = get_symbols(self.default_idx)
        # additional contracted indices
        k, l, c, d = get_symbols('klcd')
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']

        # (1 - P_ij)(1 - P_ab) <ic||ka> t_jk^bc
        base = eri([i, c, k, a]) * t2.tensor(indices=[j, k, b, c])
        itmd = (base.sympy - base.copy().permute((i, j)).sympy
                - base.copy().permute((a, b)).sympy
                + base.copy().permute((i, j), (a, b)).sympy)
        # (1 - P_ab) f_ac t_ij^bc
        base = fock([a, c]) * t2.tensor(indices=[i, j, b, c])
        itmd += base.sympy - base.copy().permute((a, b)).sympy
        # (1 - P_ij) f_jk t_ik^ab
        base = fock([j, k]) * t2.tensor(indices=[i, k, a, b])
        itmd += base.sympy - base.copy().permute((i, j)).sympy
        # - 0.5 * <ab||cd> t_ij^cd
        itmd -= (Rational(1, 2) * eri((a, b, c, d)) *
                 t2.tensor(indices=(i, j, c, d), return_sympy=True))
        # -0.5 * <ij||kl> t_kl^ab
        itmd -= (Rational(1, 2) * eri((i, j, k, l)) *
                 t2.tensor(indices=(k, l, a, b), return_sympy=True))
        # + <ij||ab>
        itmd += eri((i, j, a, b))
        target = (i, j, a, b)
        contracted = (k, l, c, d)
        return base_expr(itmd, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        # placeholder for 0, will be replaced in factor_intermediate
        return AntiSymmetricTensor("Zero", indices[:2], indices[2:])


class t1_2_re_residual(RegisteredIntermediate):
    """
    Residual of the second order RE singles amplitudes.
    """
    _itmd_type: str = 're_residual'
    _order: int = 3  # according to MP the maximum order of the residual is 3
    _default_idx: tuple[str] = ('i', 'a')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, a = get_symbols(self.default_idx)
        # additional contracted indices
        j, k, b, c = get_symbols('jkbc')

        # t amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        ts2: t1_2 = self._registry['t_amplitude']['t1_2']

        # - {V^{ib}_{ja}} {t2^{b}_{j}}
        itmd = -eri([i, b, j, a]) * ts2.tensor(indices=[j, b],
                                               return_sympy=True)
        # + {f^{a}_{b}} {t2^{b}_{i}}
        itmd += fock([a, b]) * ts2.tensor(indices=[i, b], return_sympy=True)
        # - {f^{i}_{j}} {t2^{a}_{j}}
        itmd -= fock([i, j]) * ts2.tensor(indices=[j, a], return_sympy=True)
        # + \frac{{V^{ja}_{bc}} {t1^{bc}_{ij}}}{2}
        itmd += (Rational(1, 2) * eri([j, a, b, c])
                 * t2.tensor(indices=[i, j, b, c], return_sympy=True))
        # + \frac{{V^{jk}_{ib}} {t1^{ab}_{jk}}}{2}
        itmd += (Rational(1, 2) * eri([j, k, i, b])
                 * t2.tensor(indices=[j, k, a, b], return_sympy=True))
        # - {f^{j}_{b}} {t1^{ab}_{ij}}
        itmd -= fock([j, b]) * t2.tensor(indices=[i, j, a, b],
                                         return_sympy=True)
        target = (i, a)
        contracted = (j, k, b, c)
        return base_expr(itmd, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        # placeholder for 0, will be replaced in factor_intermediate
        return AntiSymmetricTensor("Zero", (indices[0],), (indices[1],))


class t2_2_re_residual(RegisteredIntermediate):
    """
    Residual of the second order RE doubles amplitudes.
    """
    _itmd_type: str = 're_residual'
    _order: int = 3  # according to MP the maximum order of the residual is 3
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, a, b = get_symbols(self.default_idx)
        # additional contracted indices
        k, l, c, d = get_symbols('klcd')
        # t2_1 class instance
        t2: t2_2 = self._registry['t_amplitude']['t2_2']

        # (1 - P_ij)(1 - P_ab) <ic||ka> t_jk^bc
        base = eri([i, c, k, a]) * t2.tensor(indices=[j, k, b, c])
        itmd = (base.sympy - base.copy().permute((i, j)).sympy
                - base.copy().permute((a, b)).sympy
                + base.copy().permute((i, j), (a, b)).sympy)
        # (1 - P_ab) f_ac t_ij^bc
        base = fock([a, c]) * t2.tensor(indices=[i, j, b, c])
        itmd += base.sympy - base.copy().permute((a, b)).sympy
        # (1 - P_ij) f_jk t_ik^ab
        base = fock([j, k]) * t2.tensor(indices=[i, k, a, b])
        itmd += base.sympy - base.copy().permute((i, j)).sympy
        # - 0.5 * <ab||cd> t_ij^cd
        itmd -= (Rational(1, 2) * eri((a, b, c, d)) *
                 t2.tensor(indices=(i, j, c, d), return_sympy=True))
        # -0.5 * <ij||kl> t_kl^ab
        itmd -= (Rational(1, 2) * eri((i, j, k, l)) *
                 t2.tensor(indices=(k, l, a, b), return_sympy=True))
        target = (i, j, a, b)
        contracted = (k, l, c, d)
        return base_expr(itmd, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        # placeholder for 0, will be replaced in factor_intermediate
        return AntiSymmetricTensor("Zero", indices[:2], indices[2:])


class p0_2_oo(RegisteredIntermediate):
    """
    Second order contribution to the occupied occupied block of the MP
    one-particle density matrix.
    """
    _itmd_type: str = 'mp_density'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j = get_symbols(self.default_idx)
        # additional contracted indices (1o / 2v)
        k, a, b = get_symbols('kab')
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the density
        p0 = (- Rational(1, 2) *
              t2(indices=(i, k, a, b), return_sympy=True) *
              t2(indices=(j, k, a, b), return_sympy=True))
        return base_expr(p0, (i, j), (k, a, b))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}2", (indices[0],), (indices[1],), 1
        )


class p0_2_vv(RegisteredIntermediate):
    """
    Second order contribution to the virtual virtual block of the MP
    one-particle density matrix.
    """
    _itmd_type: str = 'mp_density'
    _order: int = 2
    _default_idx: tuple[str] = ('a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        a, b = get_symbols(self.default_idx)
        # additional contracted indices (2o / 1v)
        i, j, c = get_symbols('ijc')
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the density
        p0 = (Rational(1, 2) *
              t2(indices=(i, j, a, c), return_sympy=True) *
              t2(indices=(i, j, b, c), return_sympy=True))
        return base_expr(p0, (a, b), (i, j, c))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}2", (indices[0],), (indices[1],), 1)


class p0_3_oo(RegisteredIntermediate):
    """
    Third order contribution to the occupied occupied block of the MP
    one-particle density matrix.
    """
    _itmd_type: str = 'mp_density'
    _order: int = 3
    _default_idx: tuple[str] = ('i', 'j')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 2v)
        k, a, b = get_symbols('kab')
        # t amplitude cls
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        td2: t2_2 = self._registry['t_amplitude']['t2_2']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        td2 = td2.expand_itmd if fully_expand else td2.tensor
        # build the density
        p0 = (- Rational(1, 2) *
              t2(indices=(i, k, a, b), return_sympy=True) *
              td2(indices=(j, k, a, b), return_sympy=True))
        p0 += p0.subs({i: j, j: i}, simultaneous=True)

        target = (i, j)
        if fully_expand:
            p0 = e.Expr(p0, target_idx=target).substitute_contracted().sympy
            contracted = tuple(sorted(
                [s for s in p0.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (k, a, b)
        return base_expr(p0, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}3", (indices[0],), (indices[1],), 1)


class p0_3_ov(RegisteredIntermediate):
    """
    Third order contribution to the occupied virtual block of the MP
    one-particle density matrix.
    """
    _itmd_type: str = 'mp_density'
    _order: int = 3
    _default_idx: tuple[str] = ('i', 'a')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, a = get_symbols(self.default_idx)
        # generate additional contracted indices (2o / 2v)
        j, k, b, c = get_symbols('jkbc')
        # t_amplitude cls instances
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        ts2: t1_2 = self._registry['t_amplitude']['t1_2']
        tt2: t3_2 = self._registry['t_amplitude']['t3_2']
        ts3: t1_3 = self._registry['t_amplitude']['t1_3']
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
        # build the density
        # - t^ab_ij(1) t^b_j(2)
        p0 = (- t2(indices=(i, j, a, b), return_sympy=True) *
              ts2(indices=(j, b), return_sympy=True))
        # - 0.25 * t^bc_jk(1) t^abc_ijk(2)
        p0 -= (Rational(1, 4) *
               t2(indices=(j, k, b, c), return_sympy=True) *
               tt2(indices=(i, j, k, a, b, c), return_sympy=True))
        # + t^a_i(3)
        p0 += ts3(indices=(i, a), return_sympy=True)

        target = (i, a)
        if fully_expand:
            p0 = e.Expr(p0, target_idx=target).substitute_contracted().sympy
            contracted = tuple(sorted(
                [s for s in p0.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (j, k, b, c)
        return base_expr(p0, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}3", (indices[0],), (indices[1],), 1)


class p0_3_vv(RegisteredIntermediate):
    """
    Third order contribution to the virtual virtual block of the MP
    one-particle density matrix.
    """
    _itmd_type: str = 'mp_density'
    _order: int = 3
    _default_idx: tuple[str] = ('a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        a, b = get_symbols(self.default_idx)
        # additional contracted indices (2o / 1v)
        i, j, c = get_symbols('ijc')
        # t_amplitude cls instances
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        td2: t2_2 = self._registry['t_amplitude']['t2_2']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        td2 = td2.expand_itmd if fully_expand else td2.tensor
        # build the density
        p0 = (Rational(1, 2) *
              t2(indices=(i, j, a, c), return_sympy=True) *
              td2(indices=(i, j, b, c), return_sympy=True))
        p0 += p0.subs({a: b, b: a}, simultaneous=True)

        target = (a, b)
        if fully_expand:
            p0 = e.Expr(p0, target_idx=target).substitute_contracted().sympy
            contracted = tuple(sorted(
                [s for s in p0.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = (i, j, c)
        return base_expr(p0, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor(
            f"{tensor_names.gs_density}3", (indices[0],), (indices[1],), 1)


class t2eri_1(RegisteredIntermediate):
    """t2eri1 in adcc / pi1 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'a')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, k, a = get_symbols(self.default_idx)
        # generate additional contracted indices (2v)
        b, c = get_symbols('bc')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = (t2(indices=(i, j, b, c), return_sympy=True) *
                 eri((k, a, b, c)))
        return base_expr(t2eri, (i, j, k, a), (b, c))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eri1', indices[:2], indices[2:])


class t2eri_2(RegisteredIntermediate):
    """t2eri2 in adcc / pi2 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'a')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, k, a = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        b, l = get_symbols('bl')  # noqa E741
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = (t2(indices=(i, l, a, b), return_sympy=True) *
                 eri((l, k, j, b)))
        return base_expr(t2eri, (i, j, k, a), (b, l))

    def _build_tensor(self, indices) -> NonSymmetricTensor:
        return NonSymmetricTensor('t2eri2', indices)


class t2eri_3(RegisteredIntermediate):
    """t2eri3 in adcc / pi3 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2o)
        k, l = get_symbols('kl')  # noqa E741
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = (t2(indices=(k, l, a, b), return_sympy=True) *
                 eri((i, j, k, l)))
        return base_expr(t2eri, (i, j, a, b), (k, l))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eri3', indices[:2], indices[2:])


class t2eri_4(RegisteredIntermediate):
    """t2eri4 in adcc / pi4 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        k, c = get_symbols('kc')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = (t2(indices=(j, k, a, c), return_sympy=True) *
                 eri((k, b, i, c)))
        return base_expr(t2eri, (i, j, a, b), (k, c))

    def _build_tensor(self, indices) -> NonSymmetricTensor:
        return NonSymmetricTensor('t2eri4', indices)


class t2eri_5(RegisteredIntermediate):
    """t2eri5 in adcc / pi5 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2v)
        c, d = get_symbols('cd')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = (t2(indices=(i, j, c, d), return_sympy=True) *
                 eri((a, b, c, d)))
        return base_expr(t2eri, (i, j, a, b), (c, d))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eri5', indices[:2], indices[2:])


class t2eri_6(RegisteredIntermediate):
    """t2eri6 in adcc / pi6 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a', 'b', 'c')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, a, b, c = get_symbols(self.default_idx)
        # generate additional contracted indices (2o)
        j, k = get_symbols('jk')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = (t2(indices=(j, k, b, c), return_sympy=True) *
                 eri((j, k, i, a)))
        return base_expr(t2eri, (i, a, b, c), (j, k))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eri6', indices[:2], indices[2:])


class t2eri_7(RegisteredIntermediate):
    """t2eri7 in adcc / pi7 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a', 'b', 'c')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, a, b, c = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        j, d = get_symbols('jd')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        t2eri = (t2(indices=(i, j, b, d), return_sympy=True) *
                 eri((j, c, a, d)))
        return base_expr(t2eri, (i, a, b, c), (j, d))

    def _build_tensor(self, indices) -> NonSymmetricTensor:
        return NonSymmetricTensor('t2eri7', indices)


class t2eri_A(RegisteredIntermediate):
    """pia intermediate in libadc"""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'a')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, j, k, a = get_symbols(self.default_idx)
        # t2eri cls instances for generating the itmd
        pi1: t2eri_1 = self._registry['misc']['t2eri_1']
        pi2: t2eri_2 = self._registry['misc']['t2eri_2']
        pi1 = pi1.expand_itmd if fully_expand else pi1.tensor
        pi2 = pi2.expand_itmd if fully_expand else pi2.tensor
        # build the itmd
        pia = (0.5 * pi1(indices=(i, j, k, a), return_sympy=True)
               + pi2(indices=(i, j, k, a), return_sympy=True)
               - pi2(indices=(j, i, k, a), return_sympy=True))
        target = (i, j, k, a)
        if fully_expand:
            pia = e.Expr(pia, target_idx=target).substitute_contracted().sympy
            contracted = tuple(sorted(
                [s for s in pia.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = tuple()
        return base_expr(pia, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eriA', indices[:2], indices[2:])


class t2eri_B(RegisteredIntermediate):
    """pib intermediate in libadc"""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a', 'b', 'c')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, a, b, c = get_symbols(self.default_idx)
        # t2eri cls instances for generating the itmd
        pi6: t2eri_6 = self._registry['misc']['t2eri_6']
        pi7: t2eri_7 = self._registry['misc']['t2eri_7']
        pi6 = pi6.expand_itmd if fully_expand else pi6.tensor
        pi7 = pi7.expand_itmd if fully_expand else pi7.tensor
        # build the itmd
        pib = (-0.5 * pi6(indices=(i, a, b, c), return_sympy=True)
               + pi7(indices=(i, a, b, c), return_sympy=True)
               - pi7(indices=(i, a, c, b), return_sympy=True))
        target = (i, a, b, c)
        if fully_expand:
            pib = e.Expr(pib, target_idx=target).substitute_contracted().sympy
            contracted = tuple(sorted(
                [s for s in pib.atoms(Index) if s not in target],
                key=sort_idx_canonical
            ))
        else:
            contracted = tuple()
        return base_expr(pib, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eriB', indices[:2], indices[2:])


class t2sq(RegisteredIntermediate):
    """t2sq intermediate from adcc and libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a', 'j', 'b')

    @cached_member
    def _build_expanded_itmd(self, fully_expand: bool = True):
        i, a, j, b = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        c, k = get_symbols('ck')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        t2 = t2.expand_itmd if fully_expand else t2.tensor
        # build the intermediate
        itmd = (t2(indices=(i, k, a, c), return_sympy=True) *
                t2(indices=(j, k, b, c), return_sympy=True))
        return base_expr(itmd, (i, a, j, b), (k, c))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2sq', indices[:2], indices[2:], 1)


def eri(idx: str | list[Index] | list[str]) -> AntiSymmetricTensor:
    """
    Builds an antisymmetric electron repulsion integral.
    Indices may be provided as list of sympy symbols or as string.
    """
    idx = get_symbols(idx)
    if len(idx) != 4:
        raise Inputerror(f'4 indices required to build a ERI. Got: {idx}.')
    return AntiSymmetricTensor(tensor_names.eri, idx[:2], idx[2:])


def fock(idx: str | list[Index] | list[str]) -> AntiSymmetricTensor:
    """
    Builds a fock matrix element.
    Indices may be provided as list of sympy symbols or as string.
    """
    idx = get_symbols(idx)
    if len(idx) != 2:
        raise Inputerror('2 indices required to build a Fock matrix element.'
                         f'Got: {idx}.')
    return AntiSymmetricTensor(tensor_names.fock, idx[:1], idx[1:])


def orb_energy(idx: str | Index) -> NonSymmetricTensor:
    """
    Builds an orbital energy.
    Indices may be provided as list of sympy symbols or as string.
    """
    idx = get_symbols(idx)
    if len(idx) != 1:
        raise Inputerror("1 index required to build a orbital energy. Got: "
                         f"{idx}.")
    return NonSymmetricTensor(tensor_names.orb_energy, idx)
