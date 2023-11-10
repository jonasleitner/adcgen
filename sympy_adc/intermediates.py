from .indices import (
    get_symbols, index_space, order_substitutions, idx_sort_key
)
from .indices import Indices as idx_cls
from .misc import Inputerror, Singleton, cached_property, cached_member
from . import expr_container as e
from .eri_orbenergy import EriOrbenergy
from .sympy_objects import NonSymmetricTensor, AntiSymmetricTensor
from .symmetry import LazyTermMap

from sympy import S, Dummy, Rational, Pow

from collections import namedtuple


base_expr = namedtuple('base_expr', ['expr', 'target', 'contracted'])


class Intermediates(metaclass=Singleton):
    def __init__(self):
        self._registered: dict = RegisteredIntermediate()._registry
        self._available: dict = {
            name: obj for objects in self._registered.values()
            for name, obj in objects.items()
        }

    @property
    def available(self) -> dict:
        return self._available

    @property
    def types(self) -> list[str]:
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
    _registry: dict[str, dict[str]] = {}

    def __init_subclass__(cls):
        if (itmd_type := cls._itmd_type) not in cls._registry:
            cls._registry[itmd_type] = {}
        if (name := cls.__name__) not in cls._registry[itmd_type]:
            cls._registry[itmd_type][name] = cls()

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def order(self) -> int:
        if (order := getattr(self, '_order', None)) is None:
            raise AttributeError(f"No order defined for {self.name}.")
        return order

    @property
    def default_idx(self) -> tuple[str]:
        if (idx := getattr(self, '_default_idx', None)) is None:
            raise AttributeError(f"No default indices defined for {self.name}")
        return idx

    @property
    def itmd_type(self) -> str:
        if (itmd_type := getattr(self, '_itmd_type', None)) is None:
            raise AttributeError(f"No itmd_type defined for {self.name}.")
        return itmd_type

    def validate_indices(self, **kwargs) -> tuple[Dummy]:
        indices = kwargs.pop('indices', None)
        default = self.default_idx
        # indices are provided as index string
        if indices is not None:
            if any(val is not None for val in kwargs.values()):
                raise Inputerror("If indices are provided via the indices "
                                 "keyword, no further indices can be provided")
        else:  # either no indices provided or via upper + lower
            lower = kwargs.get('lower')
            upper = kwargs.get('upper')
            if lower is None and upper is None:
                # no indices provided -> use default
                indices = default
            elif lower is None or upper is None:
                raise Inputerror(f"Invalid indices {kwargs} provided for "
                                 f"intermediate {self.name}.")
            else:  # all spaces provided -> use the provided indices
                if self.itmd_type == 't_amplitude':  # order = lower, upper
                    indices = lower + upper
                else:  # order = upper, lower
                    indices = upper + lower
        indices = tuple(get_symbols(indices))
        # check that we have the correct amount of indices and the correct
        # amount of occupied and virtual indices
        if len(indices) != len(default) or \
                any(index_space(idx.name) != index_space(ref) for idx, ref in
                    zip(indices, default)):
            raise Inputerror(f"Invalid indices {indices} provided for "
                             f"intermediate {self.name}.")
        return indices

    def expand_itmd(self, indices: str = None, lower: str = None,
                    upper: str = None, return_sympy: bool = False):
        # check that the provided indices are fine for the itmd
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)

        # build a cached base version of the intermediate where we can just
        # substitute indices
        expanded_itmd = self._build_expanded_itmd

        # build the substitution dict
        subs = {}
        # map target indices onto each other
        if (base_target := expanded_itmd.target) is not None:
            subs.update({o: n for o, n in zip(base_target, idx)})
        # map contracted indices onto each other
        if (base_contracted := expanded_itmd.contracted) is not None:
            spaces = [index_space(s.name) for s in base_contracted]
            contracted = idx_cls().get_generic_indices(
                n_o=spaces.count('occ'), n_v=spaces.count('virt'),
                n_g=spaces.count('general')
            )
            for old, sp in zip(base_contracted, spaces):
                subs[old] = contracted[sp].pop(0)
            if any(li for li in contracted.values()):
                raise RuntimeError("Generated more contracted indices than "
                                   f"necessary. {contracted} are left.")

        # do some extra work with the substitutions to avoid using the
        # simultantous=True option for subs (very slow)
        subs = order_substitutions(subs)
        itmd = expanded_itmd.expr.subs(subs)

        if itmd is S.Zero and expanded_itmd.expr is not S.Zero:
            raise ValueError(f"The substitutions {subs} ar enot valid for "
                             f"{expanded_itmd.expr}.")

        if not return_sympy:
            itmd = e.Expr(itmd, target_idx=idx)
        return itmd

    def tensor(self, indices: str = None, upper: str = None, lower: str = None,
               return_sympy: bool = False):
        # check that the provided indices are sufficient for the itmd
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)

        # build the tensor object
        tensor = self._build_tensor(indices=idx)
        if return_sympy:
            return tensor
        else:
            if isinstance(tensor, AntiSymmetricTensor):
                if tensor.bra_ket_sym is S.One:  # bra ket symmetry
                    return e.Expr(tensor, sym_tensors=[tensor.symbol.name])
                elif tensor.bra_ket_sym is S.NegativeOne:  # bra ket anisym
                    return e.Expr(tensor, antisym_tensors=[tensor.symbol.name])
            return e.Expr(tensor)

    @cached_property
    def tensor_symmetry(self) -> dict:
        """Determines the symmetry of the itmd tensor object using the
           default indices, e.g., ijk + abc triples symmetry for t3_2."""
        return self.tensor().terms[0].symmetry()

    @cached_member
    def itmd_term_map(self,
                      factored_itmds: tuple[str] = tuple()) -> LazyTermMap:
        """Returns a map that lazily determines which permutation of target
           indices map terms in the intermediate definition onto other terms.
           Since the form (and order of terms) depends on the previsously
           factored intermediates, a term map for each variant has to be
           created.
           """
        # - load the appropriate version of the intermediate
        itmd = self._prepare_itmd(factored_itmds)
        return LazyTermMap(itmd)

    @cached_member
    def _prepare_itmd(self, factored_itmds: list[str] = tuple()) -> e.Expr:
        """"Generate the default variant of the intermediate, simplify it
            as much as possible and factor all given intermediates in the
            provided order."""
        from .reduce_expr import factor_eri_parts, factor_denom
        from itertools import chain

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

        print('\n', '-'*80, sep='')
        print(f"Preparing Intermediate: Factoring {factored_itmds}")

        if factored_itmds:
            available = Intermediates().available
            # iterate through factored_itmds and factor them one after another
            # in the simplified base itmd
            for i, it in enumerate(factored_itmds):
                print('-'*80)
                print(f"Factoring {it} in {self.name}:")
                itmd = available[it].factor_itmd(
                    itmd, factored_itmds=factored_itmds[:i],
                    max_order=self.order
                )
        print('\n', '-'*80, sep='')
        print(f"Done with factoring {factored_itmds} in {self.name}")
        print('-'*80)
        return itmd

    def factor_itmd(self, expr: e.Expr, factored_itmds: tuple[str] = tuple(),
                    max_order: int = None) -> e.Expr:
        """Factors the intermediate in a given expression assuming a
           real orbital basis."""

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

    @cached_property
    def _build_expanded_itmd(self):
        # build a basic version of the intermediate using minimal indices
        # 'like on paper'
        i, j, a, b = get_symbols(self.default_idx)
        denom = orb_energy(a) + orb_energy(b) - orb_energy(i) - orb_energy(j)
        return base_expr(eri((a, b, i, j)) / denom, (i, j, a, b), None)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        # guess its not worth caching here. Maybe if used a lot.
        # build the tensor
        return AntiSymmetricTensor('t1', indices[2:], indices[:2])

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
                        bk_exponent = bk.exponent
                        bk = bk.extract_pow
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
            print(f"\nFactoring {self.name} in:\n{term}\nresult:\n"
                  f"{EriOrbenergy(factored_term)}")
            factored += factored_term
        return factored


class t1_2(RegisteredIntermediate):
    """Second order MP singles amplitude."""
    _itmd_type: str = "t_amplitude"
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a')
    _min_n_terms = 2

    @cached_property
    def _build_expanded_itmd(self):
        # target_indices
        i, a = get_symbols(self.default_idx)
        # additional contracted indices
        j, k, b, c = get_symbols('jkbc')
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the amplitude
        denom = orb_energy(i) - orb_energy(a)
        term1 = (Rational(1, 2) *
                 t2.expand_itmd(indices=(i, j, b, c), return_sympy=True) *
                 eri([j, a, b, c]))
        term2 = (Rational(1, 2) *
                 t2.expand_itmd(indices=(j, k, a, b), return_sympy=True) *
                 eri([j, k, i, b]))
        return base_expr(term1/denom + term2/denom, (i, a), (j, k, b, c))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2', (indices[1],), (indices[0],))


class t2_2(RegisteredIntermediate):
    """Second order MP doubles amplitude."""
    _itmd_type: str = 't_amplitude'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2o / 2v)
        k, l, c, d = get_symbols('klcd')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the t2_2 amplitude
        denom = (orb_energy(a) + orb_energy(b) - orb_energy(i) - orb_energy(j))
        # - 0.5 t2eri_3
        itmd = (- Rational(1, 2) * eri((i, j, k, l)) *
                t2.expand_itmd(indices=(k, l, a, b), return_sympy=True))
        # - 0.5 t2eri_5
        itmd += (- Rational(1, 2) * eri((a, b, c, d)) *
                 t2.expand_itmd(indices=(i, j, c, d), return_sympy=True))
        # + (1 - P_ij) (1 - P_ab) P_ij t2eri_4
        base = (
            t2.expand_itmd(indices=(i, k, a, c)) * eri((k, b, j, c))
        )
        itmd += (base.sympy - base.copy().permute((i, j)).sympy
                 - base.copy().permute((a, b)).sympy
                 + base.copy().permute((i, j), (a, b)).sympy)
        return base_expr(itmd / denom, (i, j, a, b), (k, l, c, d))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2', indices[2:], indices[:2])


class t3_2(RegisteredIntermediate):
    """Second order MP triples amplitude."""
    _itmd_type: str = 't_amplitude'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'a', 'b', 'c')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, k, a, b, c = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        l, d = get_symbols('ld')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the t3_2 amplitude
        denom = (orb_energy(i) + orb_energy(j) + orb_energy(k)
                 - orb_energy(a) - orb_energy(b) - orb_energy(c))
        # (1 - P_ik - P_jk) (1 - P_ab - P_ac) <kd||bc> t_ij^ad
        base = t2.expand_itmd(indices=(i, j, a, d)) * eri((k, d, b, c))
        itmd = (base.sympy - base.copy().permute((i, k)).sympy
                - base.copy().permute((j, k)).sympy
                - base.copy().permute((a, b)).sympy
                - base.copy().permute((a, c)).sympy
                + base.copy().permute((i, k), (a, b)).sympy
                + base.copy().permute((i, k), (a, c)).sympy
                + base.copy().permute((j, k), (a, b)).sympy
                + base.copy().permute((j, k), (a, c)).sympy)
        # (1 - P_ij - P_ik) (1 - P_ac - P_bc) <jk||lc> t_il^ab
        base = t2.expand_itmd(indices=(i, l, a, b)) * eri((j, k, l, c))
        itmd += (base.sympy - base.copy().permute((i, j)).sympy
                 - base.copy().permute((i, k)).sympy
                 - base.copy().permute((a, c)).sympy
                 - base.copy().permute((b, c)).sympy
                 + base.copy().permute((i, j), (a, c)).sympy
                 + base.copy().permute((i, j), (b, c)).sympy
                 + base.copy().permute((i, k), (a, c)).sympy
                 + base.copy().permute((i, k), (b, c)).sympy)
        return base_expr(itmd/denom, (i, j, k, a, b, c), (l, d))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2', indices[3:], indices[:3])


class t4_2(RegisteredIntermediate):
    """Second order MP quadruple amplitudes in a factorized form that avoids
       the construction of the quadruples denominator."""
    _itmd_type: str = 't_amplitude'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'l', 'a', 'b', 'c', 'd')

    @cached_property
    def _build_expanded_itmd(self):
        from itertools import product

        i, j, k, l, a, b, c, d = get_symbols(self.default_idx)
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the t4_2 amplitude
        # (1 - P_ac - P_ad - P_bc - P_bd + P_ac P_bd) (1 - P_jk - P_jl)
        #  t_ij^ab t_kl^cd
        base: e.Expr = (
            t2.expand_itmd(indices=(i, j, a, b)) *
            t2.expand_itmd(indices=(k, l, c, d), return_sympy=True)
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

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2', indices[4:], indices[:4])


class t1_3(RegisteredIntermediate):
    """Third order MP single amplitudes."""
    _itmd_type: str = 't_amplitude'
    _order: int = 3
    _default_idx: tuple[str] = ('i', 'a')

    @cached_property
    def _build_expanded_itmd(self):
        i, a = get_symbols('ia')
        # generate additional contracted indices (2o / 2v)
        j, k, b, c = get_symbols('jkbc')
        # other intermediate class instances
        t1: t1_2 = self._registry['t_amplitude']['t1_2']
        t2: t2_2 = self._registry['t_amplitude']['t2_2']
        t3: t3_2 = self._registry['t_amplitude']['t3_2']
        # build the amplitude
        denom = orb_energy(i) - orb_energy(a)
        itmd = (Rational(1, 2) * eri([j, a, b, c]) *
                t2.expand_itmd(indices=(i, j, b, c), return_sympy=True))
        itmd += (Rational(1, 2) * eri([j, k, i, b]) *
                 t2.expand_itmd(indices=(j, k, a, b), return_sympy=True))
        itmd -= (t1.expand_itmd(indices=(j, b), return_sympy=True) *
                 eri([i, b, j, a]))
        itmd += (Rational(1, 4) * eri([j, k, b, c]) *
                 t3.expand_itmd(indices=(i, j, k, a, b, c), return_sympy=True))
        # need to keep track of all contracted indices... also contracted
        # indices within each of the second order t-amplitudes
        # -> substitute_contracted indices to minimize the number of contracted
        #    indices
        target = (i, a)
        itmd = e.Expr(itmd, target_idx=target).substitute_contracted().sympy
        contracted = tuple(sorted(
            [s for s in itmd.atoms(Dummy) if s not in target], key=idx_sort_key
        ))
        return base_expr(itmd / denom, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t3', (indices[1],), (indices[0],))


class t2_3(RegisteredIntermediate):
    """Third order MP double amplitudes."""
    _itmd_type: str = 't_amplitude'
    _order: int = 3
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2o / 2v)
        k, l, c, d = get_symbols('klcd')
        # other intermediate class instances
        _t2_1: t2_1 = self._registry['t_amplitude']['t2_1']
        t1: t1_2 = self._registry['t_amplitude']['t1_2']
        t2: t2_2 = self._registry['t_amplitude']['t2_2']
        t3: t3_2 = self._registry['t_amplitude']['t3_2']
        t4: t4_2 = self._registry['t_amplitude']['t4_2']
        # build the amplitude
        denom = orb_energy(a) + orb_energy(b) - orb_energy(i) - orb_energy(j)
        # +(1-P_ij) * <ic||ab> t^c_j(2)
        base = t1.expand_itmd(indices=(j, c)) * eri((i, c, a, b))
        itmd = base.sympy - base.permute((i, j)).sympy
        # +(1-P_ab) * <ij||ka> t^b_k(2)
        base = t1.expand_itmd(indices=(k, b)) * eri((i, j, k, a))
        itmd += base.sympy - base.permute((a, b)).sympy
        # - 0.5 * <ab||cd> t^cd_ij(2)
        itmd -= (Rational(1, 2) * eri((a, b, c, d)) *
                 t2.expand_itmd(indices=(i, j, c, d), return_sympy=True))
        # - 0.5 * <ij||kl> t^ab_kl(2)
        itmd -= (Rational(1, 2) * eri((i, j, k, l)) *
                 t2.expand_itmd(indices=(k, l, a, b), return_sympy=True))
        # + (1-P_ij)*(1-P_ab) * <jc||kb> t^ac_ik(2)
        base = t2.expand_itmd(indices=(i, k, a, c)) * eri((j, c, k, b))
        itmd += (base.sympy - base.copy().permute((i, j)).sympy
                 - base.copy().permute((a, b)).sympy
                 + base.copy().permute((i, j), (a, b)).sympy)
        # + 0.5 * (1-P_ab) * <ka||cd> t^bcd_ijk(2)
        base = t3.expand_itmd(indices=(i, j, k, b, c, d)) * eri((k, a, c, d))
        itmd += (Rational(1, 2) * base.sympy
                 - Rational(1, 2) * base.copy().permute((a, b)).sympy)
        # + 0.5 * (1-P_ij) <kl||ic> t^abc_jkl(2)
        base = t3.expand_itmd(indices=(j, k, l, a, b, c)) * eri((k, l, i, c))
        itmd += (Rational(1, 2) * base.sympy
                 - Rational(1, 2) * base.copy().permute((i, j)).sympy)
        # + 0.25 <kl||cd> t^abcd_ijkl(2)
        itmd += (Rational(1, 4) * eri((k, l, c, d)) *
                 t4.expand_itmd(indices=(i, j, k, l, a, b, c, d),
                                return_sympy=True))
        # - 0.25 <kl||cd> t^ab_ij(1) t^kl_cd(1)
        itmd -= (Rational(1, 4) * eri((k, l, c, d)) *
                 _t2_1.expand_itmd(indices=(i, j, a, b), return_sympy=True) *
                 _t2_1.expand_itmd(indices=(k, l, c, d), return_sympy=True))
        # minimize the number of contracted indices
        target = (i, j, a, b)
        itmd = e.Expr(itmd, target_idx=target).substitute_contracted().sympy
        contracted = contracted = tuple(sorted(
            [s for s in itmd.atoms(Dummy) if s not in target], key=idx_sort_key
        ))
        return base_expr(itmd / denom, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t3', indices[2:], indices[:2])


class t2_1_re_residual(RegisteredIntermediate):
    """Residual of the first order REPT doubles amplitudes.
    """
    _itmd_type: str = 're_residual'
    _order: int = 2  # according to MP the maximum order of the residual is 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, a, b = get_symbols(self.default_idx)
        # additional contracted indices
        k, l, c, d = get_symbols('klcd')
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']

        # (1 - P_ij)(1 - P_ab) <ic||ka> t_jk^bc
        base = eri([i, c, k, a]) * t2.tensor(indices=[j, k, b, c])
        itmd = (base.sympy - base.copy().permute((i, j)).sympy
                - base.copy().permute((a, b)).sympy
                + base.copy().permute((i, j), (a, b)))
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


class p0_2_oo(RegisteredIntermediate):
    """Occupied Occupied block of the 2nd order contribution of the MP density
    """
    _itmd_type: str = 'mp_density'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j')

    @cached_property
    def _build_expanded_itmd(self):
        i, j = get_symbols(self.default_idx)
        # additional contracted indices (1o / 2v)
        k, a, b = get_symbols('kab')
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the density
        p0 = (- Rational(1, 2) *
              t2.expand_itmd(indices=(i, k, a, b), return_sympy=True) *
              t2.expand_itmd(indices=(j, k, a, b), return_sympy=True))
        return base_expr(p0, (i, j), (k, a, b))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('p2', (indices[0],), (indices[1],), 1)


class p0_2_vv(RegisteredIntermediate):
    """Virtual Virtual block of the 2nd order contribution of the MP density"""
    _itmd_type: str = 'mp_density'
    _order: int = 2
    _default_idx: tuple[str] = ('a', 'b')

    @cached_property
    def _build_expanded_itmd(self):
        a, b = get_symbols(self.default_idx)
        # additional contracted indices (2o / 1v)
        i, j, c = get_symbols('ijc')
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the density
        p0 = (Rational(1, 2) *
              t2.expand_itmd(indices=(i, j, a, c), return_sympy=True) *
              t2.expand_itmd(indices=(i, j, b, c), return_sympy=True))
        return base_expr(p0, (a, b), (i, j, c))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('p2', (indices[0],), (indices[1],), 1)


class p0_3_oo(RegisteredIntermediate):
    """Occupied Occupied block of the 2nd order contribution of the MP density
    """
    _itmd_type: str = 'mp_density'
    _order: int = 3
    _default_idx: tuple[str] = ('i', 'j')

    @cached_property
    def _build_expanded_itmd(self):
        i, j = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 2v)
        k, a, b = get_symbols('kab')
        # t amplitude cls
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        td2: t2_2 = self._registry['t_amplitude']['t2_2']
        # build the density
        p0 = (- Rational(1, 2) *
              t2.expand_itmd(indices=(i, k, a, b), return_sympy=True) *
              td2.expand_itmd(indices=(j, k, a, b), return_sympy=True))
        p0 += p0.subs({i: j, j: i}, simultaneous=True)

        target = (i, j)
        p0 = e.Expr(p0, target_idx=target).substitute_contracted().sympy
        contracted = tuple(sorted(
            [s for s in p0.atoms(Dummy) if s not in target], key=idx_sort_key
        ))
        return base_expr(p0, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('p3', (indices[0],), (indices[1],), 1)


class p0_3_ov(RegisteredIntermediate):
    """Occupied Occupied block of the 2nd order contribution of the MP density
    """
    _itmd_type: str = 'mp_density'
    _order: int = 3
    _default_idx: tuple[str] = ('i', 'a')

    @cached_property
    def _build_expanded_itmd(self):
        i, a = get_symbols(self.default_idx)
        # generate additional contracted indices (2o / 2v)
        j, k, b, c = get_symbols('jkbc')
        # t_amplitude cls instances
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        ts2: t1_2 = self._registry['t_amplitude']['t1_2']
        tt2: t3_2 = self._registry['t_amplitude']['t3_2']
        ts3: t1_3 = self._registry['t_amplitude']['t1_3']
        # build the density
        # - t^ab_ij(1) t^b_j(2)
        p0 = (- t2.expand_itmd(indices=(i, j, a, b), return_sympy=True) *
              ts2.expand_itmd(indices=(j, b), return_sympy=True))
        # - 0.25 * t^bc_jk(1) t^abc_ijk(2)
        p0 -= (Rational(1, 4) *
               t2.expand_itmd(indices=(j, k, b, c), return_sympy=True) *
               tt2.expand_itmd(indices=(i, j, k, a, b, c), return_sympy=True))
        # + t^a_i(3)
        p0 += ts3.expand_itmd(indices=(i, a), return_sympy=True)

        target = (i, a)
        p0 = e.Expr(p0, target_idx=target).substitute_contracted().sympy
        contracted = tuple(sorted(
            [s for s in p0.atoms(Dummy) if s not in target], key=idx_sort_key
        ))
        return base_expr(p0, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('p3', (indices[0],), (indices[1],), 1)


class p0_3_vv(RegisteredIntermediate):
    """Virtual Virtual block of the 2nd order contribution of the MP density"""
    _itmd_type: str = 'mp_density'
    _order: int = 3
    _default_idx: tuple[str] = ('a', 'b')

    @cached_property
    def _build_expanded_itmd(self):
        a, b = get_symbols(self.default_idx)
        # additional contracted indices (2o / 1v)
        i, j, c = get_symbols('ijc')
        # t_amplitude cls instances
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        td2: t2_2 = self._registry['t_amplitude']['t2_2']
        # build the density
        p0 = (Rational(1, 2) *
              t2.expand_itmd(indices=(i, j, a, c), return_sympy=True) *
              td2.expand_itmd(indices=(i, j, b, c), return_sympy=True))
        p0 += p0.subs({a: b, b: a}, simultaneous=True)

        target = (a, b)
        p0 = e.Expr(p0, target_idx=target).substitute_contracted().sympy
        contracted = tuple(sorted(
            [s for s in p0.atoms(Dummy) if s not in target], key=idx_sort_key
        ))
        return base_expr(p0, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('p3', (indices[0],), (indices[1],), 1)


class t2eri_1(RegisteredIntermediate):
    """t2eri1 in adcc / pi1 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'a')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, k, a = get_symbols(self.default_idx)
        # generate additional contracted indices (2v)
        b, c = get_symbols('bc')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = (t2.expand_itmd(indices=(i, j, b, c), return_sympy=True) *
                 eri((k, a, b, c)))
        return base_expr(t2eri, (i, j, k, a), (b, c))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eri1', indices[:2], indices[2:])


class t2eri_2(RegisteredIntermediate):
    """t2eri2 in adcc / pi2 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'a')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, k, a = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        b, l = get_symbols('bl')  # noqa E741
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = (t2.expand_itmd(indices=(i, l, a, b), return_sympy=True) *
                 eri((l, k, j, b)))
        return base_expr(t2eri, (i, j, k, a), (b, l))

    def _build_tensor(self, indices) -> NonSymmetricTensor:
        return NonSymmetricTensor('t2eri2', indices)


class t2eri_3(RegisteredIntermediate):
    """t2eri3 in adcc / pi3 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2o)
        k, l = get_symbols('kl')  # noqa E741
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = (t2.expand_itmd(indices=(k, l, a, b), return_sympy=True) *
                 eri((i, j, k, l)))
        return base_expr(t2eri, (i, j, a, b), (k, l))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eri3', indices[:2], indices[2:])


class t2eri_4(RegisteredIntermediate):
    """t2eri4 in adcc / pi4 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        k, c = get_symbols('kc')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = (t2.expand_itmd(indices=(j, k, a, c), return_sympy=True) *
                 eri((k, b, i, c)))
        return base_expr(t2eri, (i, j, a, b), (k, c))

    def _build_tensor(self, indices) -> NonSymmetricTensor:
        return NonSymmetricTensor('t2eri4', indices)


class t2eri_5(RegisteredIntermediate):
    """t2eri5 in adcc / pi5 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'a', 'b')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, a, b = get_symbols(self.default_idx)
        # generate additional contracted indices (2v)
        c, d = get_symbols('cd')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = (t2.expand_itmd(indices=(i, j, c, d), return_sympy=True) *
                 eri((a, b, c, d)))
        return base_expr(t2eri, (i, j, a, b), (c, d))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eri5', indices[:2], indices[2:])


class t2eri_6(RegisteredIntermediate):
    """t2eri6 in adcc / pi6 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a', 'b', 'c')

    @cached_property
    def _build_expanded_itmd(self):
        i, a, b, c = get_symbols(self.default_idx)
        # generate additional contracted indices (2o)
        j, k = get_symbols('jk')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = (t2.expand_itmd(indices=(j, k, b, c), return_sympy=True) *
                 eri((j, k, i, a)))
        return base_expr(t2eri, (i, a, b, c), (j, k))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eri6', indices[:2], indices[2:])


class t2eri_7(RegisteredIntermediate):
    """t2eri7 in adcc / pi7 in libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a', 'b', 'c')

    @cached_property
    def _build_expanded_itmd(self):
        i, a, b, c = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        j, d = get_symbols('jd')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = (t2.expand_itmd(indices=(i, j, b, d), return_sympy=True) *
                 eri((j, c, a, d)))
        return base_expr(t2eri, (i, a, b, c), (j, d))

    def _build_tensor(self, indices) -> NonSymmetricTensor:
        return NonSymmetricTensor('t2eri7', indices)


class t2eri_A(RegisteredIntermediate):
    """pia intermediate in libadc"""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'j', 'k', 'a')

    @cached_property
    def _build_expanded_itmd(self):
        i, j, k, a = get_symbols(self.default_idx)
        # t2eri cls instances for generating the itmd
        pi1: t2eri_1 = self._registry['misc']['t2eri_1']
        pi2: t2eri_2 = self._registry['misc']['t2eri_2']
        # build the itmd
        pia = (0.5 * pi1.expand_itmd(indices=(i, j, k, a), return_sympy=True)
               + pi2.expand_itmd(indices=(i, j, k, a), return_sympy=True)
               - pi2.expand_itmd(indices=(j, i, k, a), return_sympy=True))
        target = (i, j, k, a)
        pia = e.Expr(pia, target_idx=target).substitute_contracted().sympy
        contracted = tuple(sorted(
            [s for s in pia.atoms(Dummy) if s not in target], key=idx_sort_key
        ))
        return base_expr(pia, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eriA', indices[:2], indices[2:])


class t2eri_B(RegisteredIntermediate):
    """pib intermediate in libadc"""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a', 'b', 'c')

    @cached_property
    def _build_expanded_itmd(self):
        i, a, b, c = get_symbols(self.default_idx)
        # t2eri cls instances for generating the itmd
        pi6: t2eri_6 = self._registry['misc']['t2eri_6']
        pi7: t2eri_7 = self._registry['misc']['t2eri_7']
        # build the itmd
        pib = (-0.5 * pi6.expand_itmd(indices=(i, a, b, c), return_sympy=True)
               + pi7.expand_itmd(indices=(i, a, b, c), return_sympy=True)
               - pi7.expand_itmd(indices=(i, a, c, b), return_sympy=True))
        target = (i, a, b, c)
        pib = e.Expr(pib, target_idx=target).substitute_contracted().sympy
        contracted = tuple(sorted(
            [s for s in pib.atoms(Dummy) if s not in target], key=idx_sort_key
        ))
        return base_expr(pib, target, contracted)

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2eriB', indices[:2], indices[2:])


class t2sq(RegisteredIntermediate):
    """t2sq intermediate from adcc and libadc."""
    _itmd_type: str = 'misc'
    _order: int = 2
    _default_idx: tuple[str] = ('i', 'a', 'j', 'b')

    @cached_property
    def _build_expanded_itmd(self):
        i, a, j, b = get_symbols(self.default_idx)
        # generate additional contracted indices (1o / 1v)
        c, k = get_symbols('ck')
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        itmd = (t2.expand_itmd(indices=(i, k, a, c), return_sympy=True) *
                t2.expand_itmd(indices=(j, k, b, c), return_sympy=True))
        return base_expr(itmd, (i, a, j, b), (k, c))

    def _build_tensor(self, indices) -> AntiSymmetricTensor:
        return AntiSymmetricTensor('t2sq', indices[:2], indices[2:], 1)


def eri(idx: str | list[Dummy] | list[str]) -> AntiSymmetricTensor:
    """Builds an electron repulsion integral using the provided indices.
       Indices may be provided as list of sympy symbols or as string."""

    idx = get_symbols(idx)
    if len(idx) != 4:
        raise Inputerror(f'4 indices required to build a ERI. Got: {idx}.')
    return AntiSymmetricTensor('V', idx[:2], idx[2:])


def fock(idx: str | list[Dummy] | list[str]) -> AntiSymmetricTensor:
    """Builds an electron repulsion integral using the provided indices.
       Indices may be provided as list of sympy symbols or as string."""

    idx = get_symbols(idx)
    if len(idx) != 2:
        raise Inputerror('2 indices required to build a Fock matrix element.'
                         f'Got: {idx}.')
    return AntiSymmetricTensor('f', idx[:1], idx[1:])


def orb_energy(idx: str | Dummy) -> NonSymmetricTensor:
    """Builds an orbital energy using the provided index.
       Indices may be provided as list of sympy symbols or as string."""

    idx = get_symbols(idx)
    if len(idx) != 1:
        raise Inputerror("1 index required to build a orbital energy. Got: "
                         f"{idx}.")
    return NonSymmetricTensor('e', idx)
