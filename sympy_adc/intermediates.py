from .indices import get_symbols, index_space
from .misc import Inputerror, Singleton
from . import expr_container as e
from .eri_orbenergy import eri_orbenergy
from .sympy_objects import NonSymmetricTensor, AntiSymmetricTensor

from sympy import S, Dummy

from collections import defaultdict


class intermediates(metaclass=Singleton):
    def __init__(self):
        self.__registered: dict = registered_intermediate()._registry
        self.__available: dict = {
            name: obj for objects in self.__registered.values()
            for name, obj in objects.items()
        }

    @property
    def available(self) -> dict:
        return self.__available

    @property
    def types(self) -> list[str]:
        return list(self.__registered.keys())

    def __getattr__(self, attr):
        try:
            return self.__registered[attr]
        except KeyError:
            raise AttributeError(f"{self} has no attribute {attr}. "
                                 f"Available intermediate types: {self.types}")


class registered_intermediate:
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

    def expand_itmd(self, *args, **kwargs):
        raise NotImplementedError("Expand itmd not implemented for "
                                  f"intermediate {self.name}.")

    def tensor(self, *args, **kwargs):
        raise NotImplementedError("Method for symbolic representation of "
                                  f"{self.name} not implemented.")

    def tensor_symmetry(self) -> dict:
        """Determines the symmetry of the itmd tensor objects, e.g.,
           triples symmetry for t3_2."""

        # try to load from cache
        if hasattr(self, '_tensor_symmetry'):
            return self._tensor_symmetry
        # determine the symmetry
        tensor = self.tensor().make_real()
        tensor_sym = tensor.terms[0].symmetry()
        if not hasattr(self, '_tensor_symmetry'):
            self._tensor_symmetry = {}
        self._tensor_symmetry = tensor_sym
        return tensor_sym

    def itmd_term_map(self, factored_itmds: list[str] = None) \
            -> dict[tuple, dict]:
        """Determines which permutations of the itmd target indices transform
           a certain itmd_term into another one, e.g.,
           ... + (1 - P_ij) * (1 - P_ab) * x for the t2_2 amplitudes.
           The target indices of the provided itmd expr need to be set
           to the target indices of the itmd."""
        from sympy import Mul

        if factored_itmds is None:
            factored_itmds = tuple()
        else:
            factored_itmds = tuple(factored_itmds)

        # try to load the term map from the cache
        cache: dict[tuple, dict] = self._term_map_cache
        if factored_itmds in cache:
            return cache[factored_itmds]

        # compute the term map
        itmd = self._prepare_itmd(factored_itmds=factored_itmds)
        itmd_symmetry = self.tensor_symmetry()

        itmd_terms = itmd.terms
        term_map = defaultdict(lambda: defaultdict(list))
        for i, term in enumerate(itmd_terms):
            # only apply permutations to a term that are not already
            # inherent to the term!
            term_sym = term.symmetry(only_target=True)
            to_check = dict(itmd_symmetry.items() - term_sym.items())
            for perms, factor in to_check.items():
                perm_term = term.permute(*perms).sympy
                for other_i, other_term in enumerate(itmd_terms):
                    if i == other_i:
                        continue
                    if (sum := perm_term + other_term.sympy) is S.Zero:
                        # P_pq X + (- P_pq X) = 0
                        factor = -1
                    elif isinstance(sum, Mul):  # only a single term?
                        # P_pq X + (+ P_pq X) = 2 P_pq X
                        factor = +1
                    else:
                        continue
                    term_map[perms][i].append((other_i, factor))
        term_map = dict(term_map)
        for key, default_d in term_map.items():
            term_map[key] = dict(default_d)
        cache[factored_itmds] = term_map
        return term_map

    def _prepare_itmd(self, factored_itmds: list[str] = None) -> e.expr:
        """"Factor all previously factorized in intermediates in the current
            intermediate."""
        from .reduce_expr import factor_eri, factor_denom
        from itertools import chain

        if factored_itmds is None:
            factored_itmds = tuple()
        else:
            factored_itmds = tuple(factored_itmds)

        # did we already factore the itmd previously?
        factored_variants = self._factored_variants
        if (itmd := factored_variants.get(factored_itmds)) is not None:
            return itmd

        if factored_variants:  # load the reduces base intermediate
            itmd = factored_variants[tuple()]
        else:  # construct the expanded intermediate
            itmd: e.expr = self.expand_itmd().expand().make_real()
            # reduce the expanded intermediate
            reduced = chain.from_iterable(
                factor_denom(sub_expr) for sub_expr in factor_eri(itmd)
            )
            itmd = e.expr(0, **itmd.assumptions)
            for term in reduced:
                itmd += term.factor()
            factored_variants[tuple()] = itmd
        # factor all previously factored intermediates in the current itmd
        if factored_itmds:
            available = intermediates().available
            factored = []
            for it in factored_itmds:
                cache_key = tuple([*factored, it])
                try:  # try to load from cache
                    itmd = factored_variants[cache_key]
                except KeyError:
                    itmd = available[it].factor_itmd(
                        itmd, factored_itmds=factored, max_order=self.order
                    )
                    factored_variants[cache_key] = itmd
                factored.append(it)
        return itmd

    def _determine_target_idx(self, sub: dict) -> list:
        """Returns the target indices of the itmd if the provided
           substitutions are applied to the default intermediate."""
        return [sub.get(idx, idx) for idx in get_symbols(self.default_idx)]

    def _minimal_itmd_indices(self, remainder: e.expr, sub: dict):
        """Minimize the target indices of the intermediate to factor."""
        from .indices import get_first_missing_index, get_symbols, index_space

        if not isinstance(remainder, e.expr) or len(remainder) != 1:
            raise Inputerror("Expected an expr of length 1 as remainder.")

        # determine the intermediate target indices if sub is applied
        itmd_indices = self._determine_target_idx(sub)

        # find all target indices that can not be changed
        # the remainder has to have the the target indices explicitly set!
        # an index can only occur once in provided_target_idx
        target = remainder.provided_target_idx
        used = defaultdict(list)
        for s in target:
            used[index_space(s.name)].append(s.name)
        # iterate over all itmd_indices and see if we can find a lower index
        minimization_sub = {}
        for idx in itmd_indices:
            if idx in target:  # skip target indices
                continue
            # find the lowest available index
            name = idx.name
            ov = index_space(name)
            new_idx = get_first_missing_index(used[ov], ov)
            used[ov].append(new_idx)
            if name == new_idx:  # already have the lowest index
                continue
            # found a lower index -> build a substitution dictionary that
            # leads to a minimization of the itmd indices if applied
            new_idx = get_symbols(new_idx)[0]
            # build a new substitution dict that minimizes the itmd indices
            additional_sub = {idx: new_idx, new_idx: idx}
            # immediately apply to the itmd_indices
            for i, s in enumerate(itmd_indices):
                itmd_indices[i] = additional_sub.get(s, s)
            # and build a minimization sub to minimize the remainder
            # with a single call to sub
            if not minimization_sub:
                minimization_sub = additional_sub
            else:
                for old, new in minimization_sub.items():
                    if new is new_idx:
                        minimization_sub[old] = idx
                        del additional_sub[new_idx]
                    elif new is idx:
                        minimization_sub[old] = new_idx
                        del additional_sub[idx]
                if additional_sub:
                    minimization_sub.update(additional_sub)

        # permute/minimize the remainder
        # the input remainder has to have a prefactor of +1!
        remainder = remainder.subs(minimization_sub, simultaneous=True)
        pref = remainder.terms[0].prefactor  # possibly a -1!
        remainder *= pref
        return remainder, itmd_indices, pref

    def factor_itmd(self, expr: e.expr, factored_itmds: list[str] = None,
                    max_order: int = None) -> e.expr:
        from .factor_intermediates import (_factor_long_intermediate,
                                           _factor_short_intermediate)
        from collections import Counter

        if not isinstance(expr, e.expr):
            raise Inputerror("Expr to factor needs to be an instance of "
                             f"{e.expr}.")
        if not expr.real:
            raise NotImplementedError("Intermediates only implemented for "
                                      "a real orbital basis.")
        if expr.sympy.is_number or \
                (factored_itmds and self.name in factored_itmds) or \
                self.name == 't4_2':
            return expr

        expr = expr.expand()
        if max_order is not None and max_order < self.order:
            return expr

        # try to reduce the number of terms that can be factored
        to_factor = e.expr(0, **expr.assumptions)
        remainder = e.expr(0, **expr.assumptions)
        max_found_order = 0
        for term in expr.terms:
            order = term.order
            max_found_order = max(max_found_order, order)
            if order < self.order:  # order high enough?
                remainder += term
            # amplitudes need (except t4_2) need to have some denominator
            elif self.itmd_type == 't_amplitude' and self.name != 't4_2':
                if any(o.exponent < 0 and o.contains_only_orb_energies
                       for o in term.objects):
                    to_factor += term
                else:
                    remainder += term
            else:
                to_factor += term
        if to_factor.sympy is S.Zero:  # don't have anything to factor
            return expr

        if factored_itmds is None:  # transform to tuple -> use as dict key
            factored_itmds = tuple()
        else:
            factored_itmds = tuple(factored_itmds)

        # prepare the intermediate before factorization
        itmd_expr = self._prepare_itmd(factored_itmds=factored_itmds)

        # extract data from the intermediate
        itmd: list[eri_orbenergy] = [eri_orbenergy(term).canonicalize_sign()
                                     for term in itmd_expr.terms]
        itmd_data: list[dict] = []
        itmd_tensors: list[Counter] = []
        for term in itmd:
            itmd_pattern = term.eri_pattern(include_exponent=False,
                                            target_idx_string=False)
            itmd_indices = [o.idx for o in term.eri.objects]
            itmd_obj_sym = [o.symmetry() for o in term.eri.objects]
            itmd_data.append({'itmd_pattern': itmd_pattern,
                              'itmd_indices': itmd_indices,
                              'itmd_obj_sym': itmd_obj_sym})
            itmd_tensors.append(Counter([t.name for t in term.eri.tensors
                                         for _ in range(t.exponent)]))

        # filter further terms
        temp = e.expr(0, **to_factor.assumptions)
        for term in to_factor.terms:
            # does the term contain all necessary tensors?
            tensors = Counter([t.name for t in term.tensors
                               for _ in range(t.exponent)])
            if any(all(tensors[t] >= n for t, n in itmd_term_tensors.items())
                   for itmd_term_tensors in itmd_tensors):
                temp += term
            else:
                remainder += term
        to_factor = temp
        if to_factor.sympy is S.Zero:
            return expr

        # actually factor the expression
        if len(itmd) == 1:  # short intermediate -> only a single term
            itmd: eri_orbenergy = itmd[0]
            itmd_pattern: dict = itmd_pattern[0]
            itmd_tensors: Counter = itmd_tensors[0]
            itmd_data: dict = itmd_data[0]
            expr = _factor_short_intermediate(to_factor, itmd, itmd_data, self)
            expr += remainder.sympy
        else:  # long intermediate -> multiple terms
            itmd_term_map = self.itmd_term_map(factored_itmds)
            for _ in range(max_found_order // self.order):
                to_factor = _factor_long_intermediate(
                    to_factor, itmd, itmd_data, itmd_term_map, self
                )
            expr = to_factor + remainder.sympy
        return expr

# -----------------------------------------------------------------------------
# INTERMEDIATE DEFINITIONS:


class t2_1(registered_intermediate):
    """First order MP doubles amplitude."""
    _itmd_type: str = 't_amplitude'  # type has to be a class variable

    def __init__(self):
        self._order: int = 1
        self._default_idx: tuple[str] = ('i', 'j', 'a', 'b')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, lower=None, upper=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, a, b = idx
        # actually build the amplitude
        denom = orb_energy(a) + orb_energy(b) - orb_energy(i) - orb_energy(j)
        t2 = eri((a, b, i, j)) / denom
        t2 = e.expr(t2, target_idx=(a, b, i, j))
        self.__cache[idx] = t2
        return t2

    def tensor(self, indices=None, lower=None, upper=None) -> e.expr:
        # guess its not worth caching here. Maybe if used a lot.
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        upper = idx[2:]
        lower = idx[:2]
        # build the tensor
        t2 = AntiSymmetricTensor('t1', upper, lower)
        return e.expr(t2)

    def factor_itmd(self, expr: e.expr, factored_itmds: list[str] = None,
                    max_order: int = None):
        from .factor_intermediates import _compare_obj
        # special function for factoring t2_1 amplitudes
        # maybe replace this later with a function for intermediates of length
        # 1, i.e., intermediates that consist of a single term
        if not isinstance(expr, e.expr):
            raise Inputerror("Expr to factor needs to be an instance of "
                             f"{e.expr}.")
        if not expr.real:
            raise NotImplementedError("Intermediates only implemented for a "
                                      "real orbital basis.")
        # do we have something to factor? did we already factor the itmd?
        if expr.sympy.is_number or \
                factored_itmds and self.name in factored_itmds:
            return expr

        expr = expr.expand()
        if max_order is None:
            max_order = max((term.order for term in expr.terms))
        if max_order < self.order:  # the order of the itmd is too high
            return expr

        # prepare the itmd and extract information
        t2 = self.expand_itmd().make_real()
        t2 = eri_orbenergy(t2).canonicalize_sign()
        t2_eri = t2.eri.objects[0]
        t2_eri_descr = t2_eri.description(include_exponent=False)
        t2_eri_idx = t2_eri.idx
        t2_eri_sym = e.expr(t2_eri.sympy).terms[0].symmetry()

        substituted_expr = e.expr(0, **expr.assumptions)
        for term in expr.terms:
            term = eri_orbenergy(term).canonicalize_sign()
            # print(f"\nterm = {term}")
            if term.denom.sympy.is_number:  # term needs to have a denominator
                substituted_expr += term.expr
                continue
            pref = term.pref
            denom = term.denom
            substituted_term = e.expr(1, **term.eri.assumptions)
            eri_indices = []
            denom_indices = []
            for eri_idx, eri in enumerate(term.eri.objects):
                if (sub := _compare_obj(eri, t2_eri, [], [],
                                        itmd_obj_descr=t2_eri_descr,
                                        itmd_obj_idx=t2_eri_idx,
                                        itmd_obj_sym=t2_eri_sym)):
                    # no need to consider the symmetry here! just use the first
                    # sub dictionary... all should be valid!
                    sub, factor = sub[0]
                    t2_sub_denom = t2.denom.copy().subs(sub, simultaneous=True)
                    if (denom_idx := term.find_matching_braket(t2_sub_denom)):
                        if len(denom_idx) != 1:
                            raise RuntimeError("Found more than one possible "
                                               "match for a denominator.")
                        denom_indices.extend(denom_idx)
                        eri_indices.append(eri_idx)
                        target = self._determine_target_idx(sub)
                        # construct the t2 ampltude and multiply to the result
                        itmd = self.tensor(indices=target).sympy
                        # adjust the final pref (sign) for factoring the itmd
                        pref /= t2.pref * factor
                        substituted_term *= itmd
            # remove the matched eri
            denom = term.cancel_denom_brakets(denom_indices)
            eri = term.cancel_eri_objects(eri_indices)
            # substituted everything in the term
            substituted_term *= eri * term.num * pref / denom
            # print(f"substituted_term = {substituted_term}")
            substituted_expr += substituted_term
        return substituted_expr


class t1_2(registered_intermediate):
    """Second order MP singles amplitude."""
    _itmd_type: str = "t_amplitude"

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'a')
        self._factored_variants: dict[tuple, e.expr] = {}
        self._term_map_cache: dict[tuple, dict] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, lower=None, upper=None) -> e.expr:
        from sympy import Rational
        from .indices import indices as idx_cls
        # target_indices
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, a = idx
        # additional contracted indices
        contracted = idx_cls().get_generic_indices(n_o=2, n_v=2)
        j, k = contracted['occ']
        b, c = contracted['virt']
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the amplitude
        denom = orb_energy(i) - orb_energy(a)
        term1 = (Rational(1, 2) *
                 t2.expand_itmd(upper=(b, c), lower=(i, j)).sympy *
                 eri([j, a, b, c]))
        term2 = (Rational(1, 2) *
                 t2.expand_itmd(upper=(a, b), lower=(j, k)).sympy *
                 eri([j, k, i, b]))
        t1 = e.expr(term1/denom + term2/denom, target_idx=(i, a))
        self.__cache[idx] = t1
        return t1

    def tensor(self, indices=None, lower=None, upper=None) -> e.expr:
        # target indices
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        i, a = idx
        t1 = AntiSymmetricTensor('t2', (a,), (i,))
        return e.expr(t1)


class t2_2(registered_intermediate):
    """Second order MP doubles amplitude."""
    _itmd_type: str = 't_amplitude'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'a', 'b')
        self._factored_variants: dict[tuple, e.expr] = {}
        self._term_map_cache: dict[tuple, dict] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, lower=None, upper=None) -> e.expr:
        from .indices import indices as idx_cls
        from sympy import Rational

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, a, b = idx
        # generate additional contracted indices (2o / 2v)
        contracted = idx_cls().get_generic_indices(n_o=2, n_v=2)
        k, l = contracted['occ']  # noqa: E741
        c, d = contracted['virt']
        # t2_1 class instance for generating t2_1 amplitudes
        t2_ampl: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the t2_2 amplitude
        denom = (orb_energy(a) + orb_energy(b) - orb_energy(i) - orb_energy(j))
        # - 0.5 t2eri_3
        term1 = (- Rational(1, 2) *
                 t2_ampl.expand_itmd(lower=(k, l), upper=(a, b)).sympy *
                 eri((i, j, k, l)))
        # - 0.5 t2eri_5
        term2 = (- Rational(1, 2) *
                 t2_ampl.expand_itmd(lower=(i, j), upper=(c, d)).sympy *
                 eri((a, b, c, d)))
        # + (1 - P_ij) (1 - P_ab) P_ij t2eri_4
        base = (
            t2_ampl.expand_itmd(lower=(i, k), upper=(a, c)) * eri((k, b, j, c))
        )
        term3 = (base.sympy - base.copy().permute((i, j)).sympy
                 - base.copy().permute((a, b)).sympy
                 + base.copy().permute((i, j), (a, b)).sympy)
        t2 = e.expr((term1 + term2 + term3) / denom, target_idx=(i, j, a, b))
        self.__cache[idx] = t2
        return t2

    def tensor(self, indices=None, lower=None, upper=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        lower = idx[:2]
        upper = idx[2:]
        t2 = AntiSymmetricTensor('t2', upper, lower)
        return e.expr(t2)


class t3_2(registered_intermediate):
    """Second order MP triples amplitude."""
    _itmd_type: str = 't_amplitude'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'k', 'a', 'b', 'c')
        self._factored_variants: dict[tuple, e.expr] = {}
        self._term_map_cache: dict[tuple, dict] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, lower=None, upper=None) -> e.expr:
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, k, a, b, c = idx
        # generate additional contracted indices (1o / 1v)
        contracted = idx_cls().get_generic_indices(n_o=1, n_v=1)
        l, d = contracted['occ'][0], contracted['virt'][0]
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the t3_2 amplitude
        denom = (orb_energy(i) + orb_energy(j) + orb_energy(k)
                 - orb_energy(a) - orb_energy(b) - orb_energy(c))
        # (1 - P_ik - P_jk) (1 - P_ab - P_ac) <kd||bc> t_ij^ad
        base = t2.expand_itmd(indices=(i, j, a, d)) * eri((k, d, b, c))
        term1 = (base.sympy - base.copy().permute((i, k)).sympy
                 - base.copy().permute((j, k)).sympy
                 - base.copy().permute((a, b)).sympy
                 - base.copy().permute((a, c)).sympy
                 + base.copy().permute((i, k), (a, b)).sympy
                 + base.copy().permute((i, k), (a, c)).sympy
                 + base.copy().permute((j, k), (a, b)).sympy
                 + base.copy().permute((j, k), (a, c)).sympy)
        # (1 - P_ij - P_ik) (1 - P_ac - P_bc) <jk||lc> t_il^ab
        base = t2.expand_itmd(indices=(i, l, a, b)) * eri((j, k, l, c))
        term2 = (base.sympy - base.copy().permute((i, j)).sympy
                 - base.copy().permute((i, k)).sympy
                 - base.copy().permute((a, c)).sympy
                 - base.copy().permute((b, c)).sympy
                 + base.copy().permute((i, j), (a, c)).sympy
                 + base.copy().permute((i, j), (b, c)).sympy
                 + base.copy().permute((i, k), (a, c)).sympy
                 + base.copy().permute((i, k), (b, c)).sympy)
        t3 = e.expr(term1/denom + term2/denom, target_idx=(i, j, k, a, b, c))
        self.__cache[idx] = t3
        return t3

    def tensor(self, indices=None, lower=None, upper=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        lower = idx[:3]
        upper = idx[3:]
        t2 = AntiSymmetricTensor('t2', upper, lower)
        return e.expr(t2)


class t4_2(registered_intermediate):
    """Second order MP quadruple amplitudes in a factorized form that avoids
       the construction of the quadruples denominator."""
    _itmd_type: str = 't_amplitude'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'k', 'l',
                                         'a', 'b', 'c', 'd')
        self._factored_variants: dict[tuple, e.expr] = {}
        self._term_map_cache: dict[tuple, dict] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, lower=None, upper=None) -> e.expr:
        from itertools import product

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, k, l, a, b, c, d = idx
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the t4_2 amplitude
        # (1 - P_ac - P_ad - P_bc - P_bd + P_ac P_bd) (1 - P_jk - P_jl)
        #  t_ij^ab t_kl^cd
        base: e.expr = (t2.expand_itmd(indices=(i, j, a, b)) *
                        t2.expand_itmd(indices=(k, l, c, d)).sympy)
        v_permutations = {tuple(tuple()): 1, ((a, c),): -1, ((a, d),): -1,
                          ((b, c),): -1, ((b, d),): -1, ((a, c), (b, d)): +1}
        o_permutations = {tuple(tuple()): 1, ((j, k),): -1, ((j, l),): -1}
        t4 = 0
        for (o_perms, o_factor), (v_perms, v_factor) in \
                product(o_permutations.items(), v_permutations.items()):
            perms = o_perms + v_perms
            t4 += o_factor * v_factor * base.copy().permute(*perms).sympy
        t4 = e.expr(t4, target_idx=(i, j, k, l, a, b, c, d))
        self.__cache[idx] = t4
        return t4

    def tensor(self, indices=None, lower=None, upper=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        lower = idx[:4]
        upper = idx[4:]
        t2 = AntiSymmetricTensor('t2', upper, lower)
        return e.expr(t2)


class t1_3(registered_intermediate):
    """Third order MP triple amplitudes."""
    _itmd_type: str = 't_amplitude'

    def __init__(self):
        self._order: int = 3
        self._default_idx: tuple[str] = ('i', 'a')
        self._factored_variants: dict[tuple, e.expr] = {}
        self._term_map_cache: dict[tuple, dict] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, lower=None, upper=None) -> e.expr:
        from sympy import Rational
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, a = idx
        # generate additional contracted indices (2o / 2v)
        contracted = idx_cls().get_generic_indices(n_o=2, n_v=2)
        j, k = contracted['occ']
        b, c = contracted['virt']
        # other intermediate class instances
        t1: t1_2 = self._registry['t_amplitude']['t1_2']
        t2: t2_2 = self._registry['t_amplitude']['t2_2']
        t3: t3_2 = self._registry['t_amplitude']['t3_2']
        # build the amplitude
        denom = orb_energy(i) - orb_energy(a)
        terms = (Rational(1, 2) *
                 t2.expand_itmd(indices=(i, j, b, c)).sympy *
                 eri([j, a, b, c]))
        terms += (Rational(1, 2) *
                  t2.expand_itmd(indices=(j, k, a, b)).sympy *
                  eri([j, k, i, b]))
        terms += (- t1.expand_itmd(indices=(j, b)).sympy * eri([i, b, j, a]))
        terms += (Rational(1, 4) *
                  t3.expand_itmd(indices=(i, j, k, a, b, c)).sympy *
                  eri([j, k, b, c]))
        itmd = e.expr(terms / denom, target_idx=(i, a))
        self.__cache[idx] = itmd
        return itmd

    def tensor(self, indices=None, lower=None, upper=None) -> e.expr:
        i, a = self.validate_indices(indices=indices, lower=lower, upper=upper)
        t3 = AntiSymmetricTensor('t3', (a,), (i,))
        return e.expr(t3)


class t2_3(registered_intermediate):
    """Third order MP triple amplitudes."""
    _itmd_type: str = 't_amplitude'

    def __init__(self):
        self._order: int = 3
        self._default_idx: tuple[str] = ('i', 'j', 'a', 'b')
        self._factored_variants: dict[tuple, e.expr] = {}
        self._term_map_cache: dict[tuple, dict] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, lower=None, upper=None) -> e.expr:
        from sympy import Rational
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, a, b = idx
        # generate additional contracted indices (2o / 2v)
        contracted = idx_cls().get_generic_indices(n_o=2, n_v=2)
        k, l = contracted['occ']  # noqa E741
        c, d = contracted['virt']
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
        terms = base.sympy - base.permute((i, j)).sympy
        # +(1-P_ab) * <ij||ka> t^b_k(2)
        base = t1.expand_itmd(indices=(k, b)) * eri((i, j, k, a))
        terms += base.sympy - base.permute((a, b)).sympy
        # - 0.5 * <ab||cd> t^cd_ij(2)
        terms -= Rational(1, 2) * \
            t2.expand_itmd(indices=(i, j, c, d)).sympy * eri((a, b, c, d))
        # - 0.5 * <ij||kl> t^ab_kl(2)
        terms -= Rational(1, 2) * \
            t2.expand_itmd(indices=(k, l, a, b)).sympy * eri((i, j, k, l))
        # + (1-P_ij)*(1-P_ab) * <jc||kb> t^ac_ik(2)
        base = t2.expand_itmd(indices=(i, k, a, c)) * eri((j, c, k, b))
        terms += base.sympy - base.copy().permute((i, j)).sympy \
            - base.copy().permute((a, b)).sympy \
            + base.copy().permute((i, j), (a, b)).sympy
        # + 0.5 * (1-P_ab) * <ka||cd> t^bcd_ijk(2)
        base = t3.expand_itmd(indices=(i, j, k, b, c, d)) * eri((k, a, c, d))
        terms += Rational(1, 2) * base.sympy \
            - Rational(1, 2) * base.copy().permute((a, b)).sympy
        # + 0.5 * (1-P_ij) <kl||ic> t^abc_jkl(2)
        base = t3.expand_itmd(indices=(j, k, l, a, b, c)) * eri((k, l, i, c))
        terms += Rational(1, 2) * base.sympy \
            - Rational(1, 2) * base.copy().permute((i, j)).sympy
        # + 0.25 <kl||cd> t^abcd_ijkl(2)
        terms += Rational(1, 4) * \
            t4.expand_itmd(indices=(i, j, k, l, a, b, c, d)).sympy * \
            eri((k, l, c, d))
        # - 0.25 <kl||cd> t^ab_ij(1) t^kl_cd(1)
        terms -= Rational(1, 4) * \
            _t2_1.expand_itmd(indices=(i, j, a, b)).sympy * \
            _t2_1.expand_itmd(indices=(k, l, c, d)).sympy * \
            eri((k, l, c, d))
        itmd = e.expr(terms / denom, target_idx=(i, j, a, b))
        self.__cache[idx] = itmd
        return itmd

    def tensor(self, indices=None, lower=None, upper=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        lower = idx[:2]
        upper = idx[2:]
        t2 = AntiSymmetricTensor('t3', upper, lower)
        return e.expr(t2)


class p0_2_oo(registered_intermediate):
    """Occupied Occupied block of the 2nd order contribution of the MP density
    """
    _itmd_type: str = 'mp_density'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from sympy import Rational
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, upper=upper, lower=lower)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j = idx
        # additional contracted indices (1o / 2v)
        contracted = idx_cls().get_generic_indices(n_o=1, n_v=2)
        k = contracted['occ'][0]
        a, b = contracted['virt']
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the density
        p0 = (- Rational(1, 2) *
              t2.expand_itmd(lower=(i, k), upper=(a, b)).sympy *
              t2.expand_itmd(lower=(j, k), upper=(a, b)).sympy)
        p0 = e.expr(p0, target_idx=(i, j))
        self.__cache[idx] = p0
        return p0

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, upper=upper, lower=lower)
        i, j = idx
        p0 = AntiSymmetricTensor('p2', (i,), (j,), 1)
        return e.expr(p0, sym_tensors=["p2"])


class p0_2_vv(registered_intermediate):
    """Virtual Virtual block of the 2nd order contribution of the MP density"""
    _itmd_type: str = 'mp_density'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('a', 'b')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from sympy import Rational
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, upper=upper, lower=lower)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        a, b = idx
        # additional contracted indices (2o / 1v)
        contracted = idx_cls().get_generic_indices(n_o=2, n_v=1)
        i, j = contracted['occ']
        c = contracted['virt'][0]
        # t2_1 class instance
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the density
        p0 = (Rational(1, 2) * t2.expand_itmd(indices=(i, j, a, c)).sympy *
              t2.expand_itmd(indices=(i, j, b, c)).sympy)
        p0 = e.expr(p0, target_idx=(a, b))
        self.__cache[idx] = p0
        return p0

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, upper=upper, lower=lower)
        a, b = idx
        p0 = AntiSymmetricTensor('p2', (a,), (b,), 1)
        return e.expr(p0, sym_tensors=["p2"])


class p0_3_oo(registered_intermediate):
    """Occupied Occupied block of the 2nd order contribution of the MP density
    """
    _itmd_type: str = 'mp_density'

    def __init__(self):
        self._order: int = 3
        self._default_idx: tuple[str] = ('i', 'j')
        self._factored_variants: dict[tuple, e.expr] = {}
        self._term_map_cache: dict[tuple, dict] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from sympy import Rational
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, upper=upper, lower=lower)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j = idx
        # generate additional contracted indices (1o / 2v)
        contracted = idx_cls().get_generic_indices(n_o=1, n_v=2)
        k = contracted['occ'][0]
        a, b = contracted['virt']
        # t amplitude cls
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        td2: t2_2 = self._registry['t_amplitude']['t2_2']
        # build the density
        p0 = - Rational(1, 2) * t2.expand_itmd(indices=(i, k, a, b)).sympy * \
            td2.expand_itmd(indices=(j, k, a, b)).sympy
        p0 += p0.subs({i: j, j: i}, simultaneous=True)
        p0 = e.expr(p0, target_idx=(i, j))
        self.__cache[idx] = p0
        return p0

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        i, j = self.validate_indices(indices=indices, upper=upper, lower=lower)
        p0 = AntiSymmetricTensor('p3', (i,), (j,), 1)
        return e.expr(p0, sym_tensors=["p2"])


class p0_3_ov(registered_intermediate):
    """Occupied Occupied block of the 2nd order contribution of the MP density
    """
    _itmd_type: str = 'mp_density'

    def __init__(self):
        self._order: int = 3
        self._default_idx: tuple[str] = ('i', 'a')
        self._factored_variants: dict[tuple, e.expr] = {}
        self._term_map_cache: dict[tuple, dict] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from sympy import Rational
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, upper=upper, lower=lower)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, a = idx
        # generate additional contracted indices (2o / 2v)
        contracted = idx_cls().get_generic_indices(n_o=2, n_v=2)
        j, k = contracted['occ']
        b, c = contracted['virt']
        # t_amplitude cls instances
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        ts2: t1_2 = self._registry['t_amplitude']['t1_2']
        tt2: t3_2 = self._registry['t_amplitude']['t3_2']
        ts3: t1_3 = self._registry['t_amplitude']['t1_3']
        # build the density
        # - t^ab_ij(1) t^b_j(2)
        p0 = - t2.expand_itmd(indices=(i, j, a, b)).sympy * \
            ts2.expand_itmd(indices=(j, b)).sympy
        # - 0.25 * t^bc_jk(1) t^abc_ijk(2)
        p0 -= Rational(1, 4) * t2.expand_itmd(indices=(j, k, b, c)).sympy * \
            tt2.expand_itmd(indices=(i, j, k, a, b, c)).sympy
        # + t^a_i(3)
        p0 += ts3.expand_itmd(indices=(i, a)).sympy
        p0 = e.expr(p0, target_idx=(i, a))
        self.__cache[idx] = p0
        return p0

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        i, a = self.validate_indices(indices=indices, upper=upper, lower=lower)
        p0 = AntiSymmetricTensor('p3', (i,), (a,), 1)
        return e.expr(p0, sym_tensors=["p3"])


class p0_3_vv(registered_intermediate):
    """Virtual Virtual block of the 2nd order contribution of the MP density"""
    _itmd_type: str = 'mp_density'

    def __init__(self):
        self._order: int = 3
        self._default_idx: tuple[str] = ('a', 'b')
        self._factored_variants: dict[tuple, e.expr] = {}
        self._term_map_cache: dict[tuple, dict] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from sympy import Rational
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, upper=upper, lower=lower)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        a, b = idx
        # additional contracted indices (2o / 1v)
        contracted = idx_cls().get_generic_indices(n_o=2, n_v=1)
        i, j = contracted['occ']
        c = contracted['virt'][0]
        # t_amplitude cls instances
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        td2: t2_2 = self._registry['t_amplitude']['t2_2']
        # build the density
        p0 = Rational(1, 2) * t2.expand_itmd(indices=(i, j, a, c)).sympy * \
            td2.expand_itmd(indices=(i, j, b, c)).sympy
        p0 += p0.subs({a: b, b: a}, simultaneous=True)
        p0 = e.expr(p0, target_idx=(a, b))
        self.__cache[idx] = p0
        return p0

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        a, b = self.validate_indices(indices=indices, upper=upper, lower=lower)
        p0 = AntiSymmetricTensor('p3', (a,), (b,), 1)
        return e.expr(p0, sym_tensors=["p3"])


class t2eri_1(registered_intermediate):
    """t2eri1 in adcc / pi1 in libadcc."""
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'k', 'a')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, k, a = idx
        # generate additional contracted indices (2v)
        b, c = idx_cls().get_generic_indices(n_v=2)['virt']
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = t2.expand_itmd(indices=(i, j, b, c)) * eri((k, a, b, c))
        t2eri = e.expr(t2eri, target_idx=(i, j, k, a))
        self.__cache[idx] = t2eri
        return t2eri

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        upper = idx[:2]
        lower = idx[2:]
        t2eri = AntiSymmetricTensor('t2eri1', upper, lower)
        return e.expr(t2eri)


class t2eri_2(registered_intermediate):
    """t2eri2 in adcc / pi2 in libadcc."""
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'k', 'a')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, k, a = idx
        # generate additional contracted indices (1o / 1v)
        contracted = idx_cls().get_generic_indices(n_o=1, n_v=1)
        b, l = contracted['virt'][0], contracted['occ'][0]  # noqa E741
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = t2.expand_itmd(indices=(i, l, a, b)) * eri((l, k, j, b))
        t2eri = e.expr(t2eri, target_idx=(i, j, k, a))
        self.__cache[idx] = t2eri
        return t2eri

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        t2eri = NonSymmetricTensor('t2eri2', idx)
        return e.expr(t2eri)


class t2eri_3(registered_intermediate):
    """t2eri3 in adcc / pi3 in libadcc."""
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'a', 'b')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, a, b = idx
        # generate additional contracted indices (2o)
        k, l = idx_cls().get_generic_indices(n_o=2)['occ']  # noqa E741
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = t2.expand_itmd(indices=(k, l, a, b)) * eri((i, j, k, l))
        t2eri = e.expr(t2eri, target_idx=(i, j, a, b))
        self.__cache[idx] = t2eri
        return t2eri

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        upper = idx[:2]
        lower = idx[2:]
        t2eri = AntiSymmetricTensor('t2eri3', upper, lower)
        return e.expr(t2eri)


class t2eri_4(registered_intermediate):
    """t2eri4 in adcc / pi4 in libadcc."""
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'a', 'b')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, a, b = idx
        # generate additional contracted indices (1o / 1v)
        contracted = idx_cls().get_generic_indices(n_o=1, n_v=1)
        k, c = contracted['occ'][0], contracted['virt'][0]
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = t2.expand_itmd(indices=(j, k, a, c)) * eri((k, b, i, c))
        t2eri = e.expr(t2eri, target_idx=(i, j, a, b))
        self.__cache[idx] = t2eri
        return t2eri

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        t2eri = NonSymmetricTensor('t2eri4', idx)
        return e.expr(t2eri)


class t2eri_5(registered_intermediate):
    """t2eri5 in adcc / pi5 in libadcc."""
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'a', 'b')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, a, b = idx
        # generate additional contracted indices (2v)
        c, d = idx_cls().get_generic_indices(n_v=2)['virt']
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = t2.expand_itmd(indices=(i, j, c, d)).sympy * eri((a, b, c, d))
        t2eri = e.expr(t2eri, target_idx=(i, j, a, b))
        self.__cache[idx] = t2eri
        return t2eri

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        upper = idx[:2]
        lower = idx[2:]
        t2eri = AntiSymmetricTensor('t2eri5', upper, lower)
        return e.expr(t2eri)


class t2eri_6(registered_intermediate):
    """t2eri6 in adcc / pi6 in libadcc."""
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'a', 'b', 'c')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, a, b, c = idx
        # generate additional contracted indices (2o)
        j, k = idx_cls().get_generic_indices(n_o=2)['occ']
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = t2.expand_itmd(indices=(j, k, b, c)).sympy * eri((j, k, i, a))
        t2eri = e.expr(t2eri, target_idx=(i, a, b, c))
        self.__cache[idx] = t2eri
        return t2eri

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        upper = idx[:2]
        lower = idx[2:]
        t2eri = AntiSymmetricTensor('t2eri6', upper, lower)
        return e.expr(t2eri)


class t2eri_7(registered_intermediate):
    """t2eri7 in adcc / pi7 in libadcc."""
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'a', 'b', 'c')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, a, b, c = idx
        # generate additional contracted indices (1o / 1v)
        contracted = idx_cls().get_generic_indices(n_o=1, n_v=1)
        j, d = contracted['occ'][0], contracted['virt'][0]
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        t2eri = t2.expand_itmd(indices=(i, j, b, d)).sympy * eri((j, c, a, d))
        t2eri = e.expr(t2eri, target_idx=(i, a, b, c))
        self.__cache[idx] = t2eri
        return t2eri

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        t2eri = NonSymmetricTensor('t2eri7', idx)
        return e.expr(t2eri)


class t2sq(registered_intermediate):
    """t2sq intermediate from adcc and libadc."""
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'a', 'j', 'b')
        self._factored_variants: dict[tuple, e.expr] = {}
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from .indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, a, j, b = idx
        # generate additional contracted indices (1o / 1v)
        contracted = idx_cls().get_generic_indices(n_o=1, n_v=1)
        c = contracted['virt'][0]
        k = contracted['occ'][0]
        # t2_1 class instance for generating t2_1 amplitudes
        t2: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the intermediate
        itmd = t2.expand_itmd(indices=(i, k, a, c)).sympy * \
            t2.expand_itmd(indices=(j, k, b, c)).sympy
        itmd = e.expr(itmd, target_idx=(i, a, j, b))
        self.__cache[idx] = itmd
        return itmd

    def tensor(self, indices=None, upper=None, lower=None) -> e.expr:
        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        upper = idx[:2]
        lower = idx[2:]
        itmd = AntiSymmetricTensor('t2sq', upper, lower, 1)
        return e.expr(itmd)


def eri(idx: str | list[Dummy] | list[str]) -> AntiSymmetricTensor:
    """Builds an electron repulsion integral using the provided indices.
       Indices may be provided as list of sympy symbols or as string."""

    idx = get_symbols(idx)
    if len(idx) != 4:
        raise Inputerror(f'4 indices required to build a ERI. Got: {idx}.')
    return AntiSymmetricTensor('V', idx[:2], idx[2:])


def orb_energy(idx: str | Dummy) -> NonSymmetricTensor:
    """Builds an orbital energy using the provided index.
       Indices may be provided as list of sympy symbols or as string."""

    idx = get_symbols(idx)
    if len(idx) != 1:
        raise Inputerror("1 index required to build a orbital energy. Got: "
                         f"{idx}.")
    return NonSymmetricTensor('e', idx)
