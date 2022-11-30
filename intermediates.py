from indices import get_symbols, index_space
from misc import Inputerror, Singleton
import expr_container as e
from simplify import make_real
from eri_orbenergy import eri_orbenergy
from sympy_objects import NonSymmetricTensor

from sympy.physics.secondquant import AntiSymmetricTensor
from sympy import S, Dummy


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

    @property
    def symmetric(self) -> bool:
        if (symmetric := getattr(self, '_symmetric', None)) is None:
            raise AttributeError(f"Symmetric not defined for {self.name}")
        return symmetric

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

    def itmd_term_map(self, itmd: e.expr = None, real: bool = False) \
            -> tuple[dict[tuple[str], dict[int, list[int]]], list[int]]:
        """Determines which permutations of the itmd target indices transform
           a certain itmd_term into another one, e.g.,
           ... + (1 - P_ij) * (1 - P_ab) * x for the t2_2 amplitudes.
           The target indices of the provided itmd expr need to be set
           to the target indices of the itmd."""
        from math import factorial
        from itertools import combinations, permutations, chain, product
        from indices import split_idx_string

        def permute_str(string: str, *perms: tuple[Dummy]) -> str:
            string: list = split_idx_string(string)
            for perm in perms:
                p, q = [s.name for s in perm]
                i1 = string.index(p)
                i2 = string.index(q)
                string[i1] = q
                string[i2] = p
            return "".join(string)

        if itmd is None:  # use the default expanded definition in this case
            itmd: e.expr = self.expand_itmd().expand()
            if real:
                itmd.make_real()
        if not isinstance(itmd, e.expr):
            raise Inputerror(f"Itmd needs to be an instance of {e.expr}.")
        # the target indices of the itmd expression should be set to the
        # the target indices of the itmd at this point.
        idx = {'o': [], 'v': []}
        for s in itmd.provided_target_idx:  # each index can only occur once
            idx[index_space(s.name)[0]].append(s)
        sym = {'o': [], 'v': []}
        for ov, ov_idx in idx.items():
            max_n_perms = factorial(len(ov_idx))
            # represent the indices as string that is permuted to keep
            # track which permutations give a new, unique result
            idx_string = "".join([s.name for s in ov_idx])
            permuted_str = [idx_string]
            # all symbol pairs
            pairs = list(combinations(ov_idx, 2))
            # all combinations of pairs -> more permutations than needed
            # smaller permutations are obtained first, i.e., those of length 1
            combs = chain.from_iterable(
                [permutations(pairs, r) for r in range(1, len(ov_idx))]
            )
            for perms in combs:
                # did we already find enough permutations?
                if len(permuted_str) == max_n_perms:
                    break
                # permute the string and check if the permutation is redundant
                perm_str = permute_str(idx_string, *perms)
                if perm_str in permuted_str:
                    continue
                # found a new relevant permutation
                permuted_str.append(perm_str)
                sym[ov].append(perms)
        # multiply the occ and virt permutations
        symmetry = sym['o'] + sym['v']
        symmetry.extend(
            o_perm + v_perm for o_perm, v_perm in product(sym['o'], sym['v'])
        )
        del sym
        # now that we have found all relevant permutations: apply them to
        # each term and check if the result gives another term
        is_zero = {
            True: lambda x: make_real(x, *x.sym_tensors).sympy is S.Zero,
            False: lambda x: x.sympy is S.Zero
        }
        is_zero = is_zero[itmd.real]
        itmd_terms = itmd.terms
        unmapped_terms = [i for i in range(len(itmd_terms))]
        itmd_term_indices = unmapped_terms.copy()
        term_map = {}
        for perms in symmetry:
            for i, other_i in combinations(itmd_term_indices, 2):
                perm_term = itmd_terms[i].permute(*perms)
                other_term = itmd_terms[other_i]
                factor = None
                if is_zero(perm_term + other_term):
                    # P_pq X + (- P_pq X) = 0
                    factor = -1
                elif is_zero(perm_term - other_term):
                    # P_pq X - (+ P_pq X) = 0
                    factor = 1
                else:
                    continue
                if perms not in term_map:
                    term_map[perms] = {}
                if i not in term_map[perms]:
                    term_map[perms][i] = []
                if other_i not in term_map[perms]:
                    term_map[perms][other_i] = []
                term_map[perms][i].append((other_i, factor))
                term_map[perms][other_i].append((i, factor))
                if i in unmapped_terms:
                    unmapped_terms.remove(i)
                if other_i in unmapped_terms:
                    unmapped_terms.remove(other_i)
        return term_map, unmapped_terms

    def _prepare_itmd(self, factored_itmds: list[str] = None,
                      real: bool = False) -> e.expr:
        """"Factor all previously factorized in intermediates in the current
            intermediate."""
        itmd: e.expr = self.expand_itmd().expand()
        if real:
            itmd.make_real()
        if factored_itmds:
            available = intermediates().available
            factored = []
            for it in factored_itmds:
                itmd = available[it].factor_itmd(
                    itmd, factored_itmds=factored, max_order=self.order
                )
                factored.append(it)
        return itmd

    def _determine_target_idx(self, sub: dict, itmd_term_map: dict = None) \
            -> list | tuple[list, dict]:
        """Returns the target indices of the itmd if the provided
           substitutions are applied to the default intermediate. Also
           the permutations in the itmd_term_map are translated as well."""
        target = [sub.get(idx, idx) for idx in get_symbols(self.default_idx)]
        if itmd_term_map is None:
            return target
        translated_term_map = {}
        for perms, perm_map in itmd_term_map.items():
            perms = tuple(
                tuple(sub.get(s, s) for s in perm) for perm in perms
            )
            translated_term_map[perms] = perm_map
        return target, translated_term_map

    def _minimal_itmd_indices(self, remainder: e.expr, sub: dict,
                              itmd_term_map: dict):
        """Minimize the target indices of the intermediate to factor."""
        from indices import get_first_missing_index, get_symbols, index_space

        if not isinstance(remainder, e.expr) or len(remainder) != 1:
            raise Inputerror("Expected a expr of length 1 as input.")
        itmd_indices, translated_term_map = \
            self._determine_target_idx(sub, itmd_term_map)
        # find all target indices that can not be changed
        # the remainder has to have the the target indices explicitly set!
        used = {'occ': [], 'virt': []}
        for s in remainder.provided_target_idx:
            used[index_space(s.name)].append(s.name)
        # iterate over all itmd_indices and see if we can find a lower index
        for idx in itmd_indices:
            name = idx.name
            ov = index_space(name)
            if name in used[ov]:  # its a target index
                continue
            new_idx = get_first_missing_index(used[ov], ov)
            if name == new_idx:  # already have the lowest index
                used[ov].append(new_idx)
                continue
            # found a lower index -> permute indices in remainder and itmd idx
            used[ov].append(new_idx)
            new_idx = get_symbols(new_idx)[0]
            remainder = remainder.permute((idx, new_idx))
            sub = {idx: new_idx, new_idx: idx}
            for i, other_idx in enumerate(itmd_indices):
                itmd_indices[i] = sub.get(other_idx, other_idx)
            # and in the itmd_term map
            new_term_map = {}
            for perms, perm_map in translated_term_map.items():
                perms = tuple(
                    tuple(sub.get(s, s) for s in perm) for perm in perms
                )
                new_term_map[perms] = perm_map
            translated_term_map = new_term_map
        remainder = eri_orbenergy(remainder)
        pref = remainder.pref  # possible that the substitutions introduce a -1
        remainder = remainder.num * remainder.eri / remainder.denom
        return remainder, itmd_indices, pref, translated_term_map

    def factor_itmd(self, expr: e.expr, factored_itmds: list[str] = None,
                    max_order: int = None) -> e.expr:
        from factor_intermediates import (_factor_long_intermediate,
                                          _factor_short_intermediate)
        from collections import Counter
        if not isinstance(expr, e.expr):
            raise Inputerror("Expr to factor needs to be an instance of "
                             f"{e.expr}.")
        if expr.sympy.is_number or \
                factored_itmds and self.name in factored_itmds:
            return expr
        expr = expr.expand()
        if max_order is None:
            # order not implemented for polynoms
            max_order = max((term.order for term in expr.terms))
        if max_order < self.order:  # the order of the itmd is to high
            return expr
        # try to reduce the number of terms that can be factored
        to_factor = e.expr(0, **expr.assumptions)
        remainder = e.expr(0, **expr.assumptions)
        if self.itmd_type == 't_amplitude':
            # only necessary to consider terms with an denominator
            for term in expr.terms:
                if any(o.exponent < 0 and o.contains_only_orb_energies
                       for o in term.objects):  # do we have a denominator?
                    to_factor += term
                else:
                    remainder += term
        else:  # TODO: add some additional prescreening?
            to_factor = expr
        if to_factor.sympy is S.Zero:  # don't have anything to factor
            return expr
        # prepare the intermediate before factorization
        itmd_expr = self._prepare_itmd(factored_itmds=factored_itmds,
                                       real=expr.real)
        # extract data from the intermediate
        itmd: list[eri_orbenergy] = [eri_orbenergy(term).canonicalize_sign()
                                     for term in itmd_expr.terms]
        itmd_data: list[dict] = []
        for term in itmd:
            itmd_pattern = term.eri_pattern(include_exponent=False,
                                            target_idx_string=False)
            itmd_tensors = Counter([t.name for t in term.eri.tensors
                                    for _ in range(t.exponent)])
            itmd_indices = [o.idx for o in term.eri.objects]
            itmd_obj_sym = [o.symmetry() for o in term.eri.objects]
            itmd_data.append({'itmd_pattern': itmd_pattern,
                              'itmd_tensors': itmd_tensors,
                              'itmd_indices': itmd_indices,
                              'itmd_obj_sym': itmd_obj_sym})
        # actually factor the expression
        if len(itmd) == 1:  # short intermediate -> only a single term
            itmd: eri_orbenergy = itmd[0]
            itmd_pattern: dict = itmd_pattern[0]
            itmd_tensors: Counter = itmd_tensors[0]
            itmd_data: dict = itmd_data[0]
            expr = _factor_short_intermediate(to_factor, itmd, itmd_data, self)
            expr += remainder.sympy
        else:  # long intermediate -> multiple terms
            # create a map that connects terms that can be mapped onto
            # each other through specific index permutations
            itmd_term_map, unmapped_itmd_terms = self.itmd_term_map(itmd_expr)
            for _ in range(max_order // self.order):
                to_factor = _factor_long_intermediate(
                    to_factor, itmd, itmd_data, itmd_term_map,
                    unmapped_itmd_terms, self
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
        self._symmetric: bool = False
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
        from factor_intermediates import _compare_obj
        # special function for factoring t2_1 amplitudes
        # maybe replace this later with a function for intermediates of length
        # 1, i.e., intermediates that consist of a single term
        if not isinstance(expr, e.expr):
            raise Inputerror("Expr to factor needs to be an instance of "
                             f"{e.expr}.")
        # do we have something to factor? did we already factor the itmd?
        if expr.sympy.is_number or \
                factored_itmds and self.name in factored_itmds:
            return expr
        if max_order is None:
            terms = (eri_orbenergy(term) for term in expr.terms)
            max_order = max((term.eri.order for term in terms))
        if max_order < self.order:  # the order of the itmd is too high
            return expr

        expr = expr.expand()
        # prepare the itmd and extract information
        t2 = self.expand_itmd()
        if expr.real:
            t2.make_real()
        t2 = eri_orbenergy(t2)
        t2.canonicalize_sign()
        t2_eri = t2.eri.objects[0]
        t2_eri_descr = t2_eri.description(include_exponent=False)
        t2_eri_idx = t2_eri.idx
        t2_eri_sym = e.expr(t2_eri.sympy).terms[0].symmetry()
        t2_pref = t2.pref * -1 if t2_eri.sign_change else t2.pref

        substituted_expr = e.expr(0, **expr.assumptions)
        for term in expr.terms:
            term = eri_orbenergy(term).canonicalize_sign()
            # print(f"term = {term}")
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
                    sub = sub[0][0]
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
                        # cancelled eri in canonical form do we need to change
                        # the sign to bring it in this form?
                        if eri.sign_change:
                            itmd *= -1
                        # adjust the final pref (sign) for factoring the itmd
                        pref /= t2_pref
                        substituted_term *= itmd
            # remove the matched eri
            denom = term.cancel_denom_brakets(denom_indices)
            eri = term.cancel_eri_objects(eri_indices)
            # substituted everything in the term
            substituted_term *= eri * term.num * pref / denom
            substituted_expr += substituted_term
        return substituted_expr


class t1_2(registered_intermediate):
    """Second order MP singles amplitude."""
    _itmd_type: str = "t_amplitude"

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'a')
        self._symmetric: bool = False
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, lower=None, upper=None) -> e.expr:
        from sympy import Rational
        from indices import indices as idx_cls
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
        self._symmetric: bool = False
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, lower=None, upper=None) -> e.expr:
        from indices import indices as idx_cls
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
        t2_1: t2_1 = self._registry['t_amplitude']['t2_1']
        # build the t2_2 amplitude
        denom = (orb_energy(a) + orb_energy(b) - orb_energy(i) - orb_energy(j))
        # - 0.5 t2eri_3
        term1 = (- Rational(1, 2) *
                 t2_1.expand_itmd(lower=(k, l), upper=(a, b)).sympy *
                 eri((i, j, k, l)))
        # - 0.5 t2eri_5
        term2 = (- Rational(1, 2) *
                 t2_1.expand_itmd(lower=(i, j), upper=(c, d)).sympy *
                 eri((a, b, c, d)))
        # + (1 - P_ij) (1 - P_ab) P_ij t2eri_4
        base = t2_1.expand_itmd(lower=(i, k), upper=(a, c)) * eri((k, b, j, c))
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


class p0_2_oo(registered_intermediate):
    _itmd_type: str = 'mp_density'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j')
        self._symmetric: bool = True
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from sympy import Rational
        from indices import indices as idx_cls

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
        p0 = AntiSymmetricTensor('p2', (i,), (j,))
        return e.expr(p0, sym_tensors=["p2"])


class p0_2_vv(registered_intermediate):
    _itmd_type: str = 'mp_density'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('a', 'b')
        self._symmetric: bool = True
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from sympy import Rational
        from indices import indices as idx_cls

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
        p0 = AntiSymmetricTensor('p2', (a,), (b,))
        return e.expr(p0, sym_tensors=["p2"])


class t2eri_3(registered_intermediate):
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'a', 'b')
        self._symmetric: bool = False
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from indices import indices as idx_cls

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
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'a', 'b')
        self._symmetric: bool = False
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from indices import indices as idx_cls

        idx = self.validate_indices(indices=indices, lower=lower, upper=upper)
        if (itmd := self.__cache.get(idx, None)) is not None:
            return itmd
        i, j, a, b = idx
        # generate additional contracted indices (1o / 1v)
        contracted = idx_cls().get_generic_indices(n_o=1, n_v=1)
        k = contracted['occ'][0]
        c = contracted['virt'][0]
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
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'j', 'a', 'b')
        self._symmetric: bool = False
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from indices import indices as idx_cls

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


class t2sq(registered_intermediate):
    _itmd_type: str = 'misc'

    def __init__(self):
        self._order: int = 2
        self._default_idx: tuple[str] = ('i', 'a', 'j', 'b')
        self._symmetric: bool = False
        self.__cache: dict = {}

    def expand_itmd(self, indices=None, upper=None, lower=None) -> e.expr:
        from indices import indices as idx_cls

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
        itmd = NonSymmetricTensor('t2sq', idx)
        return e.expr(itmd)


def eri(idx: str | list[Dummy] | list[str]) -> AntiSymmetricTensor:
    """Builds an electron repulsion integral using the provided indices.
       Indices may be provided as list of sympy symbols or as string."""

    if len(idx) != 4:
        raise Inputerror(f'4 indices required to build a ERI. Got: {idx}.')
    idx = get_symbols(idx)
    return AntiSymmetricTensor('V', idx[:2], idx[2:])


def orb_energy(idx: str | Dummy) -> NonSymmetricTensor:
    """Builds an orbital energy using the provided index.
       Indices may be provided as list of sympy symbols or as string."""

    idx = get_symbols(idx)
    if len(idx) != 1:
        raise Inputerror("1 index required to build a orbital energy. Got: "
                         f"{idx}.")
    return NonSymmetricTensor('e', idx)
