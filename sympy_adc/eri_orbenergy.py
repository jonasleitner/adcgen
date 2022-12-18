from .misc import Inputerror
from . import expr_container as e
from sympy import Pow, S


class eri_orbenergy:
    """Class that holds term, split in numerator, denominator and a remainder:
       (e_i + e_j) / (e_i + e_j - e_a + e_b) * eri1 * eri2
       The remainder is assumed to contain any tensors/deltas, but the orbital
       energy tensor e."""

    def __init__(self, term):
        # factor the term to ensure all prefactors are in the numerator
        term = term.factor()  # returns an expr
        if len(term) == 1:
            term = term.terms[0]
        else:
            raise Inputerror("Multiple terms provided.")
        # split the term in num, denom and eri
        term = term.split_orb_energy()
        # check that the denominator is of the form (a+b)*(c+d)*...
        # (a+b+c) - only a single braket - is not of length 1.
        # In this case additionally check the number of tensors included
        # in each term (should be only a single epsilon per term)
        denom = term['denom']
        if len(denom) == 1 and not denom.sympy.is_number:
            for braket in denom.terms[0].objects:
                if braket.type != 'polynom':
                    raise RuntimeError("Expected a denominator of length 1 to "
                                       f"consist of polynoms. Found {braket}.")
                if any(len([o for o in t.objects if o.type != 'prefactor'])
                       != 1 for t in braket.terms):
                    raise Inputerror("Expected a denominator of the form "
                                     f"(a+b)*(c+d). Got {denom}.")
        elif len(denom) != 1:
            if any(len([o for o in t.objects if o.type != 'prefactor']) != 1
                   for t in denom.terms):
                raise Inputerror("Expected denominator of the form (a+b+c)."
                                 f"Got {denom}.")
        self.__denom: e.expr = denom
        eri = term['remainder']
        if len(eri) != 1:
            raise Inputerror("Remainder/ERI part should consist of a single "
                             f"term. Got {eri} from term {term}.")
        self.__eri: e.term = eri.terms[0]
        # numerator can essentially be anything: a or a+b
        # factor out prefactor from numerator
        self.__pref = min([t.prefactor for t in term['num'].terms], key=abs)
        self.__num: e.expr = \
            (term['num'].factor(self.__pref) / self.__pref).doit()

    def __str__(self):
        return self.expr.__str__()

    @property
    def denom(self):  # expr
        return self.__denom

    @property
    def eri(self):  # term
        return self.__eri

    @property
    def num(self):  # expr
        return self.__num

    @property
    def pref(self):  # sympy rational
        return self.__pref

    @property
    def denom_brakets(self) -> list[e.polynom] | list[e.expr]:
        return self.denom.terms[0].objects if len(self.denom) == 1 else \
            [self.denom]

    def copy(self):
        return eri_orbenergy(self.expr)

    @property
    def expr(self):
        return self.num * self.eri / self.denom * self.pref

    def find_matching_braket(self, other_braket):
        """Checks whether a braket is equal to a braket in the denominator.
           Returns the index of the braket if this is the case."""
        if isinstance(other_braket, e.expr):
            other_exponent = 1
            other_braket = other_braket.sympy
        else:  # polynom
            other_exponent = other_braket.exponent
            other_braket = other_braket.extract_pow
        for braket_idx, braket in enumerate(self.denom_brakets):
            braket = braket.sympy if isinstance(braket, e.expr) else \
                braket.extract_pow
            if other_braket == braket:
                return [braket_idx for _ in range(other_exponent)]

    def eri_pattern(self, include_exponent=True, target_idx_string=True) \
            -> dict[int, tuple[str, list[str] | None]]:
        """Returns the pattern for each obj in the eri part as tuple
           (description, coupling)."""
        descriptions = [o.description(include_exponent=include_exponent)
                        for o in self.eri.objects]
        coupling = self.eri.coupling(target_idx_string=target_idx_string,
                                     include_exponent=include_exponent)
        return {i: (descr, coupling.get(i))
                for i, descr in enumerate(descriptions)}

    def cancel_denom_brakets(self, braket_idx_list):
        """Cancels brakets by index in the denominator, i.e., the exponent
           is lowered by 1, or the braket is completely removed if a exponent
           of 0 is reached. The new denominator is returned and the original
           object not changed."""
        from collections import Counter

        denom = self.denom_brakets
        for idx, n in Counter(braket_idx_list).items():
            braket = denom[idx]
            if isinstance(braket, e.expr):
                exponent = 1
                base = braket.sympy
            else:
                exponent = braket.exponent
                base = braket.extract_pow
            if exponent - n == 0:
                denom[idx] = None
            else:
                denom[idx] = e.expr(Pow(base, exponent - n),
                                    **braket.assumptions)
        new_denom = e.expr(1, **self.denom.assumptions)
        for remaining_braket in denom:
            if remaining_braket is None:
                continue
            new_denom *= remaining_braket
        return new_denom

    def cancel_eri_objects(self, obj_idx_list):
        """Cancels objects in the eri part by index, i.e., the exponent is
           lowered by 1 for each time the object index is provided. If a final
           exponent of 0 is reached, the object is removed entirely.
           The new eri are returned an the original object not changed."""
        from collections import Counter

        objects = self.eri.objects
        for idx, n in Counter(obj_idx_list).items():
            obj = objects[idx]
            if (exp := obj.exponent) - n == 0:
                objects[idx] = None
            else:
                objects[idx] = e.expr(
                    Pow(obj.extract_pow, exp - n), **obj.assumptions
                )
        new_eri = e.expr(1, **self.eri.assumptions)
        for remaining_obj in objects:
            if remaining_obj is None:
                continue
            new_eri *= remaining_obj
        return new_eri

    def denom_eri_sym(self, **kwargs):
        """Apply all Symmetries of the ERI's to the denominator. If the
           denominator is also symmetric or anti-symmetric, the overall
           symmetry of ERI and Denom is determined (+-1). Otherwise
           the factor it set to None.
           """
        # if the denominator is a number -> just return symmetry of eri part
        if self.denom.sympy.is_number:
            return self.eri.symmetry(**kwargs)
        # if the eri part is just a number all possible permutations of the
        # denom would be required with their symmetry
        elif self.eri.sympy.is_number:
            raise NotImplementedError("Symmetry of an expr not implemented")
        ret = {}
        for perms, factor in self.eri.symmetry(**kwargs).items():
            perm_denom = self.denom.copy().permute(*perms)
            if self.denom.sympy - perm_denom.sympy is S.Zero:
                denom_factor = +1
            elif self.denom.sympy + perm_denom.sympy is S.Zero:
                denom_factor = -1
            else:  # permutation changes the denominator
                ret[perms] = None
                continue
            ret[perms] = factor * denom_factor
        return ret

    def permute_num(self):
        from sympy import Rational
        # if the numerator is a number no permutation will do anything useful
        if self.num.sympy.is_number:
            return self
        # apply all permutations to the numerator that satisfy
        # P_pq ERI = a * ERI and P_pq Denom = b * Denom
        # with a, b in [-1, +1] and a*b = 1
        permutations = [perms for perms, factor in
                        self.denom_eri_sym(only_contracted=True).items()
                        if factor == 1]
        num = self.num.copy()
        for perms in permutations:
            num += self.num.copy().permute(*perms)
        self.__num = num * Rational(1, len(permutations) + 1)
        return self

    def canonicalize_sign(self):
        """Adjusts the sign of numerator and denominator:
           all virtual orbital energies will be added, while all occupied
           energies are subtracted. This might change numerator, denominator
           and prefactor."""
        from .indices import index_space

        def adjust_sign(expr):
            # function that extracts the sign of the occupied and virtual
            # indices in a term.

            signs = {}
            for term in expr.terms:
                idx = term.idx
                if len(idx) != 1:
                    raise RuntimeError("Expected a braket to consist of "
                                       "epsilons that each hold a single index"
                                       f". Found: {term} in {expr}.")
                ov = index_space(idx[0].name)[0]
                if ov not in signs:
                    signs[ov] = []
                signs[ov].append(term.sign)

            # map that connects sign and o/v
            desired_sign = {'o': 'plus', 'v': 'minus'}

            # adjust sign if necessary
            change_sign = []
            for ov, sign in signs.items():
                # first check that all o/v terms have the same sign
                if not all(pm == sign[0] for pm in sign):
                    raise RuntimeError("Ambiguous signs of the {ov} indices"
                                       f"in {expr}.")
                if sign[0] != desired_sign[ov]:
                    change_sign.append(True)
            if change_sign and len(change_sign) != len(signs):
                raise RuntimeError(f"Apparently not all {[ov.keys()]} spaces "
                                   f"require a sign change in {expr}.")
            return change_sign and all(change_sign)

        # numerator
        if not self.num.sympy.is_number and adjust_sign(self.num):
            self.__pref *= -1
            self.__num *= -1

        # denominator
        if not self.denom.sympy.is_number:
            denom = e.compatible_int(1)
            for braket in self.denom_brakets:
                if adjust_sign(braket):
                    if isinstance(braket, e.expr):
                        exponent = 1
                        base = braket.sympy
                    else:
                        exponent = braket.exponent
                        base = braket.extract_pow
                    if exponent % 2:
                        self.__pref *= -1
                    braket = e.expr(Pow(-1*base, exponent),
                                    **braket.assumptions)
                denom *= braket
            self.__denom = denom
        return self

    def cancel_orb_energy_frac(self) -> e.expr:
        """Try to cancel numerator and denominator."""
        from collections import Counter
        # - canonicalize the sign, i.e. all occ orbital energies are added and
        #   all virt orbital energies subtracted.
        self.canonicalize_sign()

        # if numerator or denominator are just a number -> nothing to do
        if self.num.sympy.is_number or self.denom.sympy.is_number:
            return self.expr

        # sort the brakets in the denominator by
        # 1) length (Try to cancel Triple denoms before Double denoms)
        # 2) by the rarity of the included indices (try to cancel brakets with
        #    rare indices (e.g. target indices) first)
        denom_idx = Counter(self.denom.idx)  # count the occurences

        def sort_denom(braket):
            idx = braket.idx
            return (len(braket),
                    denom_idx[min(idx, key=lambda s: denom_idx[s])],
                    sum(denom_idx[s] for s in idx))
        denom = sorted(self.denom_brakets, key=sort_denom)

        def cancel(num, denom):
            num_idx = num.idx
            for i, braket in enumerate(denom):
                # check if all indices of the braket are also in the numerator
                # if this is not the case -> can't cancel
                if not all(s in num_idx for s in braket.idx):
                    continue
                # denom = (a+b+c), single braket with exponent = 1
                #   -> exponent=1; no polynom but an expr
                if isinstance(braket, e.expr):
                    exponent = 1
                    to_subtract = braket
                # braket is part of (a+b)*(c+d) -> polynom.
                else:
                    exponent = braket.exponent
                    to_subtract = braket.extract_pow
                # construct new num by subtracting the braket that is canceled
                new_num = num - to_subtract
                # construct the new denominator:
                new_denom = denom[:i] + denom[i+1:]  # remove braket from denom
                if exponent != 1:  # just lower the exponent by 1
                    new_braket = e.expr(
                        Pow(braket.extract_pow, braket.exponent - 1),
                        **braket.assumptions
                    )
                    new_denom[i] = new_braket
                # build the new denom also as expr (currently it's a list)
                new_denom_expr = e.compatible_int(1)
                for bk in new_denom:
                    new_denom_expr *= bk
                # result = 1 / new_denom + new_num / denom
                # call cancel recursively to check whether it is possible to
                # cancel more brakets in new_num / denom
                return self.eri * self.pref / new_denom_expr + \
                    cancel(new_num, denom)
            # it was not possible to cancel any more brakets in the denom
            # result = num / denom
            denom_expr = e.compatible_int(1)
            for bk in denom:
                denom_expr *= bk
            return num * self.pref * self.eri / denom_expr
        return cancel(self.num, denom)
