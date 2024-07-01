from .misc import Inputerror
from . import expr_container as e
from .sympy_objects import SymmetricTensor, SymbolicTensor
from .logger import logger

from sympy import Pow, S, Mul, Basic


class EriOrbenergy:
    """
    Splits a single term into an orbital energy fraction, a prefactor and a
    remainder.

    Parameters
    ----------
    term : Term
        The term to split.
    """

    def __init__(self, term: e.Term | e.Expr):
        # ensure the input consists of a single term either as term or expr
        if not isinstance(term, e.Term) or not len(term) == 1:
            Inputerror("Expected a single term as input.")
        # factor the term to ensure all prefactors are in the numerator
        term: e.Expr = term.factor()  # returns an expr

        # split the term in num, denom and eri
        term = term.terms[0].split_orb_energy()
        # validate the denominator: has to be of the form: (a+b)(c+d) or (a+b)
        self._denom = term['denom']
        self._validate_denom()

        # validate eri: has to consist of a single term
        eri = term['remainder']
        if len(eri) != 1:
            raise Inputerror("Remainder/ERI part should consist of a single "
                             f"term. Got {eri} from term {term}.")
        self._eri: e.Term = eri.terms[0]

        # numerator can essentially be anything: a or a+b
        # extract the prefactor with the smallest abs value from the numerator
        # NOTE: this is not mandatory. It is also possible to just use
        #       term.prefactor as pref. Then we might have prefactors < 1
        #       in the numerator. Should not be important except for
        #       canceling the orbital energy fraction.
        #       But if we keep it like it is, we should have a more clear
        #       definition of the prefactor (only the sign might be ambiguous
        #       +0.5 vs -0.5)
        self._pref = min([t.prefactor for t in term['num'].terms], key=abs)

        # only possiblity to extract 0 should be if the numerator is 0
        if self._pref is S.Zero:
            if not term['num'].sympy.is_number:
                raise NotImplementedError(f"Extracted pref {self._pref} from "
                                          "unexpected numerator "
                                          f"{term['num']}")
            self._num = term['num']
        elif self._pref is S.One:  # nothing to factor
            self._num = term['num']
        else:
            self._num: e.Expr = \
                (term['num'].factor(self._pref) / self._pref).doit()
        # ensure that the numerator is what we expect
        self._validate_num()

    def __str__(self):
        return f"{self.pref} * [{self.num}] / [{self.denom}] * {self.eri}"

    def _validate_denom(self):
        """
        Ensures that the denominator only consists of brackets of the form
        (e_a + e_b - ...)(...).
        """
        # only objects that contain only e tensors with a single idx can
        # occur in the denominator

        if self._denom.sympy.is_number:  # denom is a number -> has to be 1
            if self._denom.sympy is not S.One:
                raise Inputerror(f"Invalid denominator {self._denom}")
        else:
            # check that each bracket consists of terms that each contain
            # a single epsilon and possibly a prefactor of -1
            for bracket in self.denom_brackets:
                if not isinstance(bracket, (e.Expr, e.Polynom)):
                    raise TypeError(f"Invalid bracket {bracket} in "
                                    f"{self._denom}.")
                for term in bracket.terms:
                    n_orb_energy = 0
                    for o in term.objects:
                        base, exp = o.base_and_exponent
                        if isinstance(base, SymbolicTensor) and exp == 1:
                            n_orb_energy += 1
                        # denominator has to contain prefactors +- 1
                        # prefactors need to be +-1 for cancelling to work
                        elif o.sympy.is_number and o.sympy is S.NegativeOne:
                            continue
                        else:
                            raise Inputerror(f"Invalid bracket {bracket} in "
                                             f"{self._denom}.")
                    if n_orb_energy != 1:
                        raise Inputerror(f"Invalid bracket {bracket} in "
                                         f"{self._denom}.")

    def _validate_num(self):
        """
        Ensures that the numerator is of the form (e_a + e_b - ...) only
        allowing prefactors +-1.
        """
        # numerator can only contain terms that consist of e tensors with a
        # single index and prefactors
        # checking that each term only contains a single tensor with exponent 1
        # ensures that each term only holds a single index

        if self._num.sympy.is_number:  # is a number -> 1 or 0 possible
            if self._num.sympy not in [S.One, S.Zero]:
                raise Inputerror(f"Invalid numerator {self._num}.")
        else:  # an expr object (a + b + ...)
            for term in self._num.terms:
                n_orb_energy = 0
                for o in term.objects:
                    base, exp = o.base_and_exponent
                    if isinstance(base, SymbolicTensor) and exp == 1:
                        n_orb_energy += 1
                    elif o.sympy.is_number:  # any prefactors allowed
                        continue
                    else:
                        raise Inputerror(f"Invalid object {o} in {self._num}.")
                if n_orb_energy != 1:
                    raise Inputerror(f"Invalid term {term} in numerator "
                                     f"{self._num}.")

    @property
    def denom(self) -> e.Expr:
        """Returns the denominator of the orbital energy fraction."""
        return self._denom

    @property
    def eri(self) -> e.Term:
        """Returns the remainder of the term."""
        return self._eri

    @property
    def num(self) -> e.Expr:
        """Returns the numerator of the orbital energy fraction."""
        return self._num

    @property
    def pref(self):  # sympy rational
        """Returns the prefactor of the term."""
        return self._pref

    @property
    def denom_brackets(self) -> tuple[e.Polynom] | tuple[e.Expr]:
        """Returns a tuple containing the brackets of the denominator."""
        if len(self.denom) != 1 or self.denom.sympy.is_number:
            return (self.denom,)
        else:  # denom consists of brackets
            return self.denom.terms[0].objects

    def copy(self):
        return EriOrbenergy(self.expr)

    @property
    def expr(self):
        """Rebuild the original term."""
        return self.num * self.eri / self.denom * self.pref

    def denom_description(self) -> str | None:
        """
        Returns a string that describes the denominator containing the
        number of brackets, as well as the length and exponent of each
        bracket.
        """
        if self.denom.is_number:
            return None

        brackets = self.denom_brackets
        bracket_data = []
        for bk in brackets:
            exponent = 1 if isinstance(bk, e.Expr) else bk.exponent
            bracket_data.append(f"{len(bk)}-{exponent}")
        # reverse sorting -> longest braket will be listed first
        bracket_data = "_".join(sorted(bracket_data, reverse=True))
        return f"{len(brackets)}_{bracket_data}"

    def cancel_denom_brackets(self, braket_idx_list: list[int]) -> e.Expr:
        """
        Cancels brackets by their index in the denominator lowering the
        exponent by 1 or removing the bracket completely if an exponent
        of 0 is reached. If an index is listed n times the exponent
        will be lowered by n.
        The original denominator is not modified.
        """
        from collections import Counter

        denom = list(self.denom_brackets)
        for idx, n in Counter(braket_idx_list).items():
            braket = denom[idx]
            if isinstance(braket, e.Expr):
                exponent = 1
                base = braket.sympy
            else:
                base, exponent = braket.base_and_exponent
            if (new_exp := exponent - n) == 0:
                denom[idx] = None
            else:
                denom[idx] = Pow(base, new_exp)
        new_denom = Mul(*(bk if isinstance(bk, Basic) else bk.sympy
                          for bk in denom if bk is not None))
        return e.Expr(new_denom, **self.denom.assumptions)

    def cancel_eri_objects(self, obj_idx_list: list[int]) -> e.Expr:
        """
        Cancels objects in the remainder (eri) part according to their index
        lowering their exponent by 1 for each time the objects index is
        provided. If a final exponent of 0 is reached, the object is removed
        from the remainder entirely.
        The original remainder is not changed.
        """
        from collections import Counter

        objects = list(self.eri.objects)
        for idx, n in Counter(obj_idx_list).items():
            obj = objects[idx]
            base, exponent = obj.base_and_exponent
            if (new_exp := exponent - n) == 0:
                objects[idx] = None
            else:
                objects[idx] = Pow(base, new_exp)
        new_eri = Mul(*(obj if isinstance(obj, Basic) else obj.sympy
                        for obj in objects if obj is not None))
        return e.Expr(new_eri, **self.eri.assumptions)

    def denom_eri_sym(self, eri_sym: dict = None, **kwargs):
        """
        Apply the symmetry of the remainder (eri) part to the denominator
        identifying the common symmetry of both parts of the term.

        Parameters
        ----------
        eri_sym : dict, optional
            The symmetry of the remainder (eri) part of the term.
            If not provided it will be determined on the fly.
        **kwargs : dict, optional
            Additional arguments that are forwarded to the 'Term.symmetry'
            method to determine the symmetry of the remainder on the fly.
        """
        # if the denominator is a number -> just return symmetry of eri part
        if self.denom.sympy.is_number:
            return self.eri.symmetry(**kwargs) if eri_sym is None else eri_sym

        if eri_sym is None:
            # if the eri part is just a number all possible permutations of the
            # denom would be required with their symmetry
            if not self.eri.idx:
                raise NotImplementedError("Symmetry of an expr (the "
                                          "denominator) not implemented")
            eri_sym = self.eri.symmetry(**kwargs)

        ret = {}
        denom = self.denom.sympy
        for perms, factor in eri_sym.items():
            perm_denom = self.denom.copy().permute(*perms).sympy
            # permutations are not valid for the denominator
            if perm_denom is S.Zero and denom is not S.Zero:
                continue

            if denom - perm_denom is S.Zero:
                ret[perms] = factor  # P_pq Denom = Denom -> +1
            elif denom + perm_denom is S.Zero:
                ret[perms] = factor * -1  # P_pq Denom = -Denom -> -1
            else:  # permutation changes the denominator
                ret[perms] = None
        return ret

    def permute_num(self, eri_sym: dict = None):
        """
        Symmetrize the orbital energy numerator by applying the common symmetry
        of the remainder (eri) part and the orbital energy denominator
        - only considering contracted indices! - to the numerator keeping the
        result normalized.
        For instance, a numerator (e_i - e_a) may be expanded to
        1/2 (e_i + e_j - e_a - e_b) by applying the permutation P_{ij}P_{ab}.
        The new prefactor is automatically extracted from the
        new numerator and added to the existing prefactor.
        The class instance is modified in place.
        """
        from sympy import Rational
        # if the numerator is a number no permutation will do anything useful
        if self.num.sympy.is_number:
            return self
        # apply all permutations to the numerator that satisfy
        # P_pq ERI = a * ERI and P_pq Denom = b * Denom
        # with a, b in [-1, +1] and a*b = 1
        permutations = [(perms, factor) for perms, factor in
                        self.denom_eri_sym(eri_sym=eri_sym,
                                           only_contracted=True).items()
                        if factor is not None]
        num = self.num.copy()
        for perms, factor in permutations:
            num += self.num.copy().permute(*perms) * factor
        num = (num * Rational(1, len(permutations) + 1)).expand()
        # this possibly introduced prefactors in the numerator again
        # -> extract the smallest prefactor and shift to self.pref
        additional_pref = min([t.prefactor for t in num.terms], key=abs)
        self._pref *= additional_pref
        if additional_pref is S.Zero:  # permuted num = 0
            if not num.is_number:
                raise ValueError("Only expected to obtain 0 as pref"
                                 "from a 0 numerator. Got "
                                 f"{additional_pref} from {num}.")
            self._num = num
        elif additional_pref is S.One:  # nothing to factor
            self._num = num
        else:
            self._num: e.Expr = (
                (num.factor(additional_pref) / additional_pref).doit()
            )
        self._validate_num()
        return self

    def canonicalize_sign(self, only_denom: bool = False):
        """
        Adjusts the sign of orbital energies in the numerator and denominator:
        virtual orbital energies are subtracted, while occupied orbital
        energies are added. The possible factor of -1 is extracted to the
        prefactor.
        Modifies the class instance in place.

        Parameters
        ----------
        only_denom : bool, optional
            If set, only the signs in the denominator will be adjusted
            (default: False).
        """

        def adjust_sign(expr: e.Expr | e.Polynom) -> bool:
            # function that extracts the sign of the occupied and virtual
            # indices in a term.

            signs = {}
            for term in expr.terms:
                idx = term.idx
                if len(idx) != 1:
                    raise RuntimeError("Expected a bracket to consist of "
                                       "epsilons that each hold a single index"
                                       f". Found: {term} in {expr}.")
                ov = idx[0].space[0]
                if ov not in signs:
                    signs[ov] = []
                signs[ov].append(term.sign)

            # map that connects sign and space
            desired_sign = {'o': 'plus', 'v': 'minus'}

            # adjust sign if necessary
            change_sign = []
            for ov, sign in signs.items():
                # first check that all o/v terms have the same sign
                if not all(pm == sign[0] for pm in sign):
                    raise RuntimeError(f"Ambiguous signs of the {ov} indices "
                                       f"in {expr} in\n{self}")
                if ov not in desired_sign:
                    raise NotImplementedError("No desired sign defined for "
                                              "orbital energies of the space "
                                              f"{ov}.")
                if sign[0] != desired_sign[ov]:
                    change_sign.append(True)
            if change_sign:
                if len(change_sign) != len(signs):
                    raise RuntimeError(f"Apparently not all {signs.keys()} "
                                       "spaces require a sign change in "
                                       f"{expr}.")
                return True
            else:
                return False

        # numerator
        if not only_denom and not self.num.sympy.is_number and \
                adjust_sign(self.num):
            self._pref *= -1
            self._num *= -1

        # denominator
        if not self.denom.sympy.is_number:
            denom = 1
            for bracket in self.denom_brackets:
                if adjust_sign(bracket):
                    if isinstance(bracket, e.Expr):
                        exponent = 1
                        base = bracket.sympy
                    else:
                        base, exponent = bracket.base_and_exponent
                    if exponent % 2:
                        self._pref *= -1
                    bracket = e.Expr(Pow(-1*base, exponent),
                                     **bracket.assumptions)
                denom *= bracket
            self._denom = denom
        return self

    def cancel_orb_energy_frac(self) -> e.Expr:
        """
        Cancel the orbital energy fraction. Thereby, long denominator brackets
        or brackets with rare indices are priorized.
        """
        from collections import Counter

        def multiply(expr_list: list[e.Expr]) -> int | e.Expr:
            res = 1
            for term in expr_list:
                res *= term
            return res

        def cancel(num: e.Expr, denom: list, pref) -> e.Expr:
            num = num.copy()  # avoid in place modification
            cancelled_result = None
            for bracket_i, bracket in enumerate(denom):
                bracket_indices = bracket.idx

                # get the prefactors of all orbital energies that occur in the
                # bracket that we currently want to remove
                relevant_prefs = [term.prefactor for term in num.terms
                                  if term.idx[0] in bracket_indices]
                # do all indices that occur in the bracket also occur in the
                # numerator?
                if len(relevant_prefs) != len(bracket_indices):
                    continue

                # find the smallest relevant prefactor and factor the prefactor
                # -> this ensures that at least one of the relevant orbital
                #    energies has a prefactor of 1
                # -> at least 1 of the orbital energies will not be present
                #    in the new numerator
                # -> can only cancel each bracket at most 1 time
                # -> no need to recurse just iterate through the list
                min_pref = min(relevant_prefs, key=abs)
                # the sign in the numerator has been fixed before entering this
                # function -> dont change it!
                if min_pref < 0:
                    min_pref *= -1

                if min_pref is not S.One:
                    pref *= min_pref
                    num = (num.factor(min_pref) / min_pref).doit()

                # all orbital energies that also occur in the bracket now
                # have at least a prefactor of 1
                # others might have a pref < 1
                # construct the new numerator by subtracting the content
                # of the bracket from the numerator. This works, because
                #  - all relevant orbital energies in the numerator have a
                #    prefactor of at least 1 and the signs in the numerator and
                #    in the bracket match
                #  - all orbital energies in the numerator have an exponent
                #    of 1
                if isinstance(bracket, e.Expr):
                    exponent = 1
                    base = bracket.sympy
                else:  # polynom
                    base, exponent = bracket.base_and_exponent
                logger.info(f"Cancelling: {e.Expr(base)}")
                num -= base
                # build the new denominator -> lower bracket exponent by 1
                if exponent == 1:
                    new_denom = denom[:bracket_i] + denom[bracket_i+1:]
                else:
                    new_denom = denom[:]
                    new_denom[bracket_i] = e.Expr(
                        Pow(base, exponent-1), **bracket.assumptions
                    )
                # result <- 1/new_denom + new_num/denom
                if cancelled_result is None:
                    cancelled_result = 0
                cancelled_result += pref * self.eri / multiply(new_denom)
                # check if we have something left to cancel
                if num.sympy.is_number:
                    if num.sympy is not S.Zero:
                        cancelled_result += \
                            pref * self.eri * num / multiply(denom)
                    break
            # return just the term if it was not possible to successfully
            # cancel any bracket
            return self.expr if cancelled_result is None else cancelled_result

        # fix the sign of the orbital energies in numerator and denominator:
        # occupied orb energies are added, while virtual ones are subtracted
        self.canonicalize_sign()

        # do we have something to do?
        if self.num.sympy.is_number or self.denom.sympy.is_number:
            return self.expr

        # sort the brackets in the denominator:
        #  - length of the braket: tiples > doubles
        #  - rarity of the contained indices: prioritize brackets with target
        #    indices
        denom_indices = Counter(self.denom.idx)

        def bracket_sort_key(bracket):
            bracket_indices = bracket.idx
            rarest_idx = min(bracket_indices, key=lambda s: denom_indices[s])
            return (-len(bracket),
                    denom_indices[rarest_idx],
                    sum(denom_indices[s] for s in bracket_indices))
        denom = sorted(self.denom_brackets, key=bracket_sort_key)

        return cancel(self.num, denom, self.pref)

    def symbolic_denominator(self):
        """
        Replaces the explicit orbital energy denominator with a tensor
        of the correct symmetry (a SymmetricTensor with bra-ket antisymmetry):
        (e_i + e_j - e_a - e_b) -> D^{ij}_{ab}.
        """
        if self.denom.sympy.is_number:  # denom is a number -> nothing to do
            return self.denom
        symbolic_denom = e.Expr(1, **self.denom.assumptions)
        has_symbolic_denom = False
        for bracket in self.denom_brackets:
            signs = {'-': set(), '+': set()}
            for term in bracket.terms:
                idx = term.idx
                if len(idx) != 1:
                    raise RuntimeError("Expected a denominator bracket to "
                                       "consists of orbital energies that each"
                                       " hold a single index. "
                                       f"Found: {term} in {bracket}.")
                pref = term.prefactor
                if pref is S.One:
                    signs['+'].add(idx[0])
                elif pref is S.NegativeOne:
                    signs['-'].add(idx[0])
                else:
                    raise RuntimeError(f"Found invalid prefactor {pref} in "
                                       f"denominator bracket {bracket}.")
            if signs['+'] & signs['-']:
                raise RuntimeError(f"Found index {idx} that is added and "
                                   f"subtracted in a denominator: {bracket}.")
            has_symbolic_denom = True
            exponent = 1 if isinstance(bracket, e.Expr) else bracket.exponent
            symbolic_denom *= Pow(
                SymmetricTensor("D", signs['+'], signs['-'], -1), exponent
            )
        if has_symbolic_denom:
            symbolic_denom.set_antisym_tensors(
                symbolic_denom.antisym_tensors + ("D",)
            )
        return symbolic_denom
