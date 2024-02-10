from .expr_container import Expr
from .misc import Inputerror
from .rules import Rules
from .indices import Index, get_symbols

from sympy.physics.secondquant import (
    F, Fd, FermionicOperator, KroneckerDelta, NO
)
from sympy import S, Add, Mul


def gen_term_orders(order, term_length, min_order):
    """Generates all combinations that contribute to the n'th order
       contribution of a term x*x*x*..., where x is expanded in a perturbation
       expansion.

       :param order: The desired order
       :type order: int
       :param term_length: The number of objects in the term to expand in
            perturbation theory.
       :type term_length: int
       :param min_order: The minimum order that should be considered
       :type min_order: int
       :return: All possible combinations of a given order
       :rtype: list
       """
    from itertools import product

    if not all(isinstance(n, int) and n >= 0
               for n in [order, term_length, min_order]):
        raise Inputerror("Order, term_length and min_order need to be "
                         "non-negative integers.")

    orders = (o for o in range(min_order, order + 1))
    combinations = product(orders, repeat=term_length)
    return [comb for comb in combinations if sum(comb) == order]


def import_from_sympy_latex(expr_string: str):
    """Function for importing an expression from sympy latex output string.
       Returns an expression container - without any assumptions."""
    import re
    from sympy.physics.secondquant import KroneckerDelta, NO, Fd, F
    from sympy import Pow, sqrt
    from .sympy_objects import NonSymmetricTensor, AntiSymmetricTensor
    from .indices import get_symbols
    from . import expr_container as e

    def import_obj(obj_str: str):

        if obj_str.isnumeric():  # prefactor
            return int(obj_str)
        elif obj_str.startswith("\\sqrt{"):  # sqrt{x} prefactor
            return sqrt(int(obj_str[:-1].replace("\\sqrt{", "", 1)))
        elif obj_str.startswith("\\delta_"):  # KroneckerDelta
            idx = obj_str[:-1].replace("\\delta_{", "", 1).split()
            idx = get_symbols(idx)
            if len(idx) != 2:
                raise RuntimeError(f"Invalid indices for delta: {idx}.")
            return KroneckerDelta(*idx)
        elif obj_str.startswith("\\left("):  # braket
            # need to take care of exponent of the braket!
            base, exponent = obj_str.rsplit('\\right)', 1)
            if exponent:  # exponent != "" -> ^{x} -> exponent != 1
                exponent = int(exponent[:-1].lstrip('^{'))
            else:
                exponent = 1
            obj_str = base.replace("\\left(", "", 1)
            return Pow(import_from_sympy_latex(obj_str).sympy, exponent)
        elif obj_str.startswith("\\left\\{"):  # NO
            no, unexpected_stuff = obj_str.rsplit("\\right\\}", 1)
            if unexpected_stuff:
                raise NotImplementedError(f"Unexpected NO object: {obj_str}.")
            obj_str = no.replace("\\left\\{", "", 1)
            return NO(import_from_sympy_latex(obj_str).sympy)
        else:  # tensor or creation/annihilation operator
            obj = []
            exponent = None
            for component in re.split('\\^|_', obj_str):
                component = component.lstrip("{").rstrip("}")
                if component.isnumeric():
                    if exponent is not None:
                        raise RuntimeError("Found more than one exponent in "
                                           f"{obj_str}.")
                    exponent = int(component)
                else:
                    obj.append(component)
            name, idx = obj[0], obj[1:]
            if name == 'a':  # creation / annihilation operator
                if len(idx) == 2 and idx[0] == "\\dagger":  # creation
                    return Fd(get_symbols(idx[1])[0])
                elif len(idx) == 1:  # annihilation
                    return F(get_symbols(idx[0])[0])
                else:
                    raise NotImplementedError("Unknown second quantized "
                                              f"operator: {obj_str}.")
            elif len(idx) == 1:  # NonSymmetricTensor
                idx = get_symbols(*idx)
                base = NonSymmetricTensor(name, idx)
            elif len(idx) == 2:  # AntiSymmetricTensor
                upper, lower = get_symbols(idx[0]), get_symbols(idx[1])
                base = AntiSymmetricTensor(name, upper, lower)
            else:
                raise NotImplementedError(f"Unknown tensor object: {obj_str}")
            if exponent is not None:
                return Pow(base, exponent)
            else:
                return base

    def split_terms(expr_string: str) -> list[str]:
        stack: list[str] = []
        terms: list[str] = []

        term_start_idx = 0
        for i, char in enumerate(expr_string):
            if char in ['{', '(']:
                stack.append(char)
            elif char == '}':
                assert stack.pop() == '{'
            elif char == ')':
                assert stack.pop() == '('
            elif char in ['+', '-'] and not stack and i != term_start_idx:
                terms.append(expr_string[term_start_idx:i])
                term_start_idx = i
        terms.append(expr_string[term_start_idx:])  # append last term
        return terms

    def import_term(term_string: str) -> list[str]:
        from sympy import Mul

        stack: list[str] = []
        objects: list[str] = []

        obj_start_idx = 0
        for i, char in enumerate(term_string):
            if char in ['{', '(']:
                stack.append(char)
            elif char == '}':
                assert stack.pop() == '{'
            elif char == ')':
                assert stack.pop() == '('
            # in case we have a denom of the form:
            # 2a+2b+4c and not 2 * (a+b+2c)
            elif char in ['+', '-'] and not stack:
                return import_from_sympy_latex(term_string).sympy
            elif char == " " and not stack and i != obj_start_idx:
                objects.append(term_string[obj_start_idx:i])
                obj_start_idx = i + 1
        objects.append(term_string[obj_start_idx:])  # last object
        return Mul(*(import_obj(o) for o in objects))

    expr_string = expr_string.strip()
    if not expr_string:
        return e.Expr(0)

    terms = split_terms(expr_string)
    if terms[0][0] not in ['+', '-']:
        terms[0] = '+ ' + terms[0]

    sympy_expr = 0
    for term in terms:
        sign = term[0]  # extract the sign of the term
        if sign not in ['+', '-']:
            raise ValueError(f"Found invalid sign {sign} in term {term}")
        term = term[1:].strip()

        sympy_term = -1 if sign == '-' else +1

        if term.startswith("\\frac"):  # fraction
            # remove frac layout and split: \\frac{...}{...}
            num, denom = term[:-1].replace("\\frac{", "", 1).split("}{")
        else:  # no denominator
            num, denom = term, None

        sympy_term *= import_term(num)
        if denom is not None:
            sympy_term /= import_term(denom)
        sympy_expr += sympy_term
    return e.Expr(sympy_expr)


def evaluate_deltas(expr, target_idx=None):
    """Slightly modified version of the evaluate_deltas function from sympy
       that takes the target indices of the expr as additional input.
       Neccessary if the einstein sum convention is not sufficient
       to determine the target indices in all terms of the expression."""

    if isinstance(expr, Add):
        return expr.func(*[evaluate_deltas(arg, target_idx)
                           for arg in expr.args])
    elif isinstance(expr, Mul):
        if target_idx is None:
            # for determining the target indices it is sufficient to use
            # atoms, which lists every index only once per object, i.e.,
            # (f_ii).atoms(Index) -> i.
            # We are only interested in indices on deltas
            # -> it is sufficient to know that an index occurs on another
            #    object. (twice on the same delta is not possible)
            deltas = []
            indices = {}
            for obj in expr.args:
                for s in obj.atoms(Index):
                    if s in indices:
                        indices[s] += 1
                    else:
                        indices[s] = 0
                if isinstance(obj, KroneckerDelta):
                    deltas.append(obj)
            target_idx = [s for s, n in indices.items() if not n]
        else:
            # find all occurrences of kronecker delta
            deltas = [d for d in expr.args if isinstance(d, KroneckerDelta)]
            target_idx = get_symbols(target_idx)

        for d in deltas:
            # If we do something, and there are more deltas, we should recurse
            # to treat the resulting expression properly
            if d._get_preferred_index():
                preferred, killable = d.args[1], d.args[0]
            else:
                preferred, killable = d.args[0], d.args[1]
            # killable is contracted -> remove killable
            if killable.is_Symbol and killable not in target_idx:
                expr = expr.subs(killable, preferred)
                if len(deltas) > 1:
                    return evaluate_deltas(expr, target_idx)
            # preferred is a contracted index that can be removed
            elif preferred.is_Symbol and preferred not in target_idx \
                    and preferred.space == killable.space:
                expr = expr.subs(preferred, killable)
                if len(deltas) > 1:
                    return evaluate_deltas(expr, target_idx)
        return expr
    else:
        return expr


def wicks(expr, rules: Rules = None, simplify_kronecker_deltas: bool = False):
    """Evaluates Wicks theorem on the provided expression only returning
       fully contracted contributions.
       The resulting Kronecker deltas are evaluated automatically if
       simplify_kronecker_deltas is set.
       If some rules are provided, they are applied to the
       resulting expression before returning.
       Adapted from 'sympy.physics.secondquant'."""

    # normal ordered operator string has to evaluate to zero
    # and a single second quantized operator can not be contracted
    if isinstance(expr, (NO, FermionicOperator)):
        return S.Zero

    # break up any NO-objects, and evaluate commutators
    expr = expr.doit(wicks=True).expand()

    if isinstance(expr, Add):
        return Add(*[wicks(term, rules=rules,
                           simplify_kronecker_deltas=simplify_kronecker_deltas)
                     for term in expr.args])
    elif isinstance(expr, Mul):
        # we don't want to mess around with commuting part of Mul
        # so we factorize it out before starting recursion
        c_part = []
        op_string = []
        for factor in expr.args:
            if factor.is_commutative:
                c_part.append(factor)
            else:
                op_string.append(factor)

        if (n := len(op_string)) == 0:  # no operators
            result = expr
        elif n == 1:  # a single operator
            return S.Zero
        else:  # at least 2 operators
            result = _contract_operator_string(op_string)
            result = (Mul(*c_part) * result).expand()
            if simplify_kronecker_deltas:
                result = evaluate_deltas(result)

    # apply rules to the result
    if rules is None:
        return result
    elif not isinstance(rules, Rules):
        raise TypeError(f"Rules needs to be of type {Rules}")

    return rules.apply(Expr(result)).sympy


def _contract_operator_string(op_string: list) -> Add:
    """Contracts the provided operator string only returning fully contracted
       contributions.
       Adapted from 'sympy.physics.secondquant'."""
    result = []
    for i in range(1, len(op_string)):
        c = _contraction(op_string[0], op_string[i])
        if c is S.Zero:
            continue
        if not i % 2:  # introduce -1 for swapping operators
            c *= S.NegativeOne

        if len(op_string) - 2 > 0:  # at least one operator left
            # remove the contracted operators from the string and recurse
            remaining = op_string[1:i] + op_string[i+1:]
            result.append(c * _contract_operator_string(remaining))
        else:  # no operators left
            result.append(c)
    return Add(*result)


def _contraction(p, q):
    """Evaluates the contraction of two sqcond quantized fermionic operators.
       Adapted from 'sympy.physics.secondquant'.
    """
    if not isinstance(p, FermionicOperator) or \
            not isinstance(q, FermionicOperator):
        raise NotImplementedError("Contraction only implemented for "
                                  "FermionicOperators.")
    if p.state.spin or q.state.spin:
        raise NotImplementedError("Contraction not implemented for indices "
                                  "with spin.")

    if isinstance(p, F) and isinstance(q, Fd):
        if p.state.assumptions0.get("below_fermi") or \
                q.state.assumptions0.get("below_fermi"):
            return S.Zero
        elif p.state.assumptions0.get("above_fermi") or \
                q.state.assumptions0.get("above_fermi"):
            return KroneckerDelta(p.state, q.state)
        else:
            return (KroneckerDelta(p.state, q.state) *
                    KroneckerDelta(q.state, Index('a', above_fermi=True)))
    elif isinstance(p, Fd) and isinstance(q, F):
        if q.state.assumptions0.get("above_fermi") or \
                p.state.assumptions0.get("above_fermi"):
            return S.Zero
        elif q.state.assumptions0.get("below_fermi") or \
                p.state.assumptions0.get("below_fermi"):
            return KroneckerDelta(p.state, q.state)
        else:
            return (KroneckerDelta(p.state, q.state) *
                    KroneckerDelta(q.state, Index('i', below_fermi=True)))
    else:  # vanish if 2xAnnihilator or 2xCreator
        return S.Zero
