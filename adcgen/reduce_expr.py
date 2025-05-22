from collections.abc import Sequence
from collections import defaultdict
import itertools
import time

from sympy import S

from .eri_orbenergy import EriOrbenergy
from .expression import ExprContainer, TermContainer
from .indices import Index
from .logger import logger
from .symmetry import Permutation
from .tensor_names import tensor_names


def reduce_expr(expr: ExprContainer) -> ExprContainer:
    """
    Fully expands all available intermediates in an expression such that the
    expression only exists of orbital energies and electron repulsion
    integrals. The expanded expression is then simplified to collect as much
    terms as possible.
    The implementation assumes a real orbital basis.
    """
    assert isinstance(expr, ExprContainer)
    expr = expr.expand()

    # check if we have anything to do
    if expr.inner.is_number:
        return expr

    # add eri and fock matrix to the sym_tensors
    braket_sym_tensors = (tensor_names.fock, tensor_names.eri)

    logger.info("".join(
        ['\n', '#'*80, '\n', ' '*25, "REDUCING EXPRESSION\n", '#'*80, '\n']
    ))

    # 1) Insert the definitions of all defined intermediates in the expr
    #    and reduce the number of terms by factoring the ERI in each term.
    start = time.perf_counter()
    logger.info("Expanding intermediates... ")
    expanded_expr: list[ExprContainer] = []
    for term_i, term in enumerate(expr.terms):
        logger.info(
            "#"*80 + f"\nExpanding term {term_i+1} of {len(expr)}: {term}... ")
        term = term.expand_intermediates(
            wrap_result=True, fully_expand=True,
            braket_sym_tensors=braket_sym_tensors
        )
        assert isinstance(term, ExprContainer)
        term = term.expand().make_real()
        logger.info(f"into {len(term)} terms.\nCollecting terms.... ")
        term = factor_eri_parts(term)
        logger.info('-'*80)
        for j, equal_eri in enumerate(term):
            # minimize the contracted indices
            # each term in eri should hold exactly the same indices
            # -> build substitutions once and apply to the whole expr
            sub = equal_eri.terms[0].substitute_contracted(
                apply_substitutions=False
            )
            assert isinstance(sub, list)
            sub_equal_eri = equal_eri.subs(sub)
            # ensure that we are not creating a complete mess
            if sub_equal_eri.inner is S.Zero and equal_eri.inner is not S.Zero:
                raise ValueError(f"Invalid substitutions {sub} for "
                                 f"{equal_eri}")
            term[j] = sub_equal_eri
            logger.info(f"\n{j+1}: {EriOrbenergy(sub_equal_eri.terms[0]).eri}")
        logger.info("-"*80 + f"\nFound {len(term)} different ERI Structures")
        expanded_expr.extend(term)
    del expr
    # 2) Now try to factor the whole expression
    #    Only necessary to consider the first term of each of the expressions
    #    in the list (they all have same ERI)
    #    -> build new term list and try to factor ERI + Denominator
    #    -> simplify the orbital energy fraction in the resulting terms
    logger.info("\nExpanding and ERI factoring took "
                f"{time.perf_counter() - start:.2f}s\n")
    logger.info("".join(['#'*80, "\n", '#'*80]))
    start = time.perf_counter()
    logger.info("\nSumming up all terms...\n" + "#"*80)
    unique_terms = [unique_expr.terms[0] for unique_expr in expanded_expr]
    logger.info("Factoring ERI...")
    unique_compatible_eri = find_compatible_eri_parts(unique_terms)
    n = 1
    n_eri_denom = 0
    factored = 0
    # - factor eri again
    for i, compatible_eri_subs in unique_compatible_eri.items():
        temp = expanded_expr[i]
        eri = EriOrbenergy(expanded_expr[i].terms[0]).eri
        logger.info("\n" + "#"*80)
        logger.info(f"ERI {n} of {len(unique_compatible_eri)}: {eri}")
        n += 1
        for other_i, sub in compatible_eri_subs.items():
            temp += expanded_expr[other_i].subs(sub)

        # collected all terms with equal ERI -> factor denominators
        eri_sym = eri.symmetry(only_contracted=True)
        logger.info("\nFactoring Denominators...")
        for j, term in enumerate(factor_denom(temp, eri_sym=eri_sym)):
            term = term.factor()
            if len(term) != 1:
                raise RuntimeError("Expected the sub expression to have "
                                   "identical Denoms and ERI, which should "
                                   "allow factorization to a single term:\n"
                                   f"{term}")
            # symmetrize the numerator and cancel the orbital energy fraction
            term = EriOrbenergy(term)
            logger.info("-"*80 + f"\nERI/Denom {j}: {term}\n")
            logger.info("Permuting numerator... ")
            term = term.permute_num(eri_sym=eri_sym)
            logger.info(f"Term now reads:\n{term}\n")
            logger.info("Cancel orbital energy fraction...")
            term = term.cancel_orb_energy_frac()
            logger.info("Done.")

            if not all(EriOrbenergy(t).num.inner.is_number
                       for t in term.terms):
                logger.warning("\nNUMERATOR NOT CANCELLED COMPLETELY:")
                for t in term.terms:
                    logger.warning(EriOrbenergy(t))

            factored += term
            n_eri_denom += 1
    del expanded_expr  # not up to date anymore
    assert isinstance(factored, ExprContainer)
    logger.info("#"*80 +
                "\n\nFactorizing and cancelling the orbital energy fractions "
                f"in {n_eri_denom} terms took "
                f"{time.perf_counter() - start:.2f}s.\n"
                f"Expression consists now of {len(factored)} terms.")

    # 3) Since we modified some denominators by canceling the orbital energy
    #    fractions, try to factor eri and denominator again
    logger.info("#"*80 + "\n\nFactoring again...")
    result = 0
    for term in itertools.chain.from_iterable(
            factor_denom(sub_expr) for sub_expr in factor_eri_parts(factored)
            ):
        # factor the resulting term again, because we can have something like
        # 2/(4*a + 4*b) * X - 1/(2 * (a + b)) * X
        result += term.factor()
    assert isinstance(result, ExprContainer)
    logger.info(f"Done. {len(result)} terms remaining.\n\n" + "#"*80)
    return result


def factor_eri_parts(expr: ExprContainer) -> list[ExprContainer]:
    """
    Finds compatible remainder (eri) parts of an expression and collects
    the terms in subexpressions.

    Returns
    list[ExprContainer]
        List of subexpressions, where each subexpression contains terms with
        equal eri parts.
    """

    if len(expr) == 1:  # trivial case
        return [expr]

    terms = expr.terms
    ret: list[ExprContainer] = []
    for i, compatible_eri_subs in find_compatible_eri_parts(terms).items():
        temp = ExprContainer(terms[i].inner, **expr.assumptions)
        for other_i, sub in compatible_eri_subs.items():
            temp += terms[other_i].subs(sub)
        ret.append(temp)
    return ret


def find_compatible_eri_parts(
        term_list: Sequence[TermContainer]
        ) -> dict[int, dict[int, list[tuple[Index, Index]]]]:
    """
    Determines the necessary index substitutions to make the remainder (eri)
    parts of terms equal to each other - so they can be factored easily.
    Does not modify the terms, but returns a dict that connects the index of
    the terms with a substitution list.
    """
    from .simplify import find_compatible_terms

    if len(term_list) == 1:  # trivial: only a single eri
        return {0: {}}

    # dont use EriOrbenergy class, but rather only do whats necessary to
    # extract the eri part of the terms
    eri_parts: list[TermContainer] = []
    for term in term_list:
        assumptions = term.assumptions
        assumptions["target_idx"] = term.target
        eris = ExprContainer(1, **assumptions)
        for o in term.objects:
            if not o.inner.is_number and not o.contains_only_orb_energies:
                eris *= o
        assert len(eris) == 1
        eri_parts.append(eris.terms[0])
    return find_compatible_terms(eri_parts)


def factor_denom(expr: ExprContainer,
                 eri_sym: dict[tuple[Permutation, ...], int] | None = None
                 ) -> list[ExprContainer]:
    """
    Finds compatible orbital energy denominators in an expression with the
    restriction that the necessary index permutations do not modify the
    remainder (eri) part of the terms.

    Parameters
    ----------
    expr : ExprContainer
        Expression to find compatible denominators in.
    eri_sym : dict, optional
        The symmetry of the eri part of the terms. Warning: if provided, all
        terms in the expression are assumed to have the same eri symmetry!

    Returns
    list[ExprContainer]
        List of subexpressions, where each subexpression contains terms with
        equal orbital energy denominators.
    """

    if len(expr) == 1:  # trivial case: single term
        return [expr]

    terms: tuple[TermContainer, ...] = expr.terms
    compatible_denoms = find_compatible_denom(terms, eri_sym=eri_sym)
    ret: list[ExprContainer] = []
    for i, compatible_denom_perms in compatible_denoms.items():
        temp = ExprContainer(terms[i].inner, **expr.assumptions)
        for other_i, perms in compatible_denom_perms.items():
            temp += terms[other_i].permute(*perms)
        ret.append(temp)
    return ret


def find_compatible_denom(
        terms: Sequence[TermContainer],
        eri_sym: dict[tuple[Permutation, ...], int] | None = None
        ) -> dict[int, dict[int, tuple[Permutation, ...]]]:
    """
    Determines the necessary index substitutions to make the orbital energy
    denominators of the terms equal to each other - so they can be factored
    easily. Only permutations that do not change the remainder (eri) part of
    the terms are considered.
    Does not modify the terms but returns a dict that connects the index of
    the terms with a substitution list.

    Parameters
    ----------
    terms : Sequence[TermContainer]
        List of terms to find compatible orbital energy denominators.
    eri_sym : dict, optional
        The symmetry of the eri part of the terms. Warning: if provided, all
        terms are assumed to have the same eri symmetry!
    """
    if len(terms) == 1:  # trivial case: single term
        return {0: {}}

    terms_imported: list[EriOrbenergy] = [
        EriOrbenergy(term).canonicalize_sign(only_denom=True)
        for term in terms
    ]

    # split the terms according to length and and number of denominator
    # brackets
    filtered_terms = defaultdict(list)
    for term_i, term in enumerate(terms_imported):
        filtered_terms[term.denom_description()].append(term_i)

    ret: dict[int, dict[int, tuple[Permutation, ...]]] = {}
    matched: set[int] = set()
    permutations: dict[int, tuple[tuple[Permutation, ...], ...]] = {}
    for term_idx_list in filtered_terms.values():
        # check which denominators are already equal
        identical_denom: dict[int, list[int]] = {}
        for i, term_i in enumerate(term_idx_list):
            if term_i in matched:
                continue
            term: EriOrbenergy = terms_imported[term_i]
            identical_denom[term_i] = []
            for other_i in range(i+1, len(term_idx_list)):
                other_term_i = term_idx_list[other_i]
                if other_term_i in matched:
                    continue
                other_term = terms_imported[other_term_i]
                if term.denom.inner == other_term.denom.inner:
                    identical_denom[term_i].append(other_term_i)
                    matched.add(other_term_i)

        if len(identical_denom) == 1:  # all denoms are equal
            term_i, matches = identical_denom.popitem()
            ret[term_i] = {other_term_i: tuple() for other_term_i in matches}
            continue

        # try to match more denominators by applying index permutations that
        # satisfy:  P_pq ERI = +- ERI  AND  P_pq Denom != +- Denom
        identical_denom_list = list(identical_denom.items())
        del identical_denom
        for i, (term_i, matches) in enumerate(identical_denom_list):
            if term_i in matched:
                continue
            ret[term_i] = {}
            for other_term_i in matches:  # add all identical denominators
                ret[term_i][other_term_i] = tuple()

            denom = terms_imported[term_i].denom.inner
            for other_i in range(i+1, len(identical_denom_list)):
                other_term_i, other_matches = identical_denom_list[other_i]
                if other_term_i in matched:
                    continue

                other_term: EriOrbenergy = terms_imported[other_term_i]
                other_denom: ExprContainer = other_term.denom

                # find all valid permutations
                if other_term_i not in permutations:
                    permutations[other_term_i] = tuple(
                        perms for perms, factor in
                        other_term.denom_eri_sym(eri_sym=eri_sym,
                                                 only_contracted=True).items()
                        if factor is None
                    )
                for perms in permutations[other_term_i]:
                    # found a permutation!
                    if denom == other_denom.copy().permute(*perms).inner:
                        ret[term_i][other_term_i] = perms
                        for match in other_matches:
                            ret[term_i][match] = perms
                        matched.add(other_term_i)
                        break
    return ret
