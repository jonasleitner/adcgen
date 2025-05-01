from adcgen.expression import ExprContainer
from adcgen.factor_intermediates import factor_intermediates
from adcgen.func import import_from_sympy_latex
from adcgen.intermediates import t2eri_2, p0_3_oo
from adcgen.simplify import simplify

from sympy import Add, S


class TestFactorIntermediates:
    def test_factor_t2_1(self):
        # just factor it once
        eri = import_from_sympy_latex(
            "{V^{ij}_{ab}}", convert_default_names=True
        ).make_real()
        denom = "{e_{a}} + {e_{b}} - {e_{i}} - {e_{j}}"
        denom = import_from_sympy_latex(denom, convert_default_names=True)
        denom.make_real()
        t = import_from_sympy_latex(
            "{t1^{ab}_{ij}}", convert_default_names=True
        ).make_real()
        remainder = import_from_sympy_latex(
            "{X_{ij}^{ab}}", convert_default_names=True
        ).make_real()

        test = eri / denom * remainder / 2
        test.set_target_idx([])
        res = factor_intermediates(test, types_or_names='t2_1')
        ref = t * remainder / 2
        assert Add(res.inner, -ref.inner) is S.Zero

        # factor it twice
        eri2 = import_from_sympy_latex(
            "{V^{jk}_{bc}}", convert_default_names=True
        ).make_real()
        denom2 = "{e_{b}} + {e_{c}} - {e_{j}} - {e_{k}}"
        denom2 = import_from_sympy_latex(denom2, convert_default_names=True)
        denom2.make_real()
        t2 = import_from_sympy_latex(
            "{t1^{bc}_{jk}}", convert_default_names=True
        ).make_real()

        test = eri * eri2 / (denom * denom2).expand() * remainder * 2
        test.set_target_idx('jk')
        res = factor_intermediates(test, types_or_names='t2_1')
        ref = t * t2 * 2 * remainder
        assert Add(res.inner, -ref.inner) is S.Zero

        # factor with exponent > 1
        test = eri * eri * eri2 / (denom * denom * denom2).expand() * \
            remainder * 2
        test.set_target_idx('jk')
        res = factor_intermediates(test, types_or_names='t2_1')
        ref = t * t * t2 * remainder * 2
        assert Add(res.inner, -ref.inner) is S.Zero

    def test_long_intermediate_complete(self):
        # rather easy, but also fast example: eri * t2_2 amplitude
        # - no intersection of the indices -> t2_2 consists of 6 terms
        test = (
            r"- \frac{{V^{cd}_{ef}} {V^{ij}_{ab}} {V^{kl}_{ef}}}{2 \left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{e}} + {e_{f}} - {e_{k}} - {e_{l}}\right)} "  # noqa E501
            r"+ \frac{{V^{ij}_{ab}} {V^{ke}_{mc}} {V^{lm}_{de}}}{\left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{d}} + {e_{e}} - {e_{l}} - {e_{m}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{km}_{de}} {V^{le}_{mc}}}{\left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{d}} + {e_{e}} - {e_{k}} - {e_{m}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{ke}_{md}} {V^{lm}_{ce}}}{\left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{c}} + {e_{e}} - {e_{l}} - {e_{m}}\right)} "  # noqa E501
            r"+ \frac{{V^{ij}_{ab}} {V^{km}_{ce}} {V^{le}_{md}}}{\left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{c}} + {e_{e}} - {e_{k}} - {e_{m}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{kl}_{mn}} {V^{mn}_{cd}}}{2 \left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{c}} + {e_{d}} - {e_{m}} - {e_{n}}\right)}"  # noqa E501
        )
        test = import_from_sympy_latex(test, convert_default_names=True)
        test.make_real()
        test.set_target_idx('ijklabcd')

        # ensure that the same result is obtained indepent of the
        # factored intermediates
        res = factor_intermediates(test, types_or_names='t2_2')
        res2 = factor_intermediates(test, types_or_names=['t2_1', 't2_2'])
        res3 = factor_intermediates(test,
                                    types_or_names=['t2_1', 't1_2', 't2_2'])
        assert simplify(res - res2).inner is S.Zero
        assert simplify(res - res3).inner is S.Zero

        ref = "{V^{ab}_{ij}} {t2^{cd}_{kl}}"
        ref = import_from_sympy_latex(ref, convert_default_names=True)
        ref.make_real()
        ref.set_target_idx('ijklabcd')
        assert simplify(res - ref).inner is S.Zero

        # - occupied indices intersect -> t2_2 consists of 4 terms
        test = (
            r"- \frac{{V^{cd}_{ef}} {V^{ij}_{ab}} {V^{ij}_{ef}}}{2 \left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right) \left({e_{e}} + {e_{f}} - {e_{i}} - {e_{j}}\right)} "  # noqa E501
            r"+ \frac{2 {V^{ie}_{kc}} {V^{ij}_{ab}} {V^{jk}_{de}}}{\left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right) \left({e_{d}} + {e_{e}} - {e_{j}} - {e_{k}}\right)} "  # noqa E501
            r"+ \frac{2 {V^{ij}_{ab}} {V^{ik}_{ce}} {V^{je}_{kd}}}{\left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right) \left({e_{c}} + {e_{e}} - {e_{i}} - {e_{k}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{ij}_{kl}} {V^{kl}_{cd}}}{2 \left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right) \left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right)}"  # noqa E501
        )
        test = import_from_sympy_latex(test, convert_default_names=True)
        test.make_real()
        test.set_target_idx('abcd')

        ref = "{V^{ab}_{ij}} {t2^{cd}_{ij}}"
        ref = import_from_sympy_latex(ref, convert_default_names=True)
        ref.make_real()
        ref.set_target_idx('abcd')

        res = factor_intermediates(test, types_or_names='t2_2')
        assert simplify(res - ref).inner is S.Zero

        # - occ and virt intersect -> t2_2 consists of 3 terms
        test = (
            r"- \frac{{V^{ab}_{cd}} {V^{ij}_{ab}} {V^{ij}_{cd}}}{2 \left({e_{a}} + {e_{b}} - {e_{i}} - {e_{j}}\right) \left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right)} "  # noqa E501
            r"+ \frac{4 {V^{ij}_{ab}} {V^{ik}_{ac}} {V^{jc}_{kb}}}{\left({e_{a}} + {e_{b}} - {e_{i}} - {e_{j}}\right) \left({e_{a}} + {e_{c}} - {e_{i}} - {e_{k}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{ij}_{kl}} {V^{kl}_{ab}}}{2 \left({e_{a}} + {e_{b}} - {e_{i}} - {e_{j}}\right) \left({e_{a}} + {e_{b}} - {e_{k}} - {e_{l}}\right)}"   # noqa E501
        )
        test = import_from_sympy_latex(test, convert_default_names=True)
        test.make_real()
        test.set_target_idx([])

        ref = "{V^{ab}_{ij}} {t2^{ab}_{ij}}"
        ref = import_from_sympy_latex(ref, convert_default_names=True)
        ref.make_real()
        ref.set_target_idx([])

        res = factor_intermediates(test, types_or_names='t2_2')
        assert simplify(res - ref).inner is S.Zero

    def test_long_intermediate_mixed_prefs(self):
        # test expression to factor the re_t2_1_residual
        test = (
            r"\frac{\delta_{i j} {V^{kl}_{bc}} {t1^{ac}_{kl}}}{2}"
            + r" - \delta_{i j} {V^{kd}_{lb}} {t1^{ac}_{km}} {t1^{cd}_{lm}}"
            + r" - \delta_{i j} {V^{kd}_{mc}} {t1^{ac}_{kl}} {t1^{bd}_{lm}}"
            + r" - \delta_{i j} {f^{k}_{m}} {t1^{ac}_{kl}} {t1^{bc}_{lm}}"
            + r" - \frac{\delta_{i j} {f^{c}_{d}} {t1^{ac}_{kl}} {t1^{bd}_{kl}}}{2}"  # noqa E501
            + r" - \frac{\delta_{i j} {V^{bc}_{de}} {t1^{ac}_{kl}} {t1^{de}_{kl}}}{4}"  # noqa E501
            + r" + \frac{\delta_{i j} {f^{b}_{d}} {t1^{ac}_{kl}} {t1^{cd}_{kl}}}{4}"  # noqa E501
            + r" - \frac{\delta_{i j} {V^{kl}_{mn}} {t1^{ac}_{kl}} {t1^{bc}_{mn}}}{4}"  # noqa E501
        )

        test = import_from_sympy_latex(test, convert_default_names=True)
        test.make_real()
        test.set_target_idx('ijab')

        res = factor_intermediates(test, types_or_names='t2_1_re_residual')

        ref = r"- \frac{\delta_{i j} {f^{b}_{d}} {t1^{ac}_{kl}} {t1^{cd}_{kl}}}{4}"  # noqa E501
        ref = import_from_sympy_latex(ref, convert_default_names=True)
        ref.make_real()

        assert Add(res.inner, -ref.inner) is S.Zero

    def test_repeated_indices(self):
        # Short intermediate
        pi2 = t2eri_2()
        # check that factorization works at all
        test = pi2.expand_itmd("ijka", fully_expand=False)
        assert isinstance(test, ExprContainer)
        test.make_real()
        res = factor_intermediates(test, ["t2_1", "t2eri_2"])
        ref = pi2.tensor(indices="ijka")
        assert isinstance(ref, ExprContainer)
        assert Add(res.inner, -ref.inner) is S.Zero
        # allow repeated indices
        test = pi2.expand_itmd("ijia", fully_expand=False)
        assert isinstance(test, ExprContainer)
        test.make_real()
        res = factor_intermediates(test, ["t2_1", "t2eri_2"],
                                   allow_repeated_itmd_indices=True)
        ref = pi2.tensor(indices="ijia")
        assert isinstance(ref, ExprContainer)
        assert Add(res.inner, -ref.inner) is S.Zero
        # don't allow repeated indices
        res = factor_intermediates(test, ["t2_1", "t2eri_2"],
                                   allow_repeated_itmd_indices=False)
        assert simplify(res - test).inner is S.Zero
        # Long intermediate
        # not working, because the number of terms of e.g., p0_3_oo is reduced
        # from 2 to 1.
        p0_oo = p0_3_oo()
        # check that factorization works at all
        test = p0_oo.expand_itmd("ij", fully_expand=False)
        assert isinstance(test, ExprContainer)
        test.make_real()
        res = factor_intermediates(test, ["t2_1", "t2_2", "p0_3_oo"])
        ref = p0_oo.tensor(indices="ij")
        assert isinstance(ref, ExprContainer)
        assert Add(res.inner, -ref.inner) is S.Zero
        # allow repeated indices -> does not work anyway
        test = p0_oo.expand_itmd("ii", fully_expand=False)
        assert isinstance(test, ExprContainer)
        test.make_real()
        res = factor_intermediates(test, ["t2_1", "t2_2", "p0_3_oo"],
                                   allow_repeated_itmd_indices=True)
        # don't allow repeated indices
        assert simplify(res - test).inner is S.Zero
        res = factor_intermediates(test, ["t2_1", "t2_2", "p0_3_oo"],
                                   allow_repeated_itmd_indices=False)
        assert simplify(res - test).inner is S.Zero
