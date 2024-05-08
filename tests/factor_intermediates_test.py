from adcgen.factor_intermediates import factor_intermediates
from adcgen.func import import_from_sympy_latex
from adcgen.simplify import simplify

from sympy import S


class TestFactorIntermediates:
    def test_factor_t2_1(self):
        # just factor it once
        eri = import_from_sympy_latex("{V^{ij}_{ab}}").make_real()
        denom = import_from_sympy_latex(
            "{e_{a}} + {e_{b}} - {e_{i}} - {e_{j}}"
        ).make_real()
        t = import_from_sympy_latex("{t1^{ab}_{ij}}").make_real()
        remainder = import_from_sympy_latex("{X_{ij}^{ab}}").make_real()

        test = eri / denom * remainder / 2
        test.set_target_idx([])
        res = factor_intermediates(test, types_or_names='t2_1')
        ref = t * remainder / 2
        assert (res.sympy - ref.sympy) is S.Zero

        # factor it twice
        eri2 = import_from_sympy_latex("{V^{jk}_{bc}}").make_real()
        denom2 = import_from_sympy_latex(
            "{e_{b}} + {e_{c}} - {e_{j}} - {e_{k}}"
        ).make_real()
        t2 = import_from_sympy_latex("{t1^{bc}_{jk}}").make_real()

        test = eri * eri2 / (denom * denom2).expand() * remainder * 2
        test.set_target_idx('jk')
        res = factor_intermediates(test, types_or_names='t2_1')
        ref = t * t2 * 2 * remainder
        assert (res.sympy - ref.sympy) is S.Zero

        # factor with exponent > 1
        test = eri * eri * eri2 / (denom * denom * denom2).expand() * \
            remainder * 2
        test.set_target_idx('jk')
        res = factor_intermediates(test, types_or_names='t2_1')
        ref = t * t * t2 * remainder * 2
        assert (res.sympy - ref.sympy) is S.Zero

    def test_long_intermediate_complete(self):
        # rather easy, but also fast example: eri * t2_2 amplitude
        # - no intersection of the indices -> t2_2 consists of 6 terms
        test = import_from_sympy_latex(
            r"- \frac{{V^{cd}_{ef}} {V^{ij}_{ab}} {V^{kl}_{ef}}}{2 \left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{e}} + {e_{f}} - {e_{k}} - {e_{l}}\right)} "  # noqa E501
            r"+ \frac{{V^{ij}_{ab}} {V^{ke}_{mc}} {V^{lm}_{de}}}{\left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{d}} + {e_{e}} - {e_{l}} - {e_{m}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{km}_{de}} {V^{le}_{mc}}}{\left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{d}} + {e_{e}} - {e_{k}} - {e_{m}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{ke}_{md}} {V^{lm}_{ce}}}{\left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{c}} + {e_{e}} - {e_{l}} - {e_{m}}\right)} "  # noqa E501
            r"+ \frac{{V^{ij}_{ab}} {V^{km}_{ce}} {V^{le}_{md}}}{\left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{c}} + {e_{e}} - {e_{k}} - {e_{m}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{kl}_{mn}} {V^{mn}_{cd}}}{2 \left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right) \left({e_{c}} + {e_{d}} - {e_{m}} - {e_{n}}\right)}"  # noqa E501
        ).make_real()
        test.set_target_idx('ijklabcd')

        # ensure that the same result is obtained indepent of the
        # factored intermediates
        res = factor_intermediates(test, types_or_names='t2_2')
        res2 = factor_intermediates(test, types_or_names=['t2_1', 't2_2'])
        res3 = factor_intermediates(test,
                                    types_or_names=['t2_1', 't1_2', 't2_2'])
        assert simplify(res - res2).sympy is S.Zero
        assert simplify(res - res3).sympy is S.Zero

        ref = import_from_sympy_latex("{V^{ab}_{ij}} {t2^{cd}_{kl}}")
        ref.make_real()
        ref.set_target_idx('ijklabcd')
        assert simplify(res - ref).sympy is S.Zero

        # - occupied indices intersect -> t2_2 consists of 4 terms
        test = import_from_sympy_latex(
            r"- \frac{{V^{cd}_{ef}} {V^{ij}_{ab}} {V^{ij}_{ef}}}{2 \left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right) \left({e_{e}} + {e_{f}} - {e_{i}} - {e_{j}}\right)} "  # noqa E501
            r"+ \frac{2 {V^{ie}_{kc}} {V^{ij}_{ab}} {V^{jk}_{de}}}{\left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right) \left({e_{d}} + {e_{e}} - {e_{j}} - {e_{k}}\right)} "  # noqa E501
            r"+ \frac{2 {V^{ij}_{ab}} {V^{ik}_{ce}} {V^{je}_{kd}}}{\left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right) \left({e_{c}} + {e_{e}} - {e_{i}} - {e_{k}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{ij}_{kl}} {V^{kl}_{cd}}}{2 \left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right) \left({e_{c}} + {e_{d}} - {e_{k}} - {e_{l}}\right)}"  # noqa E501
        ).make_real()
        test.set_target_idx('abcd')

        ref = import_from_sympy_latex("{V^{ab}_{ij}} {t2^{cd}_{ij}}")
        ref.make_real()
        ref.set_target_idx('abcd')

        res = factor_intermediates(test, types_or_names='t2_2')
        assert simplify(res - ref).sympy is S.Zero

        # - occ and virt intersect -> t2_2 consists of 3 terms
        test = import_from_sympy_latex(
            r"- \frac{{V^{ab}_{cd}} {V^{ij}_{ab}} {V^{ij}_{cd}}}{2 \left({e_{a}} + {e_{b}} - {e_{i}} - {e_{j}}\right) \left({e_{c}} + {e_{d}} - {e_{i}} - {e_{j}}\right)} "  # noqa E501
            r"+ \frac{4 {V^{ij}_{ab}} {V^{ik}_{ac}} {V^{jc}_{kb}}}{\left({e_{a}} + {e_{b}} - {e_{i}} - {e_{j}}\right) \left({e_{a}} + {e_{c}} - {e_{i}} - {e_{k}}\right)} "  # noqa E501
            r"- \frac{{V^{ij}_{ab}} {V^{ij}_{kl}} {V^{kl}_{ab}}}{2 \left({e_{a}} + {e_{b}} - {e_{i}} - {e_{j}}\right) \left({e_{a}} + {e_{b}} - {e_{k}} - {e_{l}}\right)}"   # noqa E501
        ).make_real()
        test.set_target_idx([])

        ref = import_from_sympy_latex("{V^{ab}_{ij}} {t2^{ab}_{ij}}")
        ref.make_real()
        ref.set_target_idx([])

        res = factor_intermediates(test, types_or_names='t2_2')
        assert simplify(res - ref).sympy is S.Zero

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

        test = import_from_sympy_latex(test).make_real()
        test.set_target_idx('ijab')

        res = factor_intermediates(test, types_or_names='re_residual')

        ref = r"- \frac{\delta_{i j} {f^{b}_{d}} {t1^{ac}_{kl}} {t1^{cd}_{kl}}}{4}"  # noqa E501
        ref = import_from_sympy_latex(ref).make_real()

        assert (res.sympy - ref.sympy) is S.Zero
