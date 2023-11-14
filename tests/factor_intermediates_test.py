from sympy_adc.factor_intermediates import factor_intermediates
from sympy_adc.func import import_from_sympy_latex

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
