from adcgen.operators import Operators
from adcgen.groundstate import GroundState
from adcgen.intermediate_states import IntermediateStates
from adcgen.expr_container import Expr
from adcgen.properties import Properties
from adcgen.reduce_expr import factor_eri_parts, factor_denom
from adcgen.simplify import simplify, remove_tensor
from adcgen.factor_intermediates import factor_intermediates
from adcgen.logger import logger

import itertools
import json
import argparse


class Generator:
    def __init__(self, names):
        mp_op = Operators(variant='mp')
        re_op = Operators(variant='re')
        mp = GroundState(mp_op, first_order_singles=False)
        mp_with_singles = GroundState(mp_op, first_order_singles=True)
        re = GroundState(re_op, first_order_singles=False)
        isr_pp = IntermediateStates(mp, variant='pp')
        prop_pp = Properties(isr_pp)
        self.op = {'mp': mp_op,
                   're': re_op}
        self.gs = {'mp': mp,
                   're': re,
                   'mp_with_singles': mp_with_singles}
        self.isr = {'pp': isr_pp}
        self.prop = {'pp': prop_pp}

        self.names = names

    def generate(self):
        generators = [m for m in dir(self)
                      if m.startswith('gen') and m != "generate"]

        if self.names:  # filter generators by name
            generators = [m for m in generators
                          if any(name in m for name in self.names)]
        for m in generators:  # run all remaining generators
            print(f"Running {m} ... ", end='', flush=True)
            getattr(self, m)()
            print("Done")

    def gen_operators(self):
        outfile = "operators.json"

        results: dict = {}
        # build h1 and h0
        for variant in ['mp', 're']:
            results[variant] = {}
            op = self.op[variant]
            results[variant]['h0'] = str(
                Expr(op.h0[0]).substitute_contracted()
            )
            results[variant]['h1'] = str(
                Expr(op.h1[0]).substitute_contracted()
            )
        # general operators
        for n_create in range(1, 4):
            for n_annihilate in range(1, 4):
                op = self.op["mp"]
                res = Expr(op.operator(n_create, n_annihilate)[0])
                res.substitute_contracted()
                results[f"{n_create}_{n_annihilate}"] = str(res)
        write_json(results, outfile)

    def gen_gs_energy(self):
        outfile = "gs_energy.json"

        results: dict = {}
        for variant in ['mp', 're']:
            results[variant] = {}
            gs = self.gs[variant]
            for order in [0, 1, 2]:
                results[variant][order] = str(
                    Expr(gs.energy(order)).substitute_contracted()
                )
        write_json(results, outfile)

    def gen_gs_psi(self):
        outfile = "gs_psi.json"

        results: dict = {}
        for order in [0, 1, 2]:
            results[order] = {}
            for braket in ['bra', 'ket']:
                if order == 1:
                    results[order][braket] = {}
                    # first order with singles
                    mp = self.gs['mp_with_singles']
                    results[order][braket]['with_singles'] = str(
                        Expr(mp.psi(order, braket)).substitute_contracted()
                    )
                    # first order without singles
                    mp = self.gs['mp']
                    results[order][braket]['no_singles'] = str(
                        Expr(mp.psi(order, braket)).substitute_contracted()
                    )
                else:
                    mp = self.gs['mp']
                    results[order][braket] = str(
                        Expr(mp.psi(order, braket)).substitute_contracted()
                    )
        write_json(results, outfile)

    def gen_gs_expectation_value(self):
        outfile = "gs_expectation_value.json"

        results: dict = {}
        for variant in ["mp", "re"]:
            results[variant] = {}
            gs = self.gs["mp"]
            for n_particles in [1]:
                results[variant][n_particles] = {}
                for order in [0, 1, 2]:
                    results[variant][n_particles][order] = {}
                    dump: dict = results[variant][n_particles][order]

                    # complex non symmetric expec value
                    res = Expr(gs.expectation_value(
                        order=order, n_particles=n_particles
                    ))
                    res.substitute_contracted()
                    res = simplify(res)
                    dump["expectation_value"] = str(res)
                    # dump the real symmetric expec value
                    res.make_real()
                    res.set_sym_tensors(["d"])
                    res = simplify(res)
                    dump["real_symmetric_expectation_value"] = str(res)
                    # dump the real symmetric density matrix
                    dump["real_symmetric_dm"] = {}
                    density = remove_tensor(res, "d")
                    for block, dm_expr in density.items():
                        assert len(block) == 1
                        block = block[0]
                        dump["real_symmetric_dm"][block] = str(dm_expr)
        write_json(results, outfile)

    def gen_gs_amplitude(self):

        def simplify_mp(ampl):
            res = 0
            for term in itertools.chain.from_iterable(
                            factor_denom(sub_expr)
                            for sub_expr in factor_eri_parts(ampl)):
                res += term.factor()
            return res

        outfile = "gs_amplitude.json"

        spaces = {1: [('ph', 'ia'), ('pphh', 'ijab')],
                  2: [('ph', 'ia'), ('pphh', 'ijab'), ('ppphhh', 'ijkabc'),
                      ('pppphhhh', 'ijklabcd')]}

        results = {}
        for variant in ['mp', 're']:
            results[variant] = {}
            for order in [1, 2]:
                results[variant][order] = {}
                for sp, idx in spaces[order]:
                    if variant == "re" and sp in ["ppphhh", "pppphhhh"]:
                        continue
                    ampl = self.gs[variant].amplitude(order, sp, idx)
                    # no einstein sum convention -> need to set target idx!
                    ampl = Expr(ampl, target_idx=idx).substitute_contracted()
                    if variant == 'mp':
                        ampl = simplify_mp(ampl)
                    else:
                        ampl = simplify(ampl)
                    results[variant][order][sp] = str(ampl)
        write_json(results, outfile)

    def gen_precursor(self):
        outfile = "isr_precursor.json"

        to_generate = {'pp': {('ph', 'ia'): [0, 1, 2]}}

        results = {}
        for variant, spaces in to_generate.items():
            results[variant] = {}
            isr = self.isr[variant]
            for (sp, indices), orders in spaces.items():
                results[variant][sp] = {}
                for o in orders:
                    results[variant][sp][o] = {}
                    bra = Expr(isr.precursor(o, sp, 'bra', indices))
                    ket = Expr(isr.precursor(o, sp, 'ket', indices))
                    results[variant][sp][o]['bra'] = str(bra)
                    results[variant][sp][o]['ket'] = str(ket)
        write_json(results, outfile)

    def gen_precursor_overlap(self):
        outfile = "isr_precursor_overlap.json"

        to_generate = {'pp': {('ph,ph', 'ia,jb'): [0, 1, 2]}}

        results = {}
        for variant, blocks in to_generate.items():
            results[variant] = {}
            isr = self.isr[variant]
            for (b, indices), orders in blocks.items():
                results[variant][b] = {}
                for o in orders:
                    res = Expr(isr.overlap_precursor(o, b, indices))
                    results[variant][b][o] = str(res)
        write_json(results, outfile)

    def gen_adc_properties_expectation_value(self):
        outfile = "properties_expectation_value.json"

        to_generate = {'pp': {1: [0, 1, 2]}}

        results = {}
        for adc_variant, operators in to_generate.items():
            results[adc_variant] = {}
            prop = self.prop[adc_variant]
            for n_particles, orders in operators.items():
                results[adc_variant][n_particles] = {}
                for adc_order in orders:
                    results[adc_variant][n_particles][adc_order] = {}
                    dump = results[adc_variant][n_particles][adc_order]
                    # dump the complex non symmetric result
                    res = Expr(prop.expectation_value(
                        adc_order=adc_order, n_particles=n_particles
                    ))
                    res.substitute_contracted()
                    res = simplify(res)
                    dump["expectation_value"] = str(res)
                    # dump the real result for a symmetric operator
                    # for a single state
                    res.make_real()
                    res.set_sym_tensors(["d"])
                    res.rename_tensor("X", "Y")
                    res = simplify(res)
                    dump["real_symmetric_state_expectation_value"] = str(res)
                    # dump the real symmetric density matrix
                    dump["real_symmetric_state_dm"] = {}
                    res = factor_intermediates(
                        res, ['t_amplitude', 'mp_density'], adc_order
                    )
                    density = remove_tensor(res, "d")
                    for block, expr in density.items():
                        assert len(block) == 1
                        block = block[0]
                        dump["real_symmetric_state_dm"][block] = str(expr)
        write_json(results, outfile)

    def gen_adc_properties_trans_moment(self):
        outfile = "properties_trans_moment.json"

        to_generate = {"pp": {(1, 1): [0, 1, 2]}}

        results = {}
        for adc_variant, operators in to_generate.items():
            results[adc_variant] = {}
            prop = self.prop[adc_variant]
            for (n_create, n_annihilate), orders in operators.items():
                op_key = f"{n_create}_{n_annihilate}"
                results[adc_variant][op_key] = {}
                for adc_order in orders:
                    results[adc_variant][op_key][adc_order] = {}
                    dump = results[adc_variant][op_key][adc_order]
                    # dump the complex non symmetric result
                    res = Expr(prop.trans_moment(
                        adc_order=adc_order, n_create=n_create,
                        n_annihilate=n_annihilate
                    ))
                    res.substitute_contracted()
                    res = simplify(res)
                    dump["expectation_value"] = str(res)
                    # dump the real result
                    # (operator should not be symmetric for transition dm)
                    res.make_real()
                    res = simplify(res)
                    dump["real_expectation_value"] = str(res)
                    # dump the real transition denstiy matrix
                    dump["real_transition_dm"] = {}
                    res = factor_intermediates(
                        res, ["t_amplitude", "mp_density"], adc_order
                    )
                    density = remove_tensor(res, "d")
                    for block, expr in density.items():
                        assert len(block) == 1
                        block = block[0]
                        dump["real_transition_dm"][block] = str(expr)
        write_json(results, outfile)


def write_json(data, filename):
    json.dump(data, open(filename, 'w'), indent=2)


def parse_cmdline():
    parser = argparse.ArgumentParser(
        prog='generate data',
        description='generates test data for consistency tests'
    )
    parser.add_argument('name', type=str, nargs='*',
                        help=(
                            'only generates data for tests that include'
                            'at least 1 of the provided strings in their name'
                        ))
    return parser.parse_args()


def main():
    parser = parse_cmdline()
    generator = Generator(parser.name)
    logger.setLevel("WARNING")
    generator.generate()


if __name__ == "__main__":
    main()
