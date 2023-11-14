from sympy_adc.operators import Operators
from sympy_adc.groundstate import GroundState
from sympy_adc.isr import IntermediateStates
from sympy_adc.expr_container import Expr
from sympy_adc.reduce_expr import factor_eri_parts, factor_denom
from sympy_adc.simplify import simplify

import itertools
import json
import argparse


class Generator:
    def __init__(self, names):
        # TODO: add argparse options to only generate a subset of data
        mp_op = Operators(variant='mp')
        re_op = Operators(variant='re')
        mp = GroundState(mp_op, first_order_singles=False)
        mp_with_singles = GroundState(mp_op, first_order_singles=True)
        re = GroundState(re_op, first_order_singles=False)
        isr_pp = IntermediateStates(mp, variant='pp')
        self.op = {'mp': mp_op,
                   're': re_op}
        self.gs = {'mp': mp,
                   're': re,
                   'mp_with_singles': mp_with_singles}
        self.isr = {'pp': isr_pp}

        self.names = names

    def generate(self):
        generators = [m for m in dir(self)
                      if m.startswith('gen') and m != "generate"]

        if self.names:  # filter generators by name
            generators = [m for m in generators
                          if any(name in m for name in self.names)]
        for m in generators:  # run all remaining generators
            print(f"Running {m} ... ", end='', flush=True)
            mute_and_run(getattr(self, m))
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
        for op_string in ['ca', 'ccaa', 'cccaaa']:
            op = self.op['mp']  # does not depend on the variant
            results[op_string] = str(
                Expr(op.operator(op_string)[0]).substitute_contracted()
            )
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

    def gen_amplitude(self):

        def simplify_mp(ampl):
            res = 0
            for term in itertools.chain.from_iterable(
                            factor_denom(sub_expr)
                            for sub_expr in factor_eri_parts(ampl)):
                res += term.factor()
            return res

        outfile = "amplitude.json"

        spaces = {1: [('ph', 'ia'), ('pphh', 'ijab')],
                  2: [('ph', 'ia'), ('pphh', 'ijab'), ('ppphhh', 'ijkabc'),
                      ('pppphhhh', 'ijklabcd')]}

        results = {}
        for variant in ['mp', 're']:
            results[variant] = {}
            for order in [1, 2]:
                if variant == 're' and order == 2:
                    # TODO: add them when I know how they should look like
                    continue
                results[variant][order] = {}
                for sp, idx in spaces[order]:
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


def mute_and_run(f):
    import io
    import sys
    sys.stdout = io.StringIO()
    f()
    sys.stdout = sys.__stdout__


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
    generator.generate()


if __name__ == "__main__":
    main()
