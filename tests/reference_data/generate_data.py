from sympy_adc.operators import Operators
from sympy_adc.groundstate import GroundState
from sympy_adc.expr_container import Expr

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
        self.op = {'mp': mp_op,
                   're': re_op}
        self.gs = {'mp': mp,
                   're': re,
                   'mp_with_singles': mp_with_singles}

        self.names = names

    def generate(self):
        generators = [m for m in dir(self)
                      if m.startswith('gen') and m != "generate"]

        if self.names:  # filter generators by name
            generators = [m for m in generators
                          if any(name in m for name in self.names)]
        for m in generators:  # run all remaining generators
            print(f"Running {m} ... ", end='')
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
