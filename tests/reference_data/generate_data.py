from sympy_adc.operators import Operators
from sympy_adc.groundstate import GroundState
from sympy_adc.expr_container import Expr

import json


class Generator:
    def __init__(self):
        # TODO: add argparse options to only generate a subset of data
        mp_op = Operators(variant='mp')
        re_op = Operators(variant='re')
        mp = GroundState(mp_op, first_order_singles=False)
        re = GroundState(re_op, first_order_singles=False)
        self.op = {'mp': mp_op,
                   're': re_op}
        self.gs = {'mp': mp,
                   're': re}

    def gen_all(self):
        self.operators()
        self.gs_energy()

    def operators(self):
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

    def gs_energy(self):
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


def write_json(data, filename):
    json.dump(data, open(filename, 'w'), indent=2)


def main():
    generator = Generator()
    generator.gen_all()


if __name__ == "__main__":
    main()
