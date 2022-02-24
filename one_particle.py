from sympy import Rational
from sympy.physics.secondquant import wicks
from math import factorial

from secular_matrix import secular_matrix
from isr import get_order_two
from indices import get_n_ov_from_space


class one_particle_operator:
    def __init__(self, isr):
        self.isr = isr
        self.gs = isr.gs
        self.h = isr.gs.h
        self.indices = isr.gs.indices
        self.variant = isr.variant

    def opdm_block(self, block, order):
        if type(block) == str:
            if len(block.split(",")) > 2:
                print("A opdm block may only consist of two spaces."
                      f"Provided block: {block}.")
                exit()
            block = (block.split(",")[0], block.split(",")[1])
        if not isinstance(block, tuple):
            print("Opdm block needs to be provided as string or tuple."
                  f"{type(block)} is not valid.")
            exit()
        if len(block) > 2:
            print("A opdm block may only consist of two spaces."
                  f"Provided block: {block}.")
            exit()

        spaces = {"ket": block[1], "bra": block[0]}
        isr = {}
        ampl = {}
        pref = {}
        for bk in ["bra", "ket"]:
            sp = spaces[bk]
            if bk not in isr:
                isr[bk] = {}
            if bk not in ampl:
                ampl[bk] = {}

            # generate indices for the ISR states and the ampl vector
            n_ov = get_n_ov_from_space(sp)
            sym_is = self.indices.get_new_gen_indices(**n_ov)
            idx_is = []
            for s_list in sym_is.values():
                idx_is.extend([s.name for s in s_list])
            sym_pre = self.indices.get_new_gen_indices(**n_ov)
            idx_pre = []
            for s_list in sym_pre.values():
                idx_pre.extend([s.name for s in s_list])
            idx_is = "".join(idx_is)
            idx_pre = "".join(idx_pre)

            # import the amplitude vector
            lr = {"ket": "right", "bra": "left"}
            ampl[bk] = self.isr.amplitude_vector(sp, idx_is, lr=lr[bk])

            # prefactor from lifting sum restrictions
            pref[bk] = Rational(
                1, factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
            )

            # import all ISR states up to the requested order
            for o in range(order + 1):
                isr[bk][o] = self.isr.intermediate_state(
                    o, sp, bk, idx_is=idx_is, idx_pre=idx_pre
                )

        orders = get_order_two(order)
        res = 0
        for term in orders:
            # TODO: Account for normalization!
            i1 = (pref["bra"] * pref["ket"] *
                  ampl["bra"] * isr["bra"][term[0]] * self.h.one_particle *
                  isr["ket"][term[1]] * ampl["ket"])
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        return res

    def opdm(self, order):
        m = secular_matrix(self.isr)
        block_orders = m.block_order(order)
        max_orders = m.get_max_ptorder_spaces(order)

        res = 0
        return res
