from sympy import sqrt
from sympy.physics.secondquant import wicks
from math import factorial

from isr import get_orders_three
from indices import get_n_ov_from_space
from misc import Inputerror, cached_member, transform_to_tuple
from secular_matrix import secular_matrix


class properties:
    def __init__(self, isr):
        self.isr = isr
        self.gs = isr.gs
        self.h = isr.gs.h
        self.indices = isr.gs.indices
        self.variant = isr.variant

    def __shifted_one_particle_op(self, order):
        d = self.h.one_particle if order == 0 else 0
        return d - self.gs.one_particle_operator(order)

    def __shifted_two_particle_op(self, order):
        d = self.h.two_particle if order == 0 else 0
        return d - self.gs.two_particle_operator(order)

    @cached_member
    def one_particle_block(self, order, block, indices):
        """Computes sum_pq d_{pq} <I|pq|J>^(n).
           Results checked for ADC(2)!
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        if len(block) != 2 or len(indices) != 2:
            raise Inputerror("2 space and index strings required."
                             f"Provided: {block} / {indices}.")
        sp_idx = {"bra": (block[0], indices[0]),
                  "ket": (block[1], indices[1])}
        isr = {}
        for bk in ["bra", "ket"]:
            isr[bk] = {}
            sp = sp_idx[bk][0]
            idx = sp_idx[bk][1]

            # generate indices for the isr states.
            n_ov = get_n_ov_from_space(sp)
            sym_pre = self.indices.get_isr_indices(idx, **n_ov)
            idx_pre = []
            for s_list in sym_pre.values():
                idx_pre.extend([s.name for s in s_list])
            idx_pre = "".join(idx_pre)

            for o in range(order + 1):
                isr[bk][o] = self.isr.intermediate_state(
                    o, sp, bk, idx_is=idx, idx_pre=idx_pre
                )

        left = self.isr.amplitude_vector(indices[0], "left")
        right = self.isr.amplitude_vector(indices[1], "right")
        # again not use the full prefactors from lifting the sum restrictions
        # but sqrt(1/(no! * nv!)) to keep the left and right amplitude vectors
        # normalized.
        n_ov = get_n_ov_from_space(block[0])
        prefactor_l = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )
        n_ov = get_n_ov_from_space(block[1])
        prefactor_r = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        orders = get_orders_three(order)
        res = 0
        for term in orders:
            i1 = (prefactor_l * prefactor_r * left * isr["bra"][term[0]] *
                  self.__shifted_one_particle_op(term[1]) *
                  isr["ket"][term[2]] * right)
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        return res

    @cached_member
    def one_particle_dm(self, order, adc_order):
        m = secular_matrix(self.isr)
        blocks = m.block_order(adc_order)
        res = 0
        for b, max_order in blocks.items():
            if max_order >= order:
                n_ov = get_n_ov_from_space(b[0])
                sym = self.indices.get_new_gen_indices(**n_ov)
                idx0 = []
                for s_list in sym.values():
                    idx0.extend([s.name for s in s_list])
                n_ov = get_n_ov_from_space(b[1])
                sym = self.indices.get_new_gen_indices(**n_ov)
                idx1 = []
                for s_list in sym.values():
                    idx1.extend([s.name for s in s_list])
                idx0 = "".join(idx0)
                idx1 = "".join(idx1)
                res += self.opdm_block(order, b, idx0 + "," + idx1)
        return NotImplementedError()

    @cached_member
    def two_particle_block(self, order, block, indices):
        """Computes 1/4 sum_pqrs d_{pqrs} <I|pqsr|J>^(n).
           Did not check any results yet!
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        if len(block) != 2 or len(indices) != 2:
            raise Inputerror("2 space and index strings required."
                             f"Provided: {block} / {indices}.")

        sp_idx = {"bra": (block[0], indices[0]),
                  "ket": (block[1], indices[1])}
        isr = {}
        for bk in ["bra", "ket"]:
            isr[bk] = {}
            sp = sp_idx[bk][0]
            idx = sp_idx[bk][1]

            # generate indices for the precursor states of the ISR states
            n_ov = get_n_ov_from_space(sp)
            sym_pre = self.indices.get_isr_indices(idx, **n_ov)
            idx_pre = []
            for s_list in sym_pre.values():
                idx_pre.extend([s.name for s in s_list])
            idx_pre = "".join(idx_pre)

            for o in range(order + 1):
                isr[bk][o] = self.isr.intermediate_state(
                    o, sp, bk, idx_is=idx, idx_pre=idx_pre
                )

        left = self.isr.amplitude_vector(indices[0], "left")
        right = self.isr.amplitude_vector(indices[1], "right")
        # again not use the full prefactors from lifting the sum restrictions
        # but sqrt(1/(no! * nv!)) to keep the left and right amplitude vectors
        # normalized.
        n_ov = get_n_ov_from_space(block[0])
        prefactor_l = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )
        n_ov = get_n_ov_from_space(block[1])
        prefactor_r = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        orders = get_orders_three(order)
        res = 0
        # what about prefactors here??
        for term in orders:
            i1 = (prefactor_l * prefactor_r * left * isr["bra"][term[0]] *
                  self.__shifted_two_particle_op(term[1]) *
                  isr["ket"][term[2]] * right)
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        return res
