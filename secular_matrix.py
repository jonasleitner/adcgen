from sympy.physics.secondquant import wicks, evaluate_deltas
from sympy import sqrt

from math import factorial

from isr import get_orders_three
from indices import (check_repeated_indices, split_idxstring,
                     get_n_ov_from_space)
from misc import cached_member


class secular_matrix:
    def __init__(self, isr):
        self.isr = isr
        self.gs = isr.gs
        self.h = isr.gs.h
        self.indices = isr.gs.indices

    def __get_shifted_h(self, order):
        get_H = {
            0: self.h.h0,
            1: self.h.h1,
        }
        h = get_H[order] if order < 2 else 0
        return h - self.gs.energy(order)

    @cached_member
    def precursor_matrix(self, order, block, indices):
        """Computes a certain block of the secular matrix in the
           basis of the precursor states.
           Substitute_dummies: 1) messes around with the indices -> very hard
           to identifie to which matrix element (e.g. M_ia,jb vs M_ic,kb)
           a term belongs. 2) for coupling blocks and the pphh,pphh block wrong
           index substitution causes all terms to cancel - which is wong.
           The custom substitute_indices function (method of the indice book
           keeping class) gives correct results for all blocks (only checked
           for ADC(2)). However, in some blocks it may be necessary to
           have a look at a few terms that did not cancel with
           substitute_indices. It may be necessary to rename/interchange some
           indices by hand for those terms to cancel correctly.
           """

        if len(block.split(",")) != 2 or len(indices.split(",")) != 2:
            print("Precursor matrix requires two block and indice strings that"
                  f"are separated by a ','. Block {block} and indice {indices}"
                  "are not valid.")
            exit()
        for space in block.split(","):
            if space not in self.isr.valid_spaces:
                print("Requested a matrix block for an unknown space."
                      f"Valid blocks are {list(self.isr.valid_spaces.keys())}")
                exit()

        if check_repeated_indices(indices.split(",")[0],
                                  indices.split(",")[1]):
            print("Indices for precursor secular matrix should not be equal."
                  f"Provided indice string: {indices}")
            exit()

        space_idx = {
            "bra": (block.split(",")[0], indices.split(",")[0]),
            "ket": (block.split(",")[1], indices.split(",")[1])
        }
        # import precursor states up to requested order
        pre = {}
        for o in range(order + 1):
            pre[o] = {}
            for bk in ["bra", "ket"]:
                sp = space_idx[bk][0]
                idx = space_idx[bk][1]
                pre[o][bk] = self.isr.precursor(o, sp, bk, indices=idx)

        orders = get_orders_three(order)
        res = 0
        for term in orders:
            i1 = pre[term[0]]["bra"] * self.__get_shifted_h(term[1]) * \
                 pre[term[2]]["ket"]
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        return res

    @cached_member
    def isr_matrix(self, order, block, indices):
        """Computes a specific block of a specific order of the secular matrix.
           Substitute_dummie: 1) messes around with the indices (see
           precursor_matrix) 2) for the coupling blocks and the 'pphh,pphh'
           block wrong index substitution causes all terms to cancel - which
           is wrong.
           The custom substitute_indices function may be used instead. It
           gives correct results for ADC(2). No additional manipulation of
           terms by hand was necessary.
           """

        if len(block.split(",")) != 2 or len(indices.split(",")) != 2:
            print("Precursor matrix requires two block and indice strings that"
                  f"are separated by a ','. Block {block} and indice {indices}"
                  "are not valid.")
            exit()
        for space in block.split(","):
            if space not in self.isr.valid_spaces:
                print("Requested a matrix block for an unknown space.",
                      f"Valid blocks are {list(self.isr.valid_spaces.keys())}")
                exit()

        if check_repeated_indices(indices.split(",")[0],
                                  indices.split(",")[1]):
            print("Indices for isr secular matrix should not be equal.",
                  f"Provided indice string: {indices}")
            exit()

        space_idx = {
            "bra": (block.split(",")[0], indices.split(",")[0]),
            "ket": (block.split(",")[1], indices.split(",")[1])
        }
        # import the ISR states up to the requested order
        isr = {}
        for bk in ["bra", "ket"]:
            sp = space_idx[bk][0]
            idx = space_idx[bk][1]

            # generate additional generic indices to construct additional
            # precursor states for the ISR basis
            n_ov = get_n_ov_from_space(sp)
            gen_idx = self.indices.get_isr_indices(
                idx, n_occ=n_ov["n_occ"], n_virt=n_ov["n_virt"]
            )
            idx_pre = []
            for symbols in gen_idx.values():
                idx_pre.extend([s.name for s in symbols])
            idx_pre = "".join(idx_pre)

            for o in range(order + 1):
                if o not in isr:
                    isr[o] = {}
                isr[o][bk] = self.isr.intermediate_state(
                    o, sp, bk, idx_is=idx, idx_pre=idx_pre
                )

        orders = get_orders_three(order)
        res = 0
        for term in orders:
            i1 = isr[term[0]]["bra"] * self.__get_shifted_h(term[1]) * \
                 isr[term[2]]["ket"]
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        return res

    @cached_member
    def mvp(self, order, mvp_space, block, indices):
        """Computes the Matrix vector product for the provided space by
           contracting the specified matrix block with an Amplitudevector.
           For example:
           space='ph', block='ph,pphh', indices='ia'
           computes the singles MVP contribution from the M_{S,D} coupling
           block.
           Substitute_dummies: works fine for the ph MVP. The pphh MVP however
           evaluates to 0, due to wrong index substitution.
           The custom substitute_indices method seems to work for all MVP
           spaces. However, it may be necessary to cancel a few terms by hand
           (interchange some indice names).
           """

        if len(mvp_space) != len(split_idxstring(indices)):
            print(f"The indices {indices} are insufficient for the"
                  f" {mvp_space} mvp.")
            exit()
        if mvp_space != block.split(",")[0]:
            print(f"The desired MVP space {mvp_space} has to be identical"
                  f"to the first secular matrix space: {block}.")
            exit()

        # generate additional indices for the secular matrix block
        b2 = block.split(",")[1]
        n_ov = get_n_ov_from_space(b2)
        idx = self.indices.get_new_gen_indices(
            n_occ=n_ov["n_occ"], n_virt=n_ov["n_virt"]
        )
        idx_str = []
        for symbols in idx.values():
            idx_str.extend([s.name for s in symbols])
        idx_str = "".join(idx_str)

        # contruct the secular matrix
        m = self.isr_matrix(order, block, indices=(indices + "," + idx_str))

        # obtain the amplitude vector
        y = self.isr.amplitude_vector(b2, idx_str)

        # Lifting index restrictions leads to a prefactor of p = 1/(no! * nv!).
        # In order to keep the resulting MVP normalized, a factor of sqrt(p)
        # is hidden inside the MVP vector, while the other part (sqrt(p))
        # is visible in the MVP expression
        # For PP ADC this leads to 1/(n!)^2 * <R|R>, which keeps the
        # normalization of the MVP.
        n_ov = get_n_ov_from_space(mvp_space)
        prefactor = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        print("prefactor from sec matrix side: ", prefactor)
        return evaluate_deltas((prefactor * m * y).expand())
