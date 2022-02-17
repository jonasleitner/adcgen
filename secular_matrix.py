from sympy.physics.secondquant import wicks, evaluate_deltas

from isr import get_orders_three
from indices import check_repeated_indices, split_idxstring
from misc import cached_member


class secular_matrix:
    def __init__(self, isr):
        self.isr = isr
        self.gs = isr.gs
        self.h = isr.gs.h
        self.indices = isr.gs.indices
        # {order: x}
        # {order: {block: {indices: x}}}
        # self.pre_matrix = {}
        self.matrix = {}

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
           Due to substitute_dummies, it is hard to identify to which matrix
           element an individual term belongs.
           """

        if len(block.split(",")) != 2 or len(indices.split(",")) != 2:
            print("Precursor matrix requires two block and indice strings that"
                  f"are separated by a ','. Block {block} and indice {indices}"
                  "are not valid.")
            exit()
        for space in block.split(","):
            if space not in self.isr.order_spaces:
                print("Requested a matrix block for an unknown space."
                      f"Valid blocks are {list(self.isr.order_spaces.keys())}")
                exit()

        sorted_idx = []
        for idx in indices.split(","):
            sorted_idx.append("".join(sorted(split_idxstring(idx))))
        if check_repeated_indices(sorted_idx[0], sorted_idx[1]):
            print("Indices for overlap matrix should not be equal."
                  f"Provided indice string: {indices}")
            exit()
        indices = ",".join(sorted_idx)

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
                pre[o][bk] = self.isr.precursor(o, sp, bk, idx)

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
           Due to substitute_dummies care has to be taken to which matrix
           element an individual term belongs.
           """

        if len(block.split(",")) != 2 or len(indices.split(",")) != 2:
            print("Precursor matrix requires two block and indice strings that"
                  f"are separated by a ','. Block {block} and indice {indices}"
                  "are not valid.")
            exit()
        for space in block.split(","):
            if space not in self.isr.order_spaces:
                print("Requested a matrix block for an unknown space.",
                      f"Valid blocks are {list(self.isr.order_spaces.keys())}")
                exit()

        sorted_idx = []
        for idx in indices.split(","):
            sorted_idx.append("".join(sorted(split_idxstring(idx))))
        if check_repeated_indices(sorted_idx[0], sorted_idx[1]):
            print("Indices for overlap matrix should not be equal.",
                  f"Provided indice string: {indices}")
            exit()
        indices = ",".join(sorted_idx)

        space_idx = {
            "bra": (block.split(",")[0], indices.split(",")[0]),
            "ket": (block.split(",")[1], indices.split(",")[1])
        }
        # import the ISR states up to the requested order
        isr = {}
        for bk in ["bra", "ket"]:
            sp = space_idx[bk][0]
            idx = space_idx[bk][1]

            # generate additional generic indices to construct the ISR basis
            gen_idx = self.indices.get_isr_indices(
                      idx, n_occ=self.isr.order_spaces[sp],
                      n_virt=self.isr.order_spaces[sp]
            )
            idx_pre = [s.name for s in gen_idx["virt"]]
            idx_pre.extend([s.name for s in gen_idx["occ"]])
            idx_pre = "".join(idx_pre)

            for o in range(order + 1):
                if o not in isr:
                    isr[o] = {}
                isr[o][bk] = self.isr.intermediate_state(
                    o, sp, bk, idx, idx_pre
                )

        orders = get_orders_three(order)
        res = 0
        for term in orders:
            i1 = isr[term[0]]["bra"] * self.__get_shifted_h(term[1]) * \
                 isr[term[2]]["ket"]
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        return res

    def mvp(self, order, mvp_space, block, indices):
        """Computes the Matrix vector product for the provided space by
           contracting the specified matrix block with an Amplitudevector.
           For example:
           space='ph', block='ph,pphh', indices='ia'
           computes the singles MVP contribution from the M_{S,D} coupling
           block.
           """

        if len(mvp_space) != len(split_idxstring(indices)):
            print(f"The indices {indices} are insufficient for the space"
                  f"{mvp_space}.")
            exit()
        if mvp_space not in block.split(","):
            print(f"The desired MVP space {mvp_space} needs to be present in"
                  f"the secular matrix block {block}.")
            exit()

        # generate additional indices for the secular matrix block
        b2 = block.split(",")[1]
        idx = self.indices.get_new_gen_indices(
            n_occ=self.isr.order_spaces[b2], n_virt=self.isr.order_spaces[b2]
        )
        idx_str = [s.name for s in idx["occ"]]
        idx_str.extend(s.name for s in idx["virt"])
        idx_str = "".join(idx_str)

        # contruct the secular matrix
        m = self.isr_matrix(order, block, indices + "," + idx_str)

        # obtain the amplitude vector
        y = self.isr.amplitude_vector(b2, idx_str)

        return evaluate_deltas((m * y).expand())
