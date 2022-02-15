from sympy import latex, Dummy
from sympy.physics.secondquant import wicks, substitute_dummies

from isr import get_orders_three
from indices import split_idxstring, pretty_indices
from misc import cached_member


class secular_matrix:
    def __init__(self, isr):
        self.isr = isr
        self.gs = isr.gs
        self.h = isr.gs.h
        self.indices = isr.gs.indices
        # {order: x}
        # {order: {block: {indices: x}}}
        self.pre_matrix = {}
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
           basis of the precursor states."""

        for space in block.split(","):
            if space not in self.isr.order_spaces:
                print("Requested a matrix block for an unknown space."
                      f"Valid blocks are {list(self.isr.order_spaces.keys())}")
                exit()

        sorted_idx = []
        for idx in indices.split(","):
            sorted_idx.append("".join(sorted(split_idxstring(idx))))
        if sorted_idx[0] == sorted_idx[1]:
            print("Indices for overlap matrix should not be equal."
                  f"Provided indice string: {indices}")
            exit()
        indices = ",".join(sorted_idx)

    def get_precursor_matrix_block(self, order, block, indices):
        # block: "ph,pphh"
        # indices: "ia,jbkl"
        for space in block.split(","):
            if space not in self.isr.order_spaces:
                print("Requested a matrix block for an unknown space.",
                      f"Valid blocks are {list(self.isr.order_spaces.keys())}")
                exit()
        sorted_idx = []
        for idx in indices.split(","):
            sorted_idx.append("".join(sorted(split_idxstring(idx))))
        if sorted_idx[0] == sorted_idx[1]:
            print("Indices for overlap matrix should not be equal.",
                  "Use e.g. ia,jb and not ia,ia.")
            exit()
        indices = ",".join(sorted_idx)

        if order not in self.pre_matrix:
            self.pre_matrix[order] = {}
        if block not in self.pre_matrix[order]:
            self.pre_matrix[order][block] = {}
        if indices not in self.pre_matrix[order][block]:
            self.__build_precursor_matrix_block(order, block, indices)
        return self.pre_matrix[order][block][indices]

    def __build_precursor_matrix_block(self, order, block, indices):
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
                id = space_idx[bk][1]
                pre[o][bk] = self.isr.get_precursor(o, sp, bk, id)

        orders = get_orders_three(order)
        res = 0
        for term in orders:
            i1 = pre[term[0]]["bra"] * self.__get_shifted_h(term[1]) * \
                 pre[term[2]]["ket"]
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        res = self.indices.substitute_indices(res)
        self.pre_matrix[order][block][indices] = res
        print(latex(res))
        print(res.atoms(Dummy))

    def get_matrix_block(self, order, block, indices):
        # block: "ph,pphh"
        # indices: "ia,jbkl"
        for space in block.split(","):
            if space not in self.isr.order_spaces:
                print("Requested a matrix block for an unknown space.",
                      f"Valid blocks are {list(self.isr.order_spaces.keys())}")
                exit()
        sorted_idx = []
        for idx in indices.split(","):
            sorted_idx.append("".join(sorted(split_idxstring(idx))))
        if sorted_idx[0] == sorted_idx[1]:
            print("Indices for overlap matrix should not be equal.",
                  "Use e.g. ia,jb and not ia,ia.")
            exit()
        indices = ",".join(sorted_idx)

        if order not in self.matrix:
            self.matrix[order] = {}
        if block not in self.matrix[order]:
            self.matrix[order][block] = {}
        if indices not in self.matrix[order][block]:
            self.__build_matrix_block(order, block, indices)
        return self.matrix[order][block][indices]

    def __build_matrix_block(self, order, block, indices):
        space_idx = {
            "bra": (block.split(",")[0], indices.split(",")[0]),
            "ket": (block.split(",")[1], indices.split(",")[1])
        }
        # import the ISR states up to the requested order
        isr = {}
        for bk in ["bra", "ket"]:
            sp = space_idx[bk][0]
            id = space_idx[bk][1]
            # obtain generic indices to construct the ISR basis
            gen_idx = self.indices.get_isr_indices(
                      id, n_occ=self.isr.order_spaces[sp],
                      n_virt=self.isr.order_spaces[sp]
            )
            idx_pre = [s.name for s in gen_idx["virt"]]
            for s in gen_idx["occ"]:
                idx_pre += s.name
            idx_pre = "".join(idx_pre)
            for o in range(order + 1):
                if o not in isr:
                    isr[o] = {}
                isr[o][bk] = self.isr.get_is(o, sp, bk, id, idx_pre)

        orders = get_orders_three(order)
        res = 0
        for term in orders:
            i1 = isr[term[0]]["bra"] * self.__get_shifted_h(term[1]) * \
                 isr[term[2]]["ket"]
            res += wicks(i1, keep_only_fully_contracted=True,
                         simplify_kronecker_deltas=True)
        self.matrix[order][block][indices] = res
        print(latex(substitute_dummies(res, new_indices=True,
                                       pretty_indices=pretty_indices)))
