from sympy.physics.secondquant import wicks
from sympy import sqrt, S

from math import factorial

from misc import (Inputerror, cached_member, transform_to_tuple,
                  process_arguments, validate_input)
from simplify import simplify


class secular_matrix:
    def __init__(self, isr):
        self.isr = isr
        self.gs = isr.gs
        self.h = isr.gs.h
        self.indices = indices()

    def __get_shifted_h(self, order):
        get_H = {
            0: self.h.h0,
            1: self.h.h1,
        }
        h = get_H[order] if order < 2 else 0
        return h - self.gs.energy(order)

    @process_arguments
    @cached_member
    def precursor_matrix_block(self, order, block, indices):
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

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        if len(block) != 2 or len(indices) != 2:
            raise Inputerror("Precursor matrix requires two block and indice "
                  f"strings. Block {block} and indice {indices} are not valid")

        if repeated_indices(indices[0], indices[1]):
            raise Inputerror("Indices for precursor secular matrix should not "
                             f"be equal. Provided indice string: {indices}")

        orders = get_order_two(order)
        res = 0
        # 1) iterate through all combinations of norm_factor*M^#
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct M^# for a given norm_factor
            # the overall order is split between the norm factor and M^#
            orders_M = get_orders_three(norm_term[1])
            matrix = 0
            for term in orders_M:
                i1 = (self.isr.precursor(term[0], space=block[0], braket="bra",
                                         indices=indices[0]) *
                      self.__get_shifted_h(term[1]) *
                      self.isr.precursor(term[2], space=block[1], braket="ket",
                                         indices=indices[1]))
                i1 = wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
                matrix += i1
            # evaluate_deltas should not be necessary here, because norm only
            # contains contracted indices
            res += (norm * matrix).expand()
        return simplify(res).sympy

    @process_arguments
    @cached_member
    def isr_matrix_block(self, order, block, indices):
        """Computes a specific block of a specific order of the secular matrix.
           Substitute_dummie: 1) messes around with the indices (see
           precursor_matrix_block) 2) for the coupling blocks and the
           'pphh,pphh' block wrong index substitution causes all terms to
           cancel - which is wrong.
           The custom substitute_indices function may be used instead. It
           gives correct results for ADC(2). No additional manipulation of
           terms by hand was necessary.
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        if len(block) != 2 or len(indices) != 2:
            raise Inputerror("Precursor matrix requires two block and index "
                             f"strings. Block {block} and indice {indices}"
                             "are not valid.")

        if repeated_indices(indices[0], indices[1]):
            raise Inputerror("Indices for isr secular matrix should not be ",
                             f"equal. Provided indice string: {indices}")

        orders = get_order_two(order)
        res = 0
        # 1) iterate through all combinations of norm_factor*M
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct M for a given norm_factor
            # the overall order is split between the norm_factor and M
            orders_M = get_orders_three(norm_term[1])
            matrix = 0
            for term in orders_M:
                i1 = (self.isr.intermediate_state(term[0], space=block[0],
                                                  braket="bra",
                                                  indices=indices[0]) *
                      self.__get_shifted_h(term[1]) *
                      self.isr.intermediate_state(term[2], space=block[1],
                                                  braket="ket",
                                                  indices=indices[1]))
                i1 = wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
                matrix += i1
            # evaluate deltas should not be necessary here
            res += (norm * matrix).expand()
        return simplify(res).sympy

    @process_arguments
    @cached_member
    def mvp_block_order(self, order, mvp_space, block, indices):
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

        mvp_space = transform_to_tuple(mvp_space)
        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        if len(mvp_space) != 1 or len(indices) != 1 or len(block) != 2:
            raise Inputerror(f"Bad input for MVP: mvp_space {mvp_space} / "
                             f"matrix block {block} / mvp indices {indices}.")
        mvp_space = mvp_space[0]
        indices = indices[0]
        if len(mvp_space) != len(split_idx_string(indices)):
            raise Inputerror(f"The indices {indices} are insufficient for the"
                             f" {mvp_space} mvp.")
        if mvp_space != block[0]:
            raise Inputerror(f"The desired MVP space {mvp_space} has to be "
                             f"identical to the first secular matrix space: "
                             f"{block}.")

        # I think it should not be possible to add the resulting mvp vector r
        # in the result by multiplying with the approrpiate tensor
        # AntiSymmetricTensor("r", sorted(idx[virt], sorted(idx[occ]))),
        # because this automatically would automatically also introduce the
        # antisymmetry of the occ and virt indices. However, in this function
        # we are just defining that r is antisym wrt index permutations. It
        # should be possible, but it requires the correct setup of the
        # resulting mvp tensor - which is not done atm.

        # generate additional indices for the secular matrix block
        n_ov = n_ov_from_space(block[1])
        idx = self.indices.get_generic_indices(**n_ov)
        idx_str = [s.name for sym in idx.values() for s in sym]
        idx_str = "".join(idx_str)

        # contruct the secular matrix
        m = self.isr_matrix_block(
            order, block, indices=(indices + "," + idx_str)
        )

        # obtain the amplitude vector
        y = self.isr.amplitude_vector(indices=idx_str, lr="right")

        # Lifting index restrictions leads to a prefactor of p = 1/(no! * nv!).
        # In order to keep the resulting amplitude vector normalized, a factor
        # of sqrt(p) is hidden inside the MVP vector, while the other part
        # (sqrt(p)) is visible in the MVP expression. This prefactor takes care
        # about the norm of the resulting MVP.
        # For the contraction with the guess amplitude vector another prefactor
        # (see below) has to be introduced.
        # For PP ADC this leads to 1/(n!)^2 * <R|R>, which keeps the
        # normalization of the MVP.
        n_ov = n_ov_from_space(mvp_space)
        prefactor_mvp = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        # The same argument leads to a similar prefactor for the contraction
        # with the amplitude vector (the norm of the amplitude vector also
        # needs to be correct). Instead of introducing a prefactor that
        # seamingly just appears out of nowhere the prefactor that is present
        # due to the contraction over the amplitude vector indices needs to be
        # adjusted. Essentially the same formula and argumentation may be used
        # only applied to a different space, namely block[1] - which is the
        # space the sum contracts.
        n_ov = n_ov_from_space(block[1])
        prefactor_ampl = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        # print(f"prefactors for {mvp_space} MVP from block {block}: "
        #       f"{prefactor_mvp}, {prefactor_ampl}.")
        return evaluate_deltas(
            (prefactor_mvp * prefactor_ampl * m * y).expand()
        ).sympy

    def max_ptorder_spaces(self, order):
        """Returns a dict with the maximum pt order of each space at the
           ADC(n) level.
           """

        space = self.isr.min_space[0]
        ret = {space: order}
        for s in range(1, int(order/2) + 1):
            space = "p" + space + "h"
            ret[space] = order - s
        return ret

    def block_order(self, order):
        """Returns the order to which each block of the ADC(n) secular
           is expanded.
           Returns a dict with the block tuple (s1, s2) as key.
           """
        from itertools import product

        max_orders = self.max_ptorder_spaces(order)
        blocks = list(product(max_orders.keys(), max_orders.keys()))
        min_space = self.isr.min_space[0]
        ret = {}
        for block in blocks:
            s1 = block[0]
            s2 = block[1]
            # diagonal
            if s1 == s2:
                dif = abs(len(s1) - len(min_space))
                ret[block] = max_orders[min_space] - dif
            # off diagonal
            else:
                dif = int(abs(len(s1) - len(s2)) / 2)
                diag = max_orders[min_space] - \
                    abs(len(max([s1, s2])) - len(min_space))
                ret[block] = diag + dif
        return ret
