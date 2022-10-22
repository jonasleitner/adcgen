from sympy.physics.secondquant import wicks
from sympy import sqrt, S

from math import factorial

from indices import repeated_indices
from misc import (Inputerror, cached_member, transform_to_tuple,
                  process_arguments, validate_input)
from func import evaluate_deltas, gen_term_orders
from simplify import simplify


class secular_matrix:
    def __init__(self, isr):
        from isr import intermediate_states
        from indices import indices
        if not isinstance(isr, intermediate_states):
            raise Inputerror("Invalid intermediate_states object.")
        self.isr = isr
        self.gs = isr.gs
        self.h = isr.gs.h
        self.indices = indices()

    def __shifted_h(self, order):
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
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("Precursor matrix requires two index strings.")

        if repeated_indices(indices[0], indices[1]):
            raise Inputerror("No index should occur in both spaces.")

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = 0
        # 1) iterate through all combinations of norm_factor*M^#
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct M^# for a given norm_factor
            # the overall order is split between the norm factor and M^#
            orders_M = gen_term_orders(
                order=norm_term[1], term_length=3, min_order=0
            )
            matrix = 0
            for term in orders_M:
                i1 = (self.isr.precursor(order=term[0], space=block[0],
                                         braket="bra", indices=indices[0]) *
                      self.__shifted_h(term[1]) *
                      self.isr.precursor(order=term[2], space=block[1],
                                         braket="ket", indices=indices[1]))
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
        """Computes a block of the secular matrix in the IS basis.
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("ISR matrix requires 2 index strings.")

        if repeated_indices(indices[0], indices[1]):
            raise Inputerror("No index should occur in both spaces of a block")

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = 0
        # 1) iterate through all combinations of norm_factor*M
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct M for a given norm_factor
            # the overall order is split between the norm_factor and M
            orders_M = gen_term_orders(
                order=norm_term[1], term_length=3, min_order=0
            )
            matrix = 0
            for term in orders_M:
                i1 = (self.isr.intermediate_state(order=term[0],
                                                  space=block[0],
                                                  braket="bra",
                                                  indices=indices[0]) *
                      self.__shifted_h(term[1]) *
                      self.isr.intermediate_state(order=term[2],
                                                  space=block[1],
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
        from indices import n_ov_from_space, extract_names

        mvp_space = transform_to_tuple(mvp_space)
        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, mvp_space=mvp_space, block=block,
                       indices=indices)
        if len(indices) != 1:
            raise Inputerror(f"Invalid index input for MVP: {indices}")
        mvp_space = mvp_space[0]
        indices = indices[0]
        if mvp_space != block[0]:
            raise Inputerror(f"The desired MVP space {mvp_space} has to be "
                             f"identical to the first secular matrix space: "
                             f"{block}.")

        # generate additional indices for the secular matrix block
        n_ov = n_ov_from_space(block[1])
        idx = self.indices.get_generic_indices(**n_ov)
        idx = "".join(extract_names(idx))

        # contruct the secular matrix block
        m = self.isr_matrix_block(
            order=order, block=block, indices=(indices, idx)
        )

        # obtain the amplitude vector
        y = self.isr.amplitude_vector(indices=idx, lr="right")

        # Lifting index restrictions leads to two prefactors
        # p = 1/sqrt(n_o! * n_v!), in order to keep the amplitude vector and
        # the resulting mvp vector normalized!
        # Note, that n_o and n_v might differ for both amplitudes, leading to
        # generally different prefactors p.
        # In order to keep both vectors normalized they are each multiplied by
        # a factor p, i.e., a factor p is 'hidden' in both vectors.

        # - To keep the equality r = M * Y we also have to multiply the right
        #   hand side of the equation with p if we multiply r with p
        n_ov = n_ov_from_space(mvp_space)
        prefactor_mvp = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        # - lifting the sum restrictions leads to a prefactor of p ** 2.
        #   However, p is hidden inside the amplitude vector -> only p present
        #   in the MVP equations
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
        for s in range(1, order//2 + 1):
            space = f"p{space}h"
            ret[space] = order - s
        return ret

    def block_order(self, order):
        """Returns the order through which each block of the ADC(n) secular
           is expanded.
           Returns a dict with the block tuple (s1, s2) as key.
           """
        from itertools import product

        max_orders = self.max_ptorder_spaces(order)
        spaces = sorted(max_orders, key=lambda sp: len(sp))
        min_space = self.isr.min_space[0]
        ret = {}
        for block in product(spaces, spaces):
            s1, s2 = block
            # diagonal
            if s1 == s2:
                ret[block] = order - (len(s1) - len(min_space))
            # off diagonal
            else:
                dif = abs(len(s1) - len(s2)) // 2
                diag = order - (len(min(block)) - len(min_space))
                ret[block] = diag - dif
        return ret
