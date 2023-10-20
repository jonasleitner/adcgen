from sympy.physics.secondquant import wicks, evaluate_deltas
from sympy import sqrt, S

from math import factorial

from .indices import repeated_indices, Indices
from .misc import (Inputerror, cached_member, transform_to_tuple,
                   process_arguments, validate_input)
from .func import gen_term_orders
from .simplify import simplify
from .expr_container import Expr
from .rules import Rules


class SecularMatrix:
    def __init__(self, isr):
        from .isr import IntermediateStates

        if not isinstance(isr, IntermediateStates):
            raise Inputerror("Invalid intermediate states object.")
        self.isr = isr
        self.gs = isr.gs
        self.h = isr.gs.h
        self.indices = Indices()

    def hamiltonian(self, order: int, subtract_gs: bool):
        h = {0: self.h.h0, 1: self.h.h1}
        h, rules = h.get(order, (0, Rules()))
        if subtract_gs:
            return h - self.gs.energy(order), rules
        else:
            return h, rules

    @process_arguments
    @cached_member
    def precursor_matrix_block(self, order: int, block: str, indices: str,
                               subtract_gs: bool = True):
        """Computes a secular matrix block of in the basis of the
           precursor states. If subtract_gs is set (which it is by default),
           the ground state energy is subtracted from the Hamiltonian.
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("Precursor matrix requires two index strings.")

        if repeated_indices(indices[0], indices[1]):
            raise Inputerror("Found repeating index in bra and ket.")
        bra_space, ket_space = block
        bra_idx, ket_idx = indices

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = 0
        # 1) iterate through all combinations of norm_factor*M^#
        for (norm_order, matrix_order) in orders:
            norm = self.gs.norm_factor(norm_order)
            if norm is S.Zero:
                continue
            # 2) construct M^# for a given norm_factor
            # the overall order is split between the norm factor and M^#
            orders_M = gen_term_orders(
                order=matrix_order, term_length=3, min_order=0
            )
            matrix = 0
            for (bra_order, op_order, ket_order) in orders_M:
                operator, rules = self.hamiltonian(op_order, subtract_gs)
                if operator == 0:
                    continue
                itmd = (self.isr.precursor(order=bra_order, space=bra_space,
                                           braket='bra', indices=bra_idx) *
                        operator *
                        self.isr.precursor(order=ket_order, space=ket_space,
                                           braket='ket', indices=ket_idx))
                itmd = wicks(itmd, keep_only_fully_contracted=True,
                             simplify_kronecker_deltas=True)
                matrix += itmd
            # evaluate_deltas should not be necessary here, because norm only
            # contains contracted indices
            res += (norm * matrix).expand()
        return simplify(Expr(res)).sympy

    @process_arguments
    @cached_member
    def isr_matrix_block(self, order: int, block: str, indices: str,
                         subtract_gs: bool = True):
        """Computes a secular matrix block in the basis of intermediate states.
           If subtract_gs is set (which it is by default) the ground state
           energy is subtracted from the Hamiltonian.
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("ISR matrix requires 2 index strings.")

        if repeated_indices(indices[0], indices[1]):
            raise Inputerror("Found a repeating index in bra and ket.")
        bra_space, ket_space = block
        bra_idx, ket_idx = indices

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = 0
        # 1) iterate through all combinations of norm_factor*M
        for (norm_order, matrix_order) in orders:
            norm = self.gs.norm_factor(norm_order)
            if norm is S.Zero:
                continue
            # 2) construct M for a given norm_factor
            # the overall order is split between the norm_factor and M
            orders_M = gen_term_orders(
                order=matrix_order, term_length=3, min_order=0
            )
            matrix = 0
            for (bra_order, op_order, ket_order) in orders_M:
                operator, rules = self.hamiltonian(op_order, subtract_gs)
                if operator == 0:
                    continue
                itmd = (self.isr.intermediate_state(order=bra_order,
                                                    space=bra_space,
                                                    braket='bra',
                                                    indices=bra_idx) *
                        operator *
                        self.isr.intermediate_state(order=ket_order,
                                                    space=ket_space,
                                                    braket='ket',
                                                    indices=ket_idx))
                itmd = wicks(itmd, keep_only_fully_contracted=True,
                             simplify_kronecker_deltas=True)
                matrix += itmd
            # evaluate deltas should not be necessary here
            res += (norm * matrix).expand()
        return simplify(Expr(res)).sympy

    @process_arguments
    @cached_member
    def mvp_block_order(self, order: int, space: str, block: str,
                        indices: str, subtract_gs: bool = True):
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
        from .indices import n_ov_from_space, extract_names

        space = transform_to_tuple(space)
        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, space=space, block=block,
                       indices=indices)
        if len(indices) != 1:
            raise Inputerror(f"Invalid index input for MVP: {indices}")
        space = space[0]
        indices = indices[0]
        if space != block[0]:
            raise Inputerror(f"The desired MVP space {space} has to match "
                             f"the bra space of the secular matrix block: "
                             f"{block}.")

        # generate additional indices for the ket state of the secular matrix
        n_ov = n_ov_from_space(block[1])
        idx = self.indices.get_generic_indices(**n_ov)
        idx = "".join(extract_names(idx))

        # contruct the secular matrix block
        m = self.isr_matrix_block(order=order, block=block,
                                  indices=(indices, idx),
                                  subtract_gs=subtract_gs)

        # generate the amplitude vector
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
        n_ov = n_ov_from_space(space)
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

        # print(f"prefactors for {space} MVP from block {block}: "
        #       f"{prefactor_mvp}, {prefactor_ampl}.")
        return evaluate_deltas(
            (prefactor_mvp * prefactor_ampl * m * y).expand()
        )

    @process_arguments
    @cached_member
    def mvp(self, adc_order: int, space: str, indices: str, order: int = None,
            subtract_gs: bool = True):
        """Computes the matrix vector product for a given space by contracting
           all relevant blocks of the ADC(n) secular matrix with the
           appropriate ADC amplitude vector.
           If the order keyword is given only contributions of the desired
           order are computed, e.g., adc_order=2, space='ph', order=2
           only computes the 2nd order contribution of the 'ph,ph' matrix
           block, since the 'ph,pphh' coupling block is not expanded through
           the desired order."""
        # validate the input parameters
        space = transform_to_tuple(space)
        indices = transform_to_tuple(indices)
        validate_input(adc_order=adc_order, space=space, indices=indices)
        if order is not None:
            validate_input(order=order)
        if len(indices) != 1:
            raise Inputerror(f"Invalid indices for MVP: {indices}")
        space, indices = space[0], indices[0]
        # check that the space is valid for the current adc variant
        if not self.isr.validate_space(space):
            raise Inputerror(f"The space {space} is not valid for the given "
                             f"adc variant {self.isr.variant}.")
        # and that the space is present at the desired adc_order
        if space not in self.max_ptorder_spaces(adc_order):
            raise Inputerror(f"The space {space} is not present in "
                             f"{self.isr.variant}-ADC({adc_order})")

        # add up all blocks that contribute to the given mvp
        mvp = 0
        for block, max_order in self.block_order(adc_order).items():
            if space != block[0] or (order is not None and max_order < order):
                continue
            if order is None:  # compute all contributions of the block
                for o in range(max_order + 1):
                    mvp += self.mvp_block_order(order=o, space=space,
                                                block=block, indices=indices,
                                                subtract_gs=subtract_gs)
            else:  # only compute contributions of the specified order
                mvp += self.mvp_block_order(order=order, space=space,
                                            block=block, indices=indices,
                                            subtract_gs=subtract_gs)
        return mvp

    @process_arguments
    @cached_member
    def expectation_value_block_order(self, order: int, block: str,
                                      subtract_gs: bool = True):
        """Computes the n-th order contribution of a specific secular matrix
           block to the energy expectation value.
           If subtract_gs is set, the ground state energy is subtracted.
           """
        from .indices import n_ov_from_space, extract_names

        block = transform_to_tuple(block)
        validate_input(order=order, block=block)

        # generate indices for the mvp
        mvp_idx = self.indices.get_generic_indices(**n_ov_from_space(block[0]))
        mvp_idx = "".join(extract_names(mvp_idx))
        # compute the MVP
        mvp = self.mvp_block_order(order, space=block[0], block=block,
                                   indices=mvp_idx, subtract_gs=subtract_gs)
        # generate the left amplitude vector
        left = self.isr.amplitude_vector(mvp_idx, lr='left')
        # call simplify -> symmetry of left amplitude vector might reduce
        #                  the number of terms
        # prefactors: I think there is no need for any further prefactors
        #  E = 1/sqrt(l) * 1/sqrt(r) sum_I,J  X_I M_I,J Y_J
        #    -> already included in the mvp function
        return simplify(Expr(left * mvp)).sympy

    @process_arguments
    @cached_member
    def expectation_value(self, adc_order: int, order: int = None,
                          subtract_gs: bool = True):
        """Computes the ADC(n) energy expectation value.
           If subtract_gs is set (which it is by default), the ground state
           energy is subtracted.
           If the order parameter is specified, only contributions of the
           specified order are computed."""
        expec = 0
        for block, max_order in self.block_order(adc_order).items():
            # is the mvp expanded through the desired order?
            # e.g. ADC(4) S -> 4 // D -> 3 // T -> 2
            if order is not None and max_order < order:
                continue
            if order is None:  # compute all contributions of the block
                for o in range(max_order + 1):
                    expec += self.expectation_value_block_order(
                        order=o, block=block, subtract_gs=subtract_gs
                    )
            else:  # only compute contibutions of the specified order
                expec += self.expectation_value_block_order(
                    order=order, block=block, subtract_gs=subtract_gs
                )
        # it should not be possible to simplify any further here, because left
        # and right amplitude vector have different names
        return expec

    def max_ptorder_spaces(self, order):
        """Returns a dict with the maximum pt order of each space at the
           ADC(n) level.
           """

        space = self.isr.min_space[0]
        ret = {space: order}
        for i in range(1, order//2 + 1):
            space = f"p{space}h"
            ret[space] = order - i
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
