from indices import extract_names, indices, n_ov_from_space
from misc import (Inputerror, cached_member, transform_to_tuple,
                  process_arguments, validate_input)
from simplify import simplify
from sympy.physics.secondquant import wicks
from sympy import sqrt, S
from math import factorial


class properties:
    """Class for computing ISR properties. The class takes two different ISR
       as input - the second is optional. If only one is given, the standard
       ISR properties may be computed by calling the appropriate methods. This
       may also be achieved if the same ISR is provided twice.
       If both ISR are given, the expectation values of operators between
       the two ISR states like pp and ip) will be calculated. This way one may
       calculate the expectation values between e.g. a EA and a IP ISR state.
       """

    def __init__(self, isr, isr_2=None):
        if not isinstance(isr, intermediate_states) or isr_2 is not None and \
                not isinstance(isr_2, intermediate_states):
            raise Inputerror("Provided ISR must be intermediate_states.")
        self.isr = isr
        self.isr_2 = self.isr if isr_2 is None else isr_2
        self.m = secular_matrix(isr)
        self.m_2 = self.m if isr_2 is None else secular_matrix(isr_2)
        # Check if both ground states are equal. Currently this only means
        # to check that either None ot both have singles enabled.
        self.gs = isr.gs
        if self.gs.singles != self.isr_2.gs.singles:
            raise Inputerror("Both ISR need to share the same GS, "
                             "i.e. neither or both have singles enabled.")
        self.h = isr.gs.h
        self.indices = indices()

    def operator(self, order, opstring, subtract_gs=True):
        """Returns a operator - defined by opstring - shifted by the
           ground state expectation value. A one particle operator may
           be obtained by the operator string 'ca'. If subtract_gs is set, the
           ground state expectation value of the corresponding operator
           is subtracted."""
        # if the operator string does not hold an equivalent amount of creation
        # and annihilation operators, the ground state expectation value will
        # be Zero. But I guess not preventing the computation should be fine,
        # because it should not introduce a lot of overhead. Though I should
        # probably introduce the operator counting in a custom wicks
        # to reduce the overhead.

        d = self.h.operator(opstring) if order == 0 else 0
        return d - self.gs.expectation_value(order, opstring) if subtract_gs \
            else d

    @process_arguments
    @cached_member
    def op_block(self, order, block, opstring, subtract_gs=True):
        """Computes the contribution of the block IJ to the expectation
           value of the given operator
           d_{pq...} X_I <I|pq...|J>^(n) Y_J.
           If subtract_gs is set, the ground state contribution is subtracted.
           """

        block = transform_to_tuple(block)
        if len(block) != 2:
            raise Inputerror("Two space strings required to define a block."
                             f"Provided: {block}.")

        # generate indices for the block and compute the prefactors for the
        # contraction over the block space
        n_ov = n_ov_from_space(block[0])
        left_idx = self.indices.get_generic_indices(**n_ov)
        left_idx = "".join(extract_names(left_idx))
        left_pref = 1 / sqrt(
                factorial(n_ov['n_occ']) * factorial(n_ov['n_virt'])
        )

        n_ov = n_ov_from_space(block[1])
        right_idx = self.indices.get_generic_indices(**n_ov)
        right_idx = "".join(extract_names(right_idx))
        right_pref = 1 / sqrt(
                factorial(n_ov['n_occ']) * factorial(n_ov['n_virt'])
        )

        # build the ADC amplitude vectors
        left_ampl = self.isr.amplitude_vector(indices=left_idx, lr='left')
        right_ampl = self.isr_2.amplitude_vector(indices=right_idx, lr='right')

        orders = get_order_two(order)
        res = 0
        # iterate over all norm*d combinations of n'th order
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # compute d for a given norm (the overall order is split inbetween
            # both factors)
            orders_d = get_orders_three(norm_term[1])
            density = 0
            for term in orders_d:
                i1 = (left_pref * right_pref * left_ampl *
                      self.isr.intermediate_state(term[0], space=block[0],
                                                  braket='bra',
                                                  indices=left_idx) *
                      self.operator(term[1], opstring, subtract_gs) *
                      self.isr_2.intermediate_state(term[2], space=block[1],
                                                    braket='ket',
                                                    indices=right_idx) *
                      right_ampl)
                density += wicks(i1, keep_only_fully_contracted=True,
                                 simplify_kronecker_deltas=True)
            res += (norm * density).expand()
        # return simplify(res).sympy
        return res

    @process_arguments
    @cached_member
    def expectation_value(self, adc_order, opstring, order=None,
                          subtract_gs=True):
        """Computes: sum_IJ sum_pq... d_{pq...} X_I <I|pq...|J> Y_J
           for ADC(n). The desired operator may be defined via the operator
           string, where e.g. 'ccaa' gives the expectation value for a two
           particle operator.
           If additionally order is specified, only the n'th
           order contributions of all blocks that are present in the ADC(n)
           secular matrix are considered - if the blocks are expanded through
           the desired pt order.
           If subtract_gs is set, the ground state contribution will be
           subtracted.
           """
        # get all blocks that are present in the ADC(n) secular matrix and
        # their corresponding maximum pt expansion order.
        blocks = self.m.block_order(adc_order)
        blocks_2 = list(self.m_2.block_order(adc_order).keys())
        # iterate over the blocks, replacing the second space in each block
        # with the corresponding space of block_2 from isr_2
        res = 0
        for i, block in enumerate(blocks):
            block_2 = blocks_2[i]
            if len(block[0])-len(block[1]) != len(block_2[0])-len(block_2[1]):
                raise RuntimeError(f"Blocks are not in the same order: {block}"
                                   f"does not match {block_2}.")
            max_order = blocks[block]
            # block is not expanded through the given order
            if order is not None and max_order < order:
                continue
            # combine the two spaces to build the correct block with mixed
            # ADC variant spaces.
            block = (block[0], block_2[1])
            if order is not None:
                res += self.op_block(order, block=block, opstring=opstring,
                                     subtract_gs=subtract_gs)
                continue
            for o in range(max_order + 1):
                res += self.op_block(o, block=block, opstring=opstring,
                                     subtract_gs=subtract_gs)
        return res

    @process_arguments
    @cached_member
    def mod_trans_moment_space(self, order, space, opstring=None, lr='left',
                               subtract_gs=True):
        """Computes sum_x X_I <I|x|Psi_0>^(n).
           The transition operator x may be defined via the operator string
           by the number of creation and annihilation operators it includes,
           e.g. 'a' defines a single annihilation operator (IP-ADC).
           If the class holds two different ISR, one may specify which ISR to
           use for the computation of the modified transition moments via lr.
           Here 'left' refers to isr, while right refers to isr_2."""
        # Subtraction of the ground state contribution is probably not
        # necessary, because all terms cancel (at least in second order
        # Singles PP-ADC). For all other ADC variants (IP/EA...) the ground
        # state expectation value is Zero, because the number of creation and
        # annihilation operators will never be equal.
        # Give the option anyway, because I'm not sure whether it will be
        # required at higher orders for PP-ADC

        space = transform_to_tuple(space)
        if len(space) != 1:
            raise Inputerror(f"1 space string is required. Provided: {space}.")
        space = space[0]

        # - generate indices for the ISR state
        n_ov = n_ov_from_space(space)
        idx = self.indices.get_generic_indices(**n_ov)
        idx = "".join(extract_names(idx))

        # - map lr on the correct intermediate_states instance
        isr = {'left': self.isr, 'right': self.isr_2}
        isr = isr[lr]

        # - if no operator string is given -> generate a default, i.e.
        #   'a' for IP- / 'ca' for PP-ADC
        if opstring is None:
            opstring = isr.min_space[0].replace('p', 'c').replace('h', 'a')

        # - generate amplitude vector and prefactor for the summation
        ampl = isr.amplitude_vector(indices=idx, lr='left')
        pref = 1 / sqrt(factorial(n_ov['n_occ']) * factorial(n_ov['n_virt']))

        # - import the gs wavefunction (possible here)
        mp = {o: self.gs.psi(order=o, braket='ket') for o in range(order + 1)}

        # iterate over all norm*d combinations
        orders = get_order_two(order)
        res = 0
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # compute d for a given norm factor
            orders_d = get_orders_three(norm_term[1])
            d = 0
            for term in orders_d:
                i1 = (pref * ampl *
                      isr.intermediate_state(order=term[0], space=space,
                                             braket='bra', indices=idx) *
                      self.operator(term[1], opstring, subtract_gs) *
                      mp[term[2]])
                d += wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
            res += (norm * d).expand()
        return simplify(res).sympy

    @process_arguments
    @cached_member
    def mod_trans_moment(self, adc_order, opstring=None, order=None, lr='left',
                         subtract_gs=True):
        """Computes sum_I sum_x X_I <I|x|Psi_0>
           for a given ADC order, taking all necessary excitation spaces I into
           account.
           The operator x may be defined via the operator string (e.g. 'ca'
           gives a one_particle operator for PP-ADC). If no operator is
           provided, the standard transition operator for the ADC variant will
           be constructed by default, i.e. 'a' for IP and 'c' for EA.
           If the parameter order is given, only contribution of the desired
           order are computed, e.g. for PP-ADC adc_order=2, order=2 gives only
           the second order <S|x|Psi_0> contribution.
           The parameter lr controls whether isr (left) or isr_2 (right) is
           used for determining the modified transition moments.
           Subtract_gs controls, wether the ground state expectation value
           is subtracted or not."""

        # obtain the maximum order through which all the spaces are expanded
        # in the secular matrix
        m = {'left': self.m, 'right': self.m_2}
        m = m[lr]
        max_orders = m.max_ptorder_spaces(adc_order)

        res = 0
        for space, max_order in max_orders.items():
            # the space is not expanded through the desired order
            if order is not None and max_order < order:
                continue
            if order is not None:
                res += self.mod_trans_moment_space(order=order, space=space,
                                                   opstring=opstring, lr=lr,
                                                   subtract_gs=subtract_gs)
                continue
            for o in range(max_order + 1):
                res += self.mod_trans_moment_space(order=o, space=space,
                                                   opstring=opstring, lr=lr,
                                                   subtract_gs=subtract_gs)
        return res
