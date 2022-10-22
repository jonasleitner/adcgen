from func import gen_term_orders
from indices import extract_names, indices, n_ov_from_space
from misc import (Inputerror, cached_member, transform_to_tuple,
                  process_arguments, validate_input)
from simplify import simplify
from sympy.physics.secondquant import wicks
from sympy import sqrt, S, sympify
from math import factorial


class properties:
    """Class for computing ISR properties. The class takes two different ISR
       as arguments - the second is optional. If only one is given, the
       standard ISR properties may be computed by calling the appropriate
       methods. This may also be achieved if the same ISR is provided twice.
       If both ISR are given, the expectation values of operators between
       the two ISR states like pp and ip will be calculated. This way one may
       calculate the expectation values between e.g. a EA and a IP ISR state.
       """

    def __init__(self, l_isr, r_isr=None):
        from isr import intermediate_states
        from secular_matrix import secular_matrix

        if not isinstance(l_isr, intermediate_states) or r_isr is not None \
                and not isinstance(r_isr, intermediate_states):
            raise Inputerror("Provided ISR must be intermediate_states.")
        self.l_isr = l_isr
        self.r_isr = self.l_isr if r_isr is None else r_isr
        self.l_m = secular_matrix(l_isr)
        self.r_m = self.l_m if r_isr is None else secular_matrix(r_isr)
        # Check if both ground states are equal. Currently this only means
        # to check that either None ot both have singles enabled.
        self.gs = l_isr.gs
        if self.gs.singles != self.r_isr.gs.singles:
            raise Inputerror("Both ISR need to share the same GS, "
                             "i.e. neither or both have singles enabled.")
        self.h = l_isr.gs.h
        self.indices = indices()

    def operator(self, order, opstring, subtract_gs=True):
        """Returns a operator - defined by opstring. For instance, a one
           particle operator may be obtained by requesting the opstring 'ca'.
           If subtract_gs is set, the ground state expectation value of the
           corresponding operator is subtracted."""
        # if the operator string does not hold an equivalent amount of creation
        # and annihilation operators, the ground state expectation value will
        # be Zero. But I guess not preventing the computation should be fine,
        # because it should not introduce a lot of overhead. Though I should
        # probably introduce the operator counting in a custom wicks
        # to reduce the overhead.
        validate_input(order=order, opstring=opstring)

        d = self.h.operator(opstring=opstring) if order == 0 else sympify(0)
        return d - self.gs.expectation_value(order=order, opstring=opstring) \
            if subtract_gs else d

    @process_arguments
    @cached_member
    def op_block(self, order, block, opstring, subtract_gs=True):
        """Computes the contribution of the IJ block to the expectation
           value of the given operator
           d_{pq...} X_I <I|pq...|J>^(n) Y_J.
           If subtract_gs is set, the ground state contribution is subtracted.
           """

        block = transform_to_tuple(block)
        validate_input(order=order, block=block, opstring=opstring)

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
        left_ampl = self.l_isr.amplitude_vector(indices=left_idx, lr='left')
        right_ampl = self.r_isr.amplitude_vector(indices=right_idx, lr='right')

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = sympify(0)
        # iterate over all norm*d combinations of n'th order
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # compute d for a given norm (the overall order is split inbetween
            # both factors)
            orders_d = gen_term_orders(
                order=norm_term[1], term_length=3, min_order=0
            )
            expec = 0
            for term in orders_d:
                i1 = (left_pref * right_pref * left_ampl *
                      self.l_isr.intermediate_state(order=term[0],
                                                    space=block[0],
                                                    braket='bra',
                                                    indices=left_idx) *
                      self.operator(term[1], opstring, subtract_gs) *
                      self.r_isr.intermediate_state(order=term[2],
                                                    space=block[1],
                                                    braket='ket',
                                                    indices=right_idx) *
                      right_ampl)
                expec += wicks(i1, keep_only_fully_contracted=True,
                               simplify_kronecker_deltas=True)
            res += (norm * expec).expand()
        return simplify(res).sympy

    @process_arguments
    @cached_member
    def expectation_value(self, adc_order, opstring, order=None,
                          subtract_gs=True):
        """Computes: sum_IJ sum_pq... d_{pq...} X_I <I|pq...|J> Y_J
           for ADC(n), i.e., for all blocks present in the ADC(n) matrix.
           The desired operator may be defined via the operator
           string, where e.g. 'ccaa' gives the expectation value for a two
           particle operator.
           If additionally order is specified, only the m'th
           order contributions of all blocks that are present in the ADC(n)
           secular matrix are determined - if the blocks are expanded through
           the desired pt order.
           If subtract_gs is set, the ground state contribution will be
           subtracted.
           """
        validate_input(adc_order=adc_order, opstring=opstring)
        if order is not None:
            validate_input(order=order)
        # get all blocks that are present in the ADC(n) secular matrix and
        # the order through which they are expanded.
        left_blocks = self.l_m.block_order(adc_order)
        left_blocks = sorted(left_blocks.items(),
                             key=lambda tpl: (len(tpl[0][0]), len(tpl[0][1])))
        right_blocks = self.r_m.block_order(adc_order)
        right_blocks = sorted(
            right_blocks, key=lambda bl: (len(bl[0]), len(bl[1]))
        )
        # iterate over the blocks, replacing the second space in each block
        # with the corresponding space of block_2 from isr_2
        # This only works for python3.7 or newer, because it assumes that
        # the two block dicts are in the same order -> which is only
        # garuanteed from python3.7
        res = sympify(0)
        for i, (l_block, max_order) in enumerate(left_blocks):
            r_block = right_blocks[i]
            # block is not expanded through the given order
            if order is not None and max_order < order:
                continue
            # combine the two spaces to build the correct block with mixed
            # ADC variant spaces.
            block = (l_block[0], r_block[1])
            if order is not None:
                res += self.op_block(order=order, block=block,
                                     opstring=opstring,
                                     subtract_gs=subtract_gs)
                continue
            for o in range(max_order + 1):
                res += self.op_block(order=o, block=block, opstring=opstring,
                                     subtract_gs=subtract_gs)
        return res

    @process_arguments
    @cached_member
    def trans_moment_space(self, order, space, opstring=None, lr_isr='left',
                           subtract_gs=True):
        """Computes d_pq... X_I <I|pq...|Psi_0>^(n).
           The transition operator may be defined via the operator string
           by the number of creation and annihilation operators it includes,
           e.g. 'a' defines a single annihilation operator (IP-ADC).
           If the class holds two different ISR, one may specify which ISR
           object should be used for the computation of the transition moment
           via the lr_isr argument, that takes either left or right.
           """
        # Subtraction of the ground state contribution is probably not
        # necessary, because all terms cancel (at least in second order
        # Singles PP-ADC). For all other ADC variants (IP/EA...) the ground
        # state expectation value is Zero, because the number of creation and
        # annihilation operators will never be equal.
        # Give the option anyway, because I'm not sure whether it will be
        # required at higher orders for PP-ADC

        space = transform_to_tuple(space)
        validate_input(order=order, space=space, lr_isr=lr_isr)
        if opstring is not None:
            validate_input(opstring=opstring)
        space = space[0]

        # - generate indices for the ISR state
        n_ov = n_ov_from_space(space)
        idx = self.indices.get_generic_indices(**n_ov)
        idx = "".join(extract_names(idx))

        # - map lr on the correct intermediate_states instance
        isr = {'left': self.l_isr, 'right': self.r_isr}
        isr = isr[lr_isr]

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
        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = sympify(0)
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # compute d for a given norm factor
            orders_d = gen_term_orders(
                order=norm_term[1], term_length=3, min_order=0
            )
            trans_mom = 0
            for term in orders_d:
                i1 = (pref * ampl *
                      isr.intermediate_state(order=term[0], space=space,
                                             braket='bra', indices=idx) *
                      self.operator(term[1], opstring, subtract_gs) *
                      mp[term[2]])
                trans_mom += wicks(i1, keep_only_fully_contracted=True,
                                   simplify_kronecker_deltas=True)
            res += (norm * trans_mom).expand()
        return simplify(res).sympy

    @process_arguments
    @cached_member
    def trans_moment(self, adc_order, opstring=None, order=None, lr_isr='left',
                     subtract_gs=True):
        """Computes sum_I sum_pq... d_pq... X_I <I|pq...|Psi_0>
           for ADC(n), taking all necessary excitation spaces I into
           account.
           The operator x may be defined via the operator string (e.g. 'ca'
           gives a one_particle operator for PP-ADC). If no operator is
           provided, the standard transition operator for the ADC variant will
           be constructed by default, i.e. 'a' for IP and 'c' for EA.
           If the parameter order is given, only contribution of the desired
           order are computed.
           The parameter lr_isr controls which isr of the possibly two
           available isr instances should be used.
           Subtract_gs controls, wether the ground state expectation value
           is subtracted or not."""

        # obtain the maximum order through which all the spaces are expanded
        # in the secular matrix
        m = {'left': self.l_m, 'right': self.r_m}
        m = m[lr_isr]
        max_orders = m.max_ptorder_spaces(adc_order)

        res = 0
        for space, max_order in max_orders.items():
            # the space is not expanded through the desired order
            if order is not None and max_order < order:
                continue
            if order is not None:
                res += self.trans_moment_space(order=order, space=space,
                                               opstring=opstring,
                                               lr_isr=lr_isr,
                                               subtract_gs=subtract_gs)
                continue
            for o in range(max_order + 1):
                res += self.trans_moment_space(order=o, space=space,
                                               opstring=opstring,
                                               lr_isr=lr_isr,
                                               subtract_gs=subtract_gs)
        return res
