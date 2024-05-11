from .func import gen_term_orders, wicks
from .indices import extract_names, Indices, n_ov_from_space
from .misc import Inputerror, cached_member, transform_to_tuple, validate_input
from .simplify import simplify
from .expr_container import Expr
from .rules import Rules
from sympy import sqrt, S, sympify
from math import factorial


class Properties:
    """Class for computing ISR properties. The class takes two different ISR
       as arguments - the second is optional. If only one is given, the
       standard ISR properties may be computed by calling the appropriate
       methods. This may also be achieved if the same ISR is provided twice.
       If both ISR are given, the expectation values of operators between
       the two ISR states like pp and ip will be calculated. This way one may
       calculate the expectation values between e.g. a EA and a IP ISR state.
       """

    def __init__(self, l_isr, r_isr=None):
        from .isr import IntermediateStates
        from .secular_matrix import SecularMatrix

        if not isinstance(l_isr, IntermediateStates) or r_isr is not None \
                and not isinstance(r_isr, IntermediateStates):
            raise Inputerror("Provided ISR must be of type "
                             f"{IntermediateStates}.")
        self.l_isr = l_isr
        self.r_isr = self.l_isr if r_isr is None else r_isr
        self.l_m = SecularMatrix(l_isr)
        self.r_m = self.l_m if r_isr is None else SecularMatrix(r_isr)
        # Check if both ground states are compatible. Currently this only means
        # to check that either None ot both have singles enabled.
        self.gs = l_isr.gs
        if self.gs.singles != self.r_isr.gs.singles:
            raise Inputerror("Both ISR need to share the same GS, "
                             "i.e. neither or both have singles enabled.")
        # also check that both isr use the same hamiltonian
        if self.l_isr.gs.h != self.r_isr.gs.h:
            raise Inputerror("The Operator of left and right isr has to be "
                             "equal")
        self.h = l_isr.gs.h
        self.indices = Indices()

    def operator(self, order: int, n_create: int, n_annihilate: int,
                 subtract_gs=True):
        """
        Constructs an arbitrary n'th-order operator.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        n_create : int
            The number of creation operators. Placed left of the annihilation
            operators.
        n_annihilate : int
            The number of annihilation operators. Placed right of the
            creation operators.
        subtract_gs : bool, optional
            If set, the ground state expectation value of the corresponding
            operator is subtracted if the operator string contains an equal
            amount of creation and annihilation operators (otherwise the
            ground state contribution vanishes).
            (Defaults to True)
        """
        validate_input(order=order)

        if order == 0:
            d, rules = self.h.operator(n_create=n_create,
                                       n_annihilate=n_annihilate)
        else:
            d, rules = sympify(0), Rules()

        if subtract_gs and n_create == n_annihilate:
            e0 = self.gs.expectation_value(order=order, n_particles=n_create)
            return d - e0, rules
        else:
            return d, rules

    @cached_member
    def expec_block_contribution(self, order: int, block: str,
                                 n_particles: int = 1,
                                 subtract_gs: bool = True):
        """
        Constructs the n'th order contribution of an individual block IJ to the
        expectation value of the operator
        d_{pq...} X_I <I|pq...|J>^(n) Y_J.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        block : str
            The block of the ADC matrix for which the expectation value
            is generated, e.g., 'ph,pphh' for the 1p-1h/2p-2h block.
        n_particles : int
            The number of creation and annihilation operators in the operator
            string. (Defaults to 1.)
        subtract_gs : bool, optional
            If set, the ground state expectation value of the corresponding
            operator is subtracted from the result. (Defaults to True)
        """

        block = transform_to_tuple(block)
        validate_input(order=order, block=block)

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
                op, rules = self.operator(order=term[1],
                                          n_create=n_particles,
                                          n_annihilate=n_particles,
                                          subtract_gs=subtract_gs)
                if op is S.Zero:
                    continue
                i1 = (left_pref * right_pref * left_ampl *
                      self.l_isr.intermediate_state(order=term[0],
                                                    space=block[0],
                                                    braket='bra',
                                                    indices=left_idx) *
                      op *
                      self.r_isr.intermediate_state(order=term[2],
                                                    space=block[1],
                                                    braket='ket',
                                                    indices=right_idx) *
                      right_ampl)
                expec += wicks(i1, simplify_kronecker_deltas=True, rules=rules)
            res += (norm * expec).expand()
        return simplify(Expr(res)).sympy

    @cached_member
    def expectation_value(self, adc_order: int, n_particles: int = 1,
                          order: int = None, subtract_gs: bool = True):
        """
        Constructs the expectation value taking all blocks into account
        that are available at the specified order of perturbation theory
        in the ADC secular matrix
        sum_IJ sum_pq... d_{pq...} X_I <I|pq...|J> Y_J.
        Note that also lower order contributions are considered, i.e., the
        ADC(0) and ADC(1) expectation values are included in the ADC(2)
        expectation value.

        Parameters
        ----------
        adc_order : int
            The perturbation theoretical order of the ADC scheme for which the
            expectation value is generated.
        n_particles : int
            The number of creation and annihilation operators in the operator
            string. (Defaults to 1)
        order : int, optional
            Only consider contributions of the specified order, e.g.,
            only the zeroth order contributions of all available blocks in
            the ADC(n) matrix.
        subtract_gs : bool, optional
            If set, the ground state expectation value is subtracted from the
            result. (Defaults to True)
        """
        validate_input(adc_order=adc_order)
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
            if order is None:
                for o in range(max_order + 1):
                    res += self.expec_block_contribution(
                        order=o, block=block, n_particles=n_particles,
                        subtract_gs=subtract_gs
                    )
            else:
                res += self.expec_block_contribution(
                    order=order, block=block, n_particles=n_particles,
                    subtract_gs=subtract_gs
                )
        return res

    @cached_member
    def trans_moment_space(self, order: int, space: str, n_create: int = None,
                           n_annihilate: int = None, lr_isr: str = 'left',
                           subtract_gs: bool = True):
        """
        Constructs the n'th-order contribution to the transition moment
        for the desired excitation space and operator
        d_pq... X_I <I|pq...|Psi_0>^(n).

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        space : str
            The excitation space, e.g., 'ph' or 'pphh' for singly or doubly
            excited configurations, respectively.
        n_create : int, optional
            The number of creation operators in the operator string.
            By default, the operator string with the lowest amount of
            creation and annihilation operators is constructed for which in
            general a non-zero result can be expected, e.g., 'ca' and 'a'
            for PP- and IP-ADC, respectively.
        n_annihilate : int, optional
            The number of annihilation operators in the operator string.
            By default, the operator string with the lowest amount of
            creation and annihilation operators is constructed for which in
            general a non-zero result can be expected, e.g., 'ca' and 'a'
            for PP- and IP-ADC, respectively.
        l_isr : str, optional
            Controls whether the left or right 'IntermediateStates' instance
            is used to construct the transition moment contribution.
            (Defaults to 'left')
        subtract_gs : bool, optional
            If set, ground state contributions are subtracted if the
            operator contains an equal amount of creation and annihilation
            operators. (Defaults to True)
        """
        # Subtraction of the ground state contribution is probably not
        # necessary, because all terms cancel (at least in second order
        # Singles PP-ADC). For all other ADC variants (IP/EA...) the ground
        # state expectation value is Zero, because the number of creation and
        # annihilation operators will never be equal.
        # Give the option anyway, because I'm not sure whether it will be
        # required at higher orders for PP-ADC

        validate_input(order=order, space=space, lr_isr=lr_isr)

        # - generate indices for the ISR state
        n_ov = n_ov_from_space(space)
        idx = self.indices.get_generic_indices(**n_ov)
        idx = "".join(extract_names(idx))

        # - map lr on the correct intermediate_states instance
        isr = {'left': self.l_isr, 'right': self.r_isr}
        isr = isr[lr_isr]

        # - if no operator string is given -> generate a default, i.e.
        #   'a' for IP- / 'ca' for PP-ADC
        if n_create is None and n_annihilate is None:
            n_create = isr.min_space[0].count('p')
            n_annihilate = isr.min_space[0].count('h')
        elif n_create is None:
            n_create = 0
        elif n_annihilate is None:
            n_annihilate = 0

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
                op, rules = self.operator(order=term[1], n_create=n_create,
                                          n_annihilate=n_annihilate,
                                          subtract_gs=subtract_gs)
                if op is S.Zero:
                    continue
                i1 = (pref * ampl *
                      isr.intermediate_state(order=term[0], space=space,
                                             braket='bra', indices=idx) *
                      op *
                      mp[term[2]])
                trans_mom += wicks(i1, simplify_kronecker_deltas=True,
                                   rules=rules)
            res += (norm * trans_mom).expand()
        return simplify(Expr(res)).sympy

    @cached_member
    def trans_moment(self, adc_order: int, n_create: int = None,
                     n_annihilate: int = None, order: int = None,
                     lr_isr: str = 'left', subtract_gs: bool = True):
        """
        Constructs the ADC(n) transition moment
        sum_I sum_pq... d_pq... X_I <I|pq...|Psi_0>
        considering all available configurations.
        Note that also lower order contributions are considered, i.e.,
        the ADC(0) and ADC(1) contributions are included in the ADC(2)
        transition moments.

        Parameters
        ----------
        adc_order : int
            The perturbation theoretical order of the ADC scheme.
        n_create : int, optional
            The number of creation operators in the operator string.
            By default, the operator string with the lowest amount of
            creation and annihilation operators is constructed for which in
            general a non-zero result can be expected, e.g., 'ca' and 'a'
            for PP- and IP-ADC, respectively.
        n_annihilate : int, optional
            The number of annihilation operators in the operator string.
            By default, the operator string with the lowest amount of
            creation and annihilation operators is constructed for which in
            general a non-zero result can be expected, e.g., 'ca' and 'a'
            for PP- and IP-ADC, respectively.
        order : int, optional
            Only consider contributions of the specified order, e.g.,
            only the zeroth order contributions of all available configurations
            in the ADC(n) matrix.
        lr_isr : str, optional
            Constrols whether the left or right 'IntermediateStates' instance
            is used to construct the transition moment.
            (Defaults to 'left')
        subtract_gs : bool, optional
            If set, the ground state contributions are subtracted if the
            operator contains an equal amount of creation and annihilation
            operators. (Defaults to True)
        """

        # obtain the maximum order through which all the spaces are expanded
        # in the secular matrix
        m = {'left': self.l_m, 'right': self.r_m}
        m = m[lr_isr]
        max_orders = m.max_ptorder_spaces(adc_order)

        res = 0
        for space, max_order in max_orders.items():
            if order is None:
                for o in range(max_order + 1):
                    res += self.trans_moment_space(order=o, space=space,
                                                   n_create=n_create,
                                                   n_annihilate=n_annihilate,
                                                   lr_isr=lr_isr,
                                                   subtract_gs=subtract_gs)
            else:
                # the space is not expanded through the desired order
                if max_order < order:
                    continue
                res += self.trans_moment_space(order=order, space=space,
                                               n_create=n_create,
                                               n_annihilate=n_annihilate,
                                               lr_isr=lr_isr,
                                               subtract_gs=subtract_gs)
        return res
