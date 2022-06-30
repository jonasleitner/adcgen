from sympy import sqrt, S
from sympy.physics.secondquant import wicks
from math import factorial

from isr import get_orders_three, get_order_two
from indices import n_ov_from_space, indices
from misc import Inputerror, cached_member, transform_to_tuple
from secular_matrix import secular_matrix
from simplify import simplify


class properties:
    def __init__(self, isr):
        self.m = secular_matrix(isr)
        self.isr = isr
        self.variant = isr.variant
        self.gs = isr.gs
        self.h = isr.gs.h
        self.indices = indices()

    def __transition_operator(self, order):
        operator = {
            "pp": self.__shifted_one_particle_op,
            # D_0 = <Psi|D|Psi> = 0 for ip, ea and dip -> no shift
            "ip": lambda o: self.h.ip_transition if o == 0 else 0,
            "ea": lambda o: self.h.ea_transition if o == 0 else 0,
            "dip": lambda o: self.h.dip_transition if o == 0 else 0,
        }
        op = operator.get(self.variant, None)
        if op is not None:
            return op(order)
        else:
            raise KeyError(f"No operator provided for {self.variant} ADC "
                           "transition moments. Operators available for "
                           f"{tuple(operator.keys())}.")

    def __shifted_one_particle_op(self, order):
        d = self.h.one_particle if order == 0 else 0
        return d - self.gs.one_particle_operator(order)

    def __shifted_two_particle_op(self, order):
        d = self.h.two_particle if order == 0 else 0
        return d - self.gs.two_particle_operator(order)

    @cached_member
    def one_particle_block(self, order, block, indices):
        """Computes sum_pq d_{pq} X_I <I|pq|J>^(n) Y_J.
           (also includes the summation over the amplitude corefficients)
           Results checked for pp-ADC(2)!
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        if len(block) != 2 or len(indices) != 2:
            raise Inputerror("2 space and index strings required."
                             f"Provided: {block} / {indices}.")

        left = self.isr.amplitude_vector(indices=indices[0], lr="left")
        right = self.isr.amplitude_vector(indices=indices[1], lr="right")
        # again not use the full prefactors from lifting the sum restrictions,
        # but sqrt(1/(no! * nv!)) to keep the left and right amplitude vectors
        # normalized.
        n_ov = n_ov_from_space(block[0])
        prefactor_l = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )
        n_ov = n_ov_from_space(block[1])
        prefactor_r = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        orders = get_order_two(order)
        res = 0
        # iterate over all norm*d combinations of n'th order
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # compute d for a given norm (the total order is split inbetween
            # both factors)
            orders_d = get_orders_three(norm_term[1])
            density = 0
            for term in orders_d:
                i1 = (prefactor_l * prefactor_r * left *
                      self.isr.intermediate_state(term[0], space=block[0],
                                                  braket="bra",
                                                  indices=indices[0]) *
                      self.__shifted_one_particle_op(term[1]) *
                      self.isr.intermediate_state(term[2], space=block[1],
                                                  braket="ket",
                                                  indices=indices[1]) *
                      right)
                i1 = wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
                density += i1
            res += (norm * density).expand()
        return simplify(res).sympy

    @cached_member
    def one_particle_operator(self, adc_order, order=None):
        """Computes: sum_I,J sum_pq d_pq X_I <I|pq|J>^(n) Y_J
           (the excited state contribution for the expectation value of
           a general one particle operator.
           adc_order specifies the which block I,J are present and defines
           their maximum order.
           If the optional parameter order is given, only terms of
           the desired order are returned, e.g. only the zeroth order terms
           of the ADC(2) matrix blocks.
           """

        # get the maximum order each block is expanded in the ADC(n)
        # secular matrix
        blocks = self.m.block_order(adc_order)
        res = 0
        for b, max_order in blocks.items():
            # skip all blocks that are not expanded up to the requested
            # order - if the parameter is given
            if order is not None and max_order < order:
                continue

            # get indices for the two blocks
            # TODO: create a function for this
            n_ov = n_ov_from_space(b[0])
            sym = self.indices.get_generic_indices(**n_ov)
            idx0 = [s.name for s_list in sym.values() for s in s_list]
            n_ov = n_ov_from_space(b[1])
            sym = self.indices.get_generic_indices(**n_ov)
            idx1 = [s.name for s_list in sym.values() for s in s_list]
            idx0 = "".join(idx0)
            idx1 = "".join(idx1)

            if order is not None:
                res += self.one_particle_block(order, b, idx0 + "," + idx1)
            else:
                for o in range(max_order + 1):
                    res += self.one_particle_block(o, b, idx0 + "," + idx1)
        return res

    @cached_member
    def two_particle_block(self, order, block, indices):
        """Computes 1/4 sum_pqrs d_{pqrs} <I|pqsr|J>^(n).
           Checked the zeroth order results for all blocks
           that are present in the pp- and ip-ADC(2) matrix!
           """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        if len(block) != 2 or len(indices) != 2:
            raise Inputerror("2 space and index strings required."
                             f"Provided: {block} / {indices}.")

        left = self.isr.amplitude_vector(indices=indices[0], lr="left")
        right = self.isr.amplitude_vector(indices=indices[1], lr="right")
        # again not use the full prefactors from lifting the sum restrictions,
        # but sqrt(1/(no! * nv!)) to keep the left and right amplitude vectors
        # normalized.
        n_ov = n_ov_from_space(block[0])
        prefactor_l = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )
        n_ov = n_ov_from_space(block[1])
        prefactor_r = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )

        orders = get_order_two(order)
        res = 0
        # iterate over all norm*d combinations of n'th order
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # compute d for a given norm factor (the total order is split
            # between norm and d)
            orders_d = get_orders_three(norm_term[1])
            density = 0
            for term in orders_d:
                i1 = (prefactor_l * prefactor_r * left *
                      self.isr.intermediate_state(term[0], space=block[0],
                                                  braket="bra",
                                                  indices=indices[0]) *
                      self.__shifted_two_particle_op(term[1]) *
                      self.isr.intermediate_state(term[2], space=block[1],
                                                  braket="ket",
                                                  indices=indices[1]) *
                      right)
                i1 = wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
                density += i1
            res += (norm * density).expand()
        return simplify(res).sympy

    @cached_member
    def two_particle_operator(self, adc_order, order=None):
        """Computes: sum_I,J 1/4 sum_pqrs d_pqrs X_I <I|pqsr|J>^(n) Y_J
           (the excited state contribution for the expectation value of
           a general two particle operator.
           adc_order specifies the which block I,J are present and defines
           their maximum order.
           If the optional parameter order is given, only terms of
           the desired order are returned, e.g. only the zeroth order terms
           of the ADC(2) matrix blocks.
           """

        # get the maximum order each block is expanded in the ADC(n)
        # secular matrix
        blocks = self.m.block_order(adc_order)
        res = 0
        for b, max_order in blocks.items():
            # skip all blocks that are not expanded up to the requested
            # order - if the parameter is given
            if order is not None and max_order < order:
                continue

            # get indices for the two blocks
            n_ov = n_ov_from_space(b[0])
            sym = self.indices.get_generic_indices(**n_ov)
            idx0 = [s.name for s_list in sym.values() for s in s_list]
            n_ov = n_ov_from_space(b[1])
            sym = self.indices.get_generic_indices(**n_ov)
            idx1 = [s.name for s_list in sym.values() for s in s_list]
            idx0 = "".join(idx0)
            idx1 = "".join(idx1)

            if order is not None:
                res += self.two_particle_block(order, b, idx0 + "," + idx1)
            else:
                for o in range(max_order + 1):
                    res += self.two_particle_block(o, b, idx0 + "," + idx1)
        return res

    def transition_moment_space(self, order, space, indices):
        """Computes sum_x X_I <I|x|Psi_0>^(n),
           where x is the appropriate transition operator for the
           ADC variant, i.e. p+q for pp-ADC, p for ip-ADC and p+ for ea-ADC.

           Checked transition moments for pp-ADC(2)!
           """

        space = transform_to_tuple(space)
        indices = transform_to_tuple(indices)
        if len(space) != 1 or len(indices) != 1:
            raise Inputerror("1 space and index string required."
                             f"Provided: {space} / {indices}.")
        space = space[0]
        indices = indices[0]

        # import left amplitude vector
        amplitude = self.isr.amplitude_vector(indices=indices, lr="left")
        n_ov = n_ov_from_space(space)
        prefactor = 1 / sqrt(
            factorial(n_ov["n_occ"]) * factorial(n_ov["n_virt"])
        )
        # import the ground state wavefunction to save some indices
        mp = {}
        for o in range(order + 1):
            mp[o] = self.gs.psi(o, 'ket')

        orders = get_order_two(order)
        res = 0
        # iterate over all norm*d combinations of n'th order
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # compute d for a given norm factor (the overall order is split
            # between the two factors)
            orders_d = get_orders_three(norm_term[1])
            transition_d = 0
            for term in orders_d:
                i1 = (prefactor * amplitude *
                      self.isr.intermediate_state(term[0], space=space,
                                                  braket="bra",
                                                  indices=indices) *
                      self.__transition_operator(term[1]) *
                      mp[term[2]])
                i1 = wicks(i1, keep_only_fully_contracted=True,
                           simplify_kronecker_deltas=True)
                transition_d += i1
            res += (norm * transition_d).expand()
        return simplify(res).sympy

    def transition_moment(self, adc_order, order=None):
        """Computes sum_I sum_x <I|x|Psi>
           through the appropriate order for ADC(n).
           Alternatively the order argument may be used to compute
           all n-th order contributions to the transition moments
           that are present in ADC(n').
           """

        # get the maximum oder to which a space is expanded
        # in the ADC(n) matrix
        max_orders = self.m.max_ptorder_spaces(adc_order)
        res = 0
        for space, max_order in max_orders.items():
            # skip all spaces that are not expanded through the
            # requested order - if a order is requested
            if order is not None and max_order < order:
                continue

            # get indices for the space
            n_ov = n_ov_from_space(space)
            sym = self.indices.get_generic_indices(**n_ov)
            idx = [s.name for s_list in sym.values() for s in s_list]
            idx = "".join(idx)

            if order is not None:
                res += self.transition_moment_space(order, space, idx)
            else:
                for o in range(max_order + 1):
                    res += self.transition_moment_space(o, space, idx)
        return res
