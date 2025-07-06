from collections.abc import Sequence
from math import factorial

from sympy.physics.secondquant import NO, Dagger
from sympy import Expr, Mul, Rational, S, latex, nsimplify, diff, symbols

from .expression import ExprContainer
from .func import gen_term_orders, evaluate_deltas
from .groundstate import GroundState
from .indices import (
    n_ov_from_space, repeated_indices, Indices, generic_indices_from_space
)
from .logger import logger
from .misc import cached_member, Inputerror, transform_to_tuple, validate_input
from .simplify import simplify
from .sympy_objects import Amplitude
from .tensor_names import tensor_names
from .wicks import wicks


class IntermediateStates:
    """
    Class for constructing epxressions for Precursor or Intermediate states.

    Parameters
    ----------
    mp : GroundState
        Representation of the underlying ground state. Used to generate
        ground state related expressions.
    variant : str, optional
        The ADC variant for which Intermediates are constructed, e.g.,
        'pp', 'ip' or 'ea' for PP-, IP- or EA-ADC expressions, respectively
        (default: 'pp').
    """
    def __init__(self, mp: GroundState, variant: str = "pp"):
        assert isinstance(mp, GroundState)
        self.gs: GroundState = mp
        self.indices: Indices = Indices()

        variants: dict[str, tuple[str, ...]] = {
            "pp": ("ph", "hp"),
            "ea": ("p",),
            "ip": ("h",),
            "dip": ("hh",),
            "dea": ("pp",),
        }
        if variant not in variants.keys():
            raise Inputerror(f"The ADC variant {variant} is not valid. "
                             "Supported variants are "
                             f"{list(variants.keys())}.")
        self.variant: str = variant
        self.min_space: tuple[str, ...] = variants[variant]

    @cached_member
    def precursor(self, order: int, space: str, braket: str, indices: str
                  ) -> Expr:
        """
        Constructs expressions for precursor states.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        space : str
            The excitation space of the desired precursor state, e.g., 'ph' or
            'pphh' for singly or doubly excited precursor states.
        braket : str
            Defines whether a bra or ket precursor state is constructed.
        indices : str
            The indices of the precursor state.
        """

        # check input parameters
        indices_tpl = transform_to_tuple(indices)
        validate_input(order=order, space=space, braket=braket,
                       indices=indices_tpl)
        if len(indices_tpl) != 1:
            raise Inputerror(f"{indices} are not valid for constructing a "
                             "precursor state.")
        indices = indices_tpl[0]
        del indices_tpl
        # check that the space is valid for the given ADC variant
        if not self.validate_space(space):
            raise Inputerror(f"{space} is not a valid space for "
                             f"{self.variant} ADC.")

        # get the target symbols of the precursor state
        idx = self.indices.get_indices(indices)
        # check compatibility of indices and space
        if idx.get(("general", "")):
            raise Inputerror(f"The provided indices {indices} include a "
                             "general index.")
        n_ov = n_ov_from_space(space)
        occupied = idx.get(("occ", ""), [])
        virtual = idx.get(("virt", ""), [])
        if len(occupied) != n_ov["occ"] or len(virtual) != n_ov["virt"]:
            raise Inputerror(f"The indices {indices} and the space {space} "
                             "are not compatible.")
        del n_ov  # prevent accidentally usage below

        # in contrast to the gs, here the operators are ordered as
        # abij instead of abji in order to stay consistent with the
        # ADC literature.
        operators = self.gs.h.excitation_operator(
            creation=virtual, annihilation=occupied, reverse_annihilation=False
        )
        if braket == "bra":
            operators = Dagger(operators)

        res = S.Zero

        # leading term:
        # no need to differentiate bra/ket here, because
        # operators * mp = mp * operators (there is always an equal number of
        # p/h operators in mp that needs to be moved to the other side.
        # Will always give +.)
        max_gs = self.gs.psi(order=order, braket=braket)
        res += Mul(NO(operators), max_gs).expand()

        # get all terms of a*b of the desired order (ground state norm)
        orders = gen_term_orders(order=order, term_length=2, min_order=0)

        # orthogonalise with respect to the ground state for pp ADC.
        # checked up to 4th order!
        if self.variant == "pp":
            # import all ground state wave functions that may not appear twice
            # in |a><b|c>, i.e. all of: order > int(order/2)
            gs_psi: dict[str, dict[int, Expr]] = {'bra': {}, 'ket': {}}
            gs_psi[braket][order] = max_gs
            for o in range(order//2 + 1, order+1):
                if not gs_psi['bra'].get(o):
                    gs_psi['bra'][o] = self.gs.psi(order=o, braket='bra')
                if not gs_psi['ket'].get(o):
                    gs_psi['ket'][o] = self.gs.psi(order=o, braket='ket')

            def get_gs_wfn(o, bk): return gs_psi[bk][o] if o > order//2 else \
                self.gs.psi(order=o, braket=bk)

            # 1) iterate through all combinations of norm_factor*projector
            for norm_term in orders:
                norm = self.gs.norm_factor(norm_term[0])
                if norm is S.Zero:
                    continue
                # 2) construct the projector for a given norm_factor
                # the overall order is split between the norm_factor and the
                # projector
                orders_projection = gen_term_orders(
                    order=norm_term[1], term_length=3, min_order=0
                )
                projection = S.Zero
                for term in orders_projection:
                    # |Y>  <--  -|X><X|Y>
                    if braket == "ket":
                        i1 = Mul(
                            get_gs_wfn(term[1], 'bra'), NO(operators),
                            get_gs_wfn(term[2], 'ket')
                        )
                        state = get_gs_wfn(term[0], 'ket')
                    # <Y|  <--  -<Y|X><X|
                    else:
                        assert braket == "bra"
                        i1 = Mul(
                            get_gs_wfn(term[0], 'bra'), NO(operators),
                            get_gs_wfn(term[1], 'ket')
                        )
                        state = get_gs_wfn(term[2], 'bra')
                    # wicks automatically expands the passed expression
                    i1 = wicks(i1, simplify_kronecker_deltas=True)
                    projection += Mul(state, i1).expand()
                projection = evaluate_deltas(projection)
                res -= Mul(norm, projection).expand()
            gs_psi.clear()

        # iterate over lower excitated spaces
        lower_spaces = self._generate_lower_spaces(space)
        for lower_space in lower_spaces:
            # get generic unique indices to generate the lower_isr_states.
            idx_isr: str = "".join(
                s.name for s in generic_indices_from_space(lower_space)
            )

            # prefactor due to the sum - sum_J |J><J|I>
            n_ov = n_ov_from_space(lower_space)
            prefactor = Rational(
                1, factorial(n_ov["occ"]) * factorial(n_ov["virt"])
            )
            del n_ov

            # orthogonalise with respsect to the lower excited ISR state
            # 1) iterate through all combinations of norm_factor*projector
            for norm_term in orders:
                norm = self.gs.norm_factor(norm_term[0])
                if norm is S.Zero:
                    continue
                # 2) construct the projector for a given norm factor
                # the overall order is split between he norm_factor and the
                # projector
                orders_projection = gen_term_orders(
                    norm_term[1], term_length=3, min_order=0
                )
                projection = S.Zero
                for term in orders_projection:
                    # |Y#>  <--  -|X><X|Y>
                    if braket == "ket":
                        i1 = Mul(
                            self.intermediate_state(order=term[1],
                                                    space=lower_space,
                                                    braket="bra",
                                                    indices=idx_isr),
                            NO(operators),
                            self.gs.psi(order=term[2], braket="ket")
                        )
                        state = self.intermediate_state(
                            order=term[0], space=lower_space, braket="ket",
                            indices=idx_isr
                        )
                    # <Y#|  <--  -<Y|X><X|
                    else:
                        assert braket == "bra"
                        i1 = Mul(
                            self.gs.psi(order=term[0], braket="bra"),
                            NO(operators),
                            self.intermediate_state(order=term[1],
                                                    space=lower_space,
                                                    braket="ket",
                                                    indices=idx_isr)
                        )
                        state = self.intermediate_state(
                            order=term[2], space=lower_space, braket="bra",
                            indices=idx_isr
                        )
                    i1 = wicks(i1, simplify_kronecker_deltas=True)
                    projection += (prefactor * state * i1).expand()
                projection = evaluate_deltas(projection)
                res -= Mul(norm, projection).expand()
        assert isinstance(res, Expr)
        logger.debug(f"precursor {space}_({indices})^({order}) {braket} = "
                     f"{latex(res)}")
        return res

    @cached_member
    def overlap_precursor(self, order: int, block: Sequence[str],
                          indices: Sequence[str]) -> Expr:
        """
        Constructs expressions for elements of the overlap matrix of the
        precursor states.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        block : Sequence[str]
            The block of the overlap matrix, e.g., 'ph,ph' for an element of
            the 1p-1h/1p-1h block.
        indices : Sequence[str]
            The indices of the overlap matrix element, e.g., 'ia,jb' for
            S_{ia,jb}.
        """

        # no need to do more validation here -> will be done in precursor
        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("2 index strings required for an overlap matrix "
                             f"block. Got: {indices}.")

        if repeated_indices(indices[0], indices[1]):
            raise Inputerror("Repeated index found in indices of precursor "
                             f"overlap matrix: {indices}.")

        orders = gen_term_orders(order=order, term_length=2, min_order=0)

        res = S.Zero
        # 1) iterate through all combinations of norm_factor*S
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct S for a given norm factor
            # the overall order is split between the norm_factor and S
            orders_overlap = gen_term_orders(
                order=norm_term[1], term_length=2, min_order=0
            )
            overlap = S.Zero
            for term in orders_overlap:
                i1 = Mul(
                    self.precursor(order=term[0], space=block[0],
                                   braket="bra", indices=indices[0]),
                    self.precursor(order=term[1], space=block[1],
                                   braket="ket", indices=indices[1])
                )
                i1 = wicks(i1, simplify_kronecker_deltas=True)
                overlap += i1
            res += (norm * overlap).expand()
        # It should be valid to simplifiy the result by permuting contracted
        # indices before returning -> should lower the overall size of the
        # final expression
        res = simplify(ExprContainer(res))
        logger.debug(f"overlap {block} S_{indices}^({order}) = {res}")
        return res.inner

    @cached_member
    def s_root(self, order: int, block: Sequence[str],
               indices: Sequence[str]) -> Expr:
        """
        Constructs expression for elements of the inverse square root of the
        precursor overlap matrix (S^{-0.5})_{I,J} by expanding
        S^{-0.5} in a Taylor series.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        block : Sequence[str]
            The desired matrix block, e.g., 'ph,pphh' for an element of the
            1p-1h/2p-2h block.
        indices : Sequence[str]
            The indices of the matrix element, e.g., 'ia,jkcd' for
            (S^{-0.5})_{ia,jkcd}.
        """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("2 index strings required for a block of the "
                             "inverse suqare root of the overlap matrix. "
                             f"Got: {indices}.")
        if repeated_indices(indices[0], indices[1]):
            raise Inputerror(f"Repeated index found in indices {indices}.")
        if block[0] != block[1]:
            raise NotImplementedError("Off diagonal blocks of the overlap "
                                      "matrix should be 0 by definition. "
                                      "Simply don't know how to handle the "
                                      "index generation needed in this case.")

        taylor_expansion = self.expand_S_taylor(order, min_order=2)
        # create an index list: first and last element are the two provided
        # idx strings
        idx: list[str] = list(indices)
        # create more indices: exponent-1 or len(taylor_expansion)-1 indices
        #  - x*x 1 additional index 'pair' is required: I,I' = I,I'' * I'',I'
        #  - x^3: I,I' = I,I'' * I'',I''' * I''',I'
        for _ in range(len(taylor_expansion) - 1):
            new_idx: str = "".join(
                s.name for s in generic_indices_from_space(block[0])
            )
            idx.insert(-1, new_idx)
        # iterate over exponents and terms, starting with the lowest exponent
        res = S.Zero
        for pref, termlist in taylor_expansion:
            # all terms in the list should have the same length, i.e.
            # all originate from x*x or x^3 etc.
            for term in termlist:
                relevant_idx = idx[:len(term)] + [idx[-1]]
                i1 = S.One * pref
                for o in term:
                    i1 *= self.overlap_precursor(
                        order=o, block=block,
                        indices=tuple(relevant_idx[:2])
                    )
                    del relevant_idx[0]
                    if i1 is S.Zero:
                        break
                assert (
                    len(relevant_idx) == 1 and
                    relevant_idx[0] == indices[1]
                )
                # in squared or higher terms S*S*... delta evaluation might
                # be necessary
                res += evaluate_deltas(i1.expand())
        assert isinstance(res, Expr)
        logger.debug(
            f"{block} S_root_{indices}^({order}) = {latex(res)}"
        )
        return res

    @cached_member
    def intermediate_state(self, order: int, space: str, braket: str,
                           indices: str) -> Expr:
        """
        Constructs expressions for intermediate states.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        space : str
            The excitation space of the desired intermediate state, e.g.,
            'ph' and 'pphh' for singly and doubly excited intermediate states.
        braket : str
            Defines whether a bra or ket intermediate state is constructed.
        indices : str
            The indices of the intermediate state.
        """
        indices_tpl: tuple[str, ...] = transform_to_tuple(indices)
        validate_input(order=order, space=space, braket=braket,
                       indices=indices_tpl)
        if len(indices_tpl) != 1:
            raise Inputerror(f"{indices} are not valid for "
                             "constructing an intermediate state.")
        indices = indices_tpl[0]
        del indices_tpl

        # generate additional indices for the precursor state
        idx_pre: str = "".join(
            s.name for s in generic_indices_from_space(space)
        )

        n_ov = n_ov_from_space(space)
        prefactor = Rational(
            1, factorial(n_ov["occ"]) * factorial(n_ov["virt"])
        )
        del n_ov

        # sandwich the IS and precursor indices together
        s_indices = {
            'bra': ",".join([indices, idx_pre]),
            'ket': ",".join([idx_pre, indices])
        }

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = S.Zero
        for term in orders:
            i1 = Mul(
                prefactor,
                self.s_root(order=term[0], block=(space, space),
                            indices=s_indices[braket]),
                self.precursor(order=term[1], space=space, braket=braket,
                               indices=idx_pre)
            )
            res += evaluate_deltas(i1.expand())
        assert isinstance(res, Expr)
        logger.debug(f"{space} ISR_({indices}^({order}) {braket} = "
                     f"{latex(res)}")
        return res

    @cached_member
    def overlap_isr(self, order: int, block: Sequence[str],
                    indices: Sequence[str]) -> Expr:
        """
        Computes a block of the overlap matrix in the basis of intermediate
        states.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        block : Sequence[str]
            The desired matrix block.
        indices : Sequence[str]
            The indices of the matrix element.
        """

        block = transform_to_tuple(block)
        indices = transform_to_tuple(indices)
        validate_input(order=order, block=block, indices=indices)
        if len(indices) != 2:
            raise Inputerror("Constructing a ISR overlap matrix block requires"
                             f" 2 index strings. Provided: {indices}.")

        orders = gen_term_orders(order=order, term_length=2, min_order=0)
        res = S.Zero
        # 1) iterate through all combinations of norm_factor*S
        for norm_term in orders:
            norm = self.gs.norm_factor(norm_term[0])
            if norm is S.Zero:
                continue
            # 2) construct S for a given norm factor
            # the overall order is split between he norm_factor and S
            orders_overlap = gen_term_orders(
                order=norm_term[1], term_length=2, min_order=0
            )
            overlap = S.Zero
            for term in orders_overlap:
                i1 = Mul(
                    self.intermediate_state(order=term[0], space=block[0],
                                            braket="bra",
                                            indices=indices[0]),
                    self.intermediate_state(order=term[1], space=block[1],
                                            braket="ket",
                                            indices=indices[1])
                )
                i1 = wicks(i1, simplify_kronecker_deltas=True)
                overlap += i1
            res += (norm * overlap).expand()
        assert isinstance(res, Expr)
        logger.debug(f"ISR overlap {block} S_{indices}^({order}) = "
                     f"{latex(res)}")
        return res

    @cached_member
    def amplitude_vector(self, indices: str, lr: str = "right") -> Expr:
        """
        Constructs an amplitude vector with the provided indices.

        Parameters
        ----------
        indices : str
            The indices of the amplitude vector.
        lr : str, optional
            Whether a left (X) or right (Y) amplitude vector is constructed
            (default: 'right').
        """

        validate_input(indices=indices, lr=lr)

        idx = self.indices.get_indices(indices)
        occ = idx.get(("occ", ""), [])
        virt = idx.get(("virt", ""), [])

        name = getattr(tensor_names, f"{lr}_adc_amplitude")
        return Amplitude(name, virt, occ)

    def expand_S_taylor(self, order: int, min_order: int = 2
                        ) -> list[tuple[Expr, list[tuple[int, ...]]]]:
        """
        Performs a Taylor expansion of the inverse square root of the
        overlap matrix
        S^{0.5} = (1 + x)^{-0.5} with x = sum_{n=1} S^(n)
        returning all n'th-order contributions.

        Parameters
        ----------
        order : int
            The perturbation theoretical order.
        min_order : int, optional
            The lowest order at which the overlap matrix S has a non-vanishing
            caontribution excluding the zeroth order contribution
            (default: 2).

        Returns
        -------
        list
            Iterable containing tuples of prefactors and perturbation
            theoretical orders, for instance, with a min_order of 2 the
            5'th order contributions read
            [(-1/2, [(5,)]), (3/8, [(2, 3), (3, 2)])].
        """
        validate_input(order=order, min_order=min_order)
        if min_order == 0:
            raise Inputerror("A minimum order of 0 does not make sense here.")

        # below min_order all orders - except the zeroth order contribution -
        # should be zero. Should be handled automatically if the corresponding
        # orders are forwarded to the overlap method.
        if order < min_order:
            return [(S.One, [(order,)])]

        x = symbols('x')
        f = (1 + x) ** -0.5
        ret: list[tuple[Expr, list[tuple[int, ...]]]] = []
        for exp in range(1, order//min_order + 1):
            f = diff(f, x)
            pref = nsimplify(
                f.subs(x, 0) * S.One / factorial(exp), rational=True
            )
            orders = gen_term_orders(
                order=order, term_length=exp, min_order=min_order
            )
            assert isinstance(pref, Expr)
            ret.append((pref, orders))
        return ret

    def _generate_lower_spaces(self, space_str: str) -> list[str]:
        """
        Generates all strings of lower excited configurations for a given
        excitation space.

        Parameters
        ----------
        space_str : str
            The space for which to construct lower excitation spaces, e.g.,
            ['ph'] for 'pphh'.
        """
        lower_spaces: list[str] = []
        for _ in range(min(space_str.count('p'), space_str.count('h'))):
            space_str = space_str.replace('p', '', 1).replace('h', '', 1)
            if not space_str:
                break
            lower_spaces.append(space_str)
        return lower_spaces

    def validate_space(self, space_str: str) -> bool:
        """
        Checks whether the given space is valid for the current ADC variant.

        Parameters
        ----------
        space_str : str
            The excitation space to validate.
        """

        if space_str in self.min_space:
            return True

        lower_spaces = self._generate_lower_spaces(space_str)
        return any(sp in self.min_space for sp in lower_spaces)
