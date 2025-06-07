from collections.abc import Callable
import re

from sympy.physics.secondquant import F, Fd, NO
from sympy import Expr, Pow, S, Symbol, sqrt, sympify

from ..indices import Index, get_symbols, split_idx_string
from ..sympy_objects import (
    Amplitude, AntiSymmetricTensor, KroneckerDelta, NonSymmetricTensor,
    SymmetricTensor
)
from ..tensor_names import tensor_names, is_adc_amplitude, is_t_amplitude
from .expr_container import ExprContainer


def import_from_sympy_latex(
        expr_string: str, convert_default_names: bool = False,
        is_amplitude: Callable[[str], bool] | None = None,
        is_symmetric_tensor: Callable[[str], bool] | None = None
        ) -> ExprContainer:
    """
    Imports an expression from a string created by the
    :py:func:`sympy.latex` function.

    Parameters
    ----------
    convert_default_names : bool, optional
        If set, all default tensor names found in the expression to import
        will be converted to the currently configured names.
        For instance, ERIs named 'V' by default will be renamed to
        whatever :py:attr:`TensorNames.eri` defines.
    is_amplitude: callable, optional
        A callable that takes a tensor name and returns whether a
        tensor with the corresponding name should be imported
        as :py:class:`Amplitude`.
        Note that this is checked after the (optional) conversion
        of default names, i.e., tensors named 't' (default name
        for ground state amplitudes) will first be converted to
        :py:attr:`TensorNames.gs_amplitude` before
        consulting the callable.
        Defaults to :py:func:`is_amplitude` defined below.
    is_symmetric_tensor: callable, optional
        A callable that takes a tensor name and returns whether a
        tensor with the corresponding name should be imported
        as :py:class:`SymmetricTensor`.
        Note that this is checked after the (optional) conversion
        of default names and after checking whether the
        tensor should be imported as :py:class:`Amplitude`.
        Tensors (with upper and lower indices) that are not
        identified as :py:class:`Amplitude` or
        :py:class:`SymmetricTensor` will finally be imported as
        :py:class:`AntiSymmetricTensor`.
        Defaults to :py:func:`is_symmetric_tensor` defined below.

    Returns
    -------
    ExprContainer
        The imported expression in a :py:class:`ExprContainer` container.
        Note that bra-ket symmetry is not set during the import.
    """
    if is_amplitude is None:
        is_amplitude = globals()["is_amplitude"]
        assert isinstance(is_amplitude, Callable)
    if is_symmetric_tensor is None:
        is_symmetric_tensor = globals()["is_symmetric_tensor"]
        assert isinstance(is_symmetric_tensor, Callable)

    expr_string = expr_string.strip()
    if not expr_string:
        return ExprContainer(0)
    # split the expression in the individual terms and (potentially)
    # add a '+' sign to the first term
    term_strings = _split_terms(expr_string)
    if not term_strings[0].startswith(("+", "-")):
        term_strings[0] = "+ " + term_strings[0]

    expr = S.Zero
    for term in term_strings:
        expr += _import_term(
            term_string=term, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )
    return ExprContainer(expr)


def is_amplitude(name: str) -> bool:
    """
    Whether a tensor with the given name should be imported as
    :py:class:`Amplitude` tensor.
    (ADC or ground state amplitude)
    """
    return is_adc_amplitude(name) or is_t_amplitude(name)


def is_symmetric_tensor(name: str) -> bool:
    """
    Whether a tensor with the given name should be imported as
    :py:class:`SymmetricTensor`.
    (Coulomb, symbolic orbital energy denominator, RI tensors)
    """
    sym_names = (
        tensor_names.coulomb, tensor_names.sym_orb_denom,
        tensor_names.ri_sym, tensor_names.ri_asym_eri,
        tensor_names.ri_asym_factor
    )
    return name in sym_names


########################
# import functionality #
########################
def _import_term(term_string: str, convert_default_names: bool,
                 is_amplitude: Callable[[str], bool],
                 is_symmetric_tensor: Callable[[str], bool]) -> Expr:
    """
    Import the given term from string
    """
    # extract the sign of the term
    sign = term_string[0]
    if sign not in ["+", "-"]:
        raise ValueError("Term string has to start with '+' or '-' sign.")
    term: Expr = S.NegativeOne if sign == "-" else S.One
    term_string = term_string[1:].strip()

    if term_string.startswith(r"\frac"):  # fraction
        term *= _import_fraction(
            term_string, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )
    else:
        term *= _import_product(
            term_string, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )
    return term


def _import_fraction(fraction: str, convert_default_names: bool,
                     is_amplitude: Callable[[str], bool],
                     is_symmetric_tensor: Callable[[str], bool]) -> Expr:
    """
    Imports a fraction '\\frac{num}{denom}' from string.
    """
    numerator, denominator = _split_fraction(fraction)
    res: Expr = S.One
    # import num
    if _is_sum(numerator):
        res *= import_from_sympy_latex(
            numerator, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        ).inner
    else:
        res *= _import_product(
            numerator, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )
    # import denom
    if _is_sum(denominator):
        res /= import_from_sympy_latex(
            denominator, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        ).inner
    else:
        res /= _import_product(
            denominator, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )
    assert isinstance(res, Expr)
    return res


def _import_product(product: str, convert_default_names: bool,
                    is_amplitude: Callable[[str], bool],
                    is_symmetric_tensor: Callable[[str], bool]) -> Expr:
    """
    Imports a product (a term that is no fraction) of objects.
    Objects are separated by a space.
    """
    # we have to have a product at this point
    res = S.One
    for obj in _split_product(product):
        res *= _import_object(
            obj, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )
    return res


def _import_object(obj_str: str, convert_default_names: bool,
                   is_amplitude: Callable[[str], bool],
                   is_symmetric_tensor: Callable[[str], bool]) -> Expr:
    """
    Imports the given object (a part of a product) from string.
    """
    if obj_str.isnumeric():  # prefactor
        return sympify(int(obj_str))
    elif obj_str.startswith(r"\sqrt{"):  # sqrt{n} prefactor
        assert obj_str[-1] == "}"
        return sqrt(import_from_sympy_latex(
            obj_str[6:-1].strip(), convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        ).inner)
    elif obj_str.startswith(r"\delta_{"):  # KroneckerDelta
        return _import_kronecker_delta(
            obj_str, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )
    elif obj_str.startswith(r"\left("):
        return _import_polynom(
            obj_str, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )
    elif obj_str.startswith(r"\left\{"):  # NO
        return _import_normal_ordered(
            obj_str, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )
    else:
        # the remaining objects are harder to identify:
        # tensor, creation, annihilation or symbol
        return _import_tensor_like(
            obj_str, convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        )


def _import_kronecker_delta(delta: str, convert_default_names: bool,
                            is_amplitude: Callable[[str], bool],
                            is_symmetric_tensor: Callable[[str], bool]
                            ) -> Expr:
    """
    Imports the given KroneckerDelta of the from
    '\\delta_{p q}' from string
    """
    _ = convert_default_names, is_amplitude, is_symmetric_tensor
    # a delta should not have an exponent!
    delta, exponent = _split_base_and_exponent(delta)
    assert exponent is None
    assert delta.startswith(r"\delta_{") and delta.endswith("}")
    # extract and import the indices
    p, q = delta[8:-1].strip().split()
    p, q = _import_index(p), _import_index(q)
    return KroneckerDelta(p, q)


def _import_polynom(polynom: str, convert_default_names: bool,
                    is_amplitude: Callable[[str], bool],
                    is_symmetric_tensor: Callable[[str], bool]
                    ) -> Expr:
    """
    Imports the given polynom of the form '\\left(...)\\right^{exp}'
    from string
    """
    # try to extract the exponent (if available) (base)^{exponent}
    base, exponent = _split_base_and_exponent(polynom)
    assert base.startswith(r"\left(") and base.endswith(r"\right)")
    # import base and exponent and build a Pow object
    res = import_from_sympy_latex(
        base[6:-7].strip(), convert_default_names=convert_default_names,
        is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
    ).inner
    if exponent is not None:
        assert exponent.startswith("{") and exponent.endswith("}")
        exponent = import_from_sympy_latex(
            exponent[1:-1].strip(),
            convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        ).inner
        res = Pow(res, exponent)
    return res


def _import_normal_ordered(no: str, convert_default_names: bool,
                           is_amplitude: Callable[[str], bool],
                           is_symmetric_tensor: Callable[[str], bool]
                           ) -> Expr:
    """
    Imports the given NormalOrdered object of the form
    '\\left\\{...\\right\\}' from string
    """
    # a NO object should not have an exponent!
    no, exponent = _split_base_and_exponent(no)
    assert exponent is None
    res = import_from_sympy_latex(
        no[7:-8], convert_default_names=convert_default_names,
        is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
    ).inner
    return NO(res)


def _import_tensor_like(tensor: str, convert_default_names: bool,
                        is_amplitude: Callable[[str], bool],
                        is_symmetric_tensor: Callable[[str], bool]
                        ) -> Expr:
    """
    Imports a tensor like object (Symbol, creation and annihilation operators
    and Tensors).
    """
    # possible input:
    # - symbol: 'A^{exp}'
    # - create: 'a_{p}^{exp}'
    # - annihilate: 'a^\dagger_{p}' or '{a^\dagger_{p}}^{exp}'
    # - antisymtensor + symtensor: {name^{upper}_{lower}}^{exp}
    # - nonsymtensor: {name_{indices}}^{exp}

    # split the object in base and exponent
    base, exponent = _split_base_and_exponent(tensor)
    # remove the outer layer of curly braces (if present)
    # so we can split with a stack size of zero
    if base.startswith("{"):
        base = base[1:]
    if base.endswith("}"):
        base = base[:-1]
    components = _split_tensor_like(base)
    assert components  # we have to have at least a name
    name = components[0]
    components = components[1:]
    # remove 1 layer of curly braces from the components (mostly indices)
    # -> there should be no curly braces left afterwards
    for i, comp in enumerate(components):
        if comp.startswith("{"):
            comp = comp[1:]
        if comp.endswith("}"):
            comp = comp[:-1]
        components[i] = comp
    # if desired map the default tensor names to their currently
    # configured name.
    # -> this allows expressions with the default names to
    #    be imported and mapped to the current configuration, correctly
    #    recognizing Amplitudes and SymmetricTensors.
    # Name should be free of curly braces by now
    if convert_default_names:
        name = tensor_names.map_default_name(name)
    # import the tensor like object
    if not components:  # Symbol
        res = Symbol(name)
    elif name == "a":  # creation or annihilation operator
        if len(components) == 2:  # create
            assert components[0] == r"\dagger"
            res = Fd(_import_index(components[1]))
        elif len(components) == 1:  # annihilate
            res = F(_import_index(components[0]))
        else:
            raise RuntimeError(f"Invalid second quantized operator: {tensor}.")
    elif len(components) == 2:  # antisymtensor, symtensor or amplitude
        upper = _import_indices(components[0])
        lower = _import_indices(components[1])
        # figure out which tensor class to use
        if is_amplitude(name):
            res = Amplitude(name, upper, lower)
        elif is_symmetric_tensor(name):
            res = SymmetricTensor(name, upper, lower)
        else:
            res = AntiSymmetricTensor(name, upper, lower)
    elif len(components) == 1:  # nonsymtensor
        res = NonSymmetricTensor(name, _import_indices(components[0]))
    else:
        raise RuntimeError(f"Unknown tensor like object {tensor}")
    # attach the exponent if necessary
    if exponent is not None:
        assert exponent.startswith("{") and exponent.endswith("}")
        exponent = import_from_sympy_latex(
            exponent[1:-1], convert_default_names=convert_default_names,
            is_amplitude=is_amplitude, is_symmetric_tensor=is_symmetric_tensor
        ).inner
        res = Pow(res, exponent)
    return res


def _import_indices(idx_str: str) -> list[Index]:
    """
    Imports the given string of indices
    """
    return [_import_index(idx) for idx in _split_indices(idx_str)]


def _import_index(index_str: str) -> Index:
    """
    Imports the given index of the form
    'a', 'a2', 'a_{\\alpha}' or 'a2_{\\alpha}'
    from string.
    """
    # extract the spin
    spin = None
    if index_str.endswith(r"_{\alpha}"):
        spin = "a"
        index_str = index_str[:-9]
    elif index_str.endswith(r"_{\beta}"):
        spin = "b"
        index_str = index_str[:-8]
    else:
        # ensure that we have a single index without a spin
        assert len(split_idx_string(index_str)) == 1
    # build the index
    idx = get_symbols(index_str, spin)
    assert len(idx) == 1
    return idx.pop()


#################################################
# functionality to split a string in components #
#################################################
def _split_terms(expr_string: str) -> list[str]:
    """
    Split the expression string into a list of term strings
    """
    # we need to split the string on +- signs while keeping track of the
    # brackets (+ in a bracket does not indicate a new term)
    stack: list[str] = []
    terms: list[str] = []
    term_start_idx = 0
    for i, char in enumerate(expr_string):
        if char in ["{", "("]:
            stack.append(char)
        elif char == "}":
            assert stack.pop() == "{"
        elif char == ")":
            assert stack.pop() == "("
        elif char in ["+", "-"] and not stack and i != term_start_idx:
            terms.append(expr_string[term_start_idx:i].strip())
            term_start_idx = i
    assert not stack
    terms.append(expr_string[term_start_idx:].strip())  # last term
    return terms


def _split_fraction(fraction: str) -> tuple[str, str]:
    """
    Splits a fraction '\\frac{num}{denom}' string in numerator and denominator
    """
    assert fraction.startswith(r"\frac{") and fraction.endswith("}")
    # remove outer opening and closing brace
    # -> num}{denom
    fraction = fraction[6:-1]
    stack = 0
    num, denom = None, None
    for i, char in enumerate(fraction):
        if char == "{":
            stack += 1
            if not stack:  # found the opening brace of the denominator
                denom = fraction[i+1:].strip()  # consume remaining string
                break
        elif char == "}":
            if not stack:  # found the closing brace of the numberator
                num = fraction[:i].strip()
            stack -= 1
    if num is None or denom is None:
        raise ValueError("Could not extract numerator and denominator from "
                         f"{fraction}")
    assert not stack
    return num, denom


def _split_product(product: str) -> list[str]:
    """
    Splits the product of objects that are separated by space into the
    individual objects.
    """
    # individual objects are separated by spaces
    stack: list[str] = []
    objects: list[str] = []
    obj_start_idx = 0
    for i, char in enumerate(product):
        if char in ["{", "("]:
            stack.append(char)
        elif char == "}":
            assert stack.pop() == "{"
        elif char == ")":
            assert stack.pop() == "("
        elif char in ["+", "-"]:
            # this breaks down if the input is a sum and no product
            assert stack
        elif char == " " and not stack and i != obj_start_idx:
            obj = product[obj_start_idx:i].strip()
            if obj:
                objects.append(obj)
            obj_start_idx = i
    objects.append(product[obj_start_idx:].strip())  # the last object
    return objects


def _split_tensor_like(obj_str: str) -> list[str]:
    """
    Splits a tensor like object (Symbol, F, Fd, SymbolicTensor)
    onto its components using the delimiters '^' and '_'.
    """
    stack = 0
    components: list[str] = []
    component_start_idx = 0
    for i, char in enumerate(obj_str):
        if char == "{":
            stack += 1
        elif char == "}":
            stack -= 1
        elif char in ["^", "_"] and not stack and i != component_start_idx:
            components.append(obj_str[component_start_idx:i].strip())
            component_start_idx = i + 1
    remainder = obj_str[component_start_idx:].strip()
    if remainder:
        components.append(remainder)
    return components


def _split_base_and_exponent(obj_str: str) -> tuple[str, str | None]:
    """
    Splits the object of the form base^exp in base and exponent.
    If the object has no exponent, None is returned as exponent.
    """
    stack = 0
    base, exponent = None, None
    for i, char in enumerate(obj_str):
        if char in ["{", "("]:
            stack += 1
        elif char in ["}", ")"]:
            stack -= 1
        elif char == "^" and not stack:
            base = obj_str[:i].strip()
            exponent = obj_str[i+1:].strip()
            break
    if base is None:  # we have no exponent
        base = obj_str
    return base, exponent


def _split_indices(idx_str: str) -> list[str]:
    """
    Splits an index string of the form 'ab2b_{\\alpha}b2_{\\beta}' into a
    list [‘a‘, 'b2', 'b_{\\alpha}', 'b2_{\\beta}'].
    """
    splitted = re.findall(r"\D\d*(?:_\{\\(?:alpha|beta)\})?", idx_str)
    # ensure that we did not drop a part of the string
    assert "".join(splitted) == idx_str
    return splitted


######################################################
# Functionality to identify the character of strings #
######################################################
def _is_sum(sum: str) -> bool:
    stack = 0
    for char in sum:
        if char in ["{", "("]:
            stack += 1
        elif char in ["}", ")"]:
            stack -= 1
        elif char in ["+", "-"] and not stack:
            return True
    assert not stack
    return False
