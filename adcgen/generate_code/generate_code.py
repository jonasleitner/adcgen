from ..expr_container import Expr, Term
from ..indices import Index
from ..logger import logger
from ..misc import Inputerror
from ..sort_expr import exploit_perm_sym
from ..symmetry import Permutation
from ..tensor_names import tensor_names

from .contraction import Contraction, term_memory_requirements
from .optimize_contractions import (
    optimize_contractions, unoptimized_contraction
)

from sympy import Symbol, Rational, Pow, Mul
from collections import Counter


def generate_code(expr: Expr, target_indices: str,
                  target_spin: str | None = None,
                  bra_ket_sym: int = 0,
                  antisymmetric_result_tensor: bool = True,
                  backend: str = "einsum", max_itmd_dim: int | None = None,
                  optimize_contraction_scheme: bool = True) -> str:
    """
    Generates contractions for a given expression using either 'einsum'
    (Python) or 'libtensor' (C++) syntax.

    Parameters
    ----------
    expr: Expr
        The expression to generate contractions for.
    target_indices: str
        String of target indices. A ',' might be inserted to indicate where
        the indices are split in upper and lower indices of the result tensor,
        e.g., 'ia,jb' for 'r^{ia}_{jb}'.
    target_spin: str | None, optional
        The spin of the target indices, e.g., 'aabb' to indicate that the
        first 2 target indices have alpha spin, while number 3 and 4 have
        beta spin. If not given, target indices without spin will be used.
    bra_ket_sym: int, optional
        The bra-ket symmetry of the result tensor. (default: 0, i.e.,
        no bra-ket symmetry)
    antisymmetric_result_tensor: bool, optional
        If set, teh result tensor will be treated as AntiSymmetricTensor
        d_{ij}^{ab} = - d_{ji}^{ab}. Otherwise, a SymmetricTensor will be used
        to mimic the symmetry of the result tensor, i.e.,
        d_{ij}^{ab} = d_{ji}^{ab}. (default: True)
    backend: str, optional
        The backend for which to generate contractions. (default: einsum)
    max_itmd_dim: int | None, optional
        Upper bound for the dimensionality of intermediate results, that
        may be generated if the contractions are optimized.
    optimize_contraction_scheme: bool, optional
        If set, we try to find the contractions with the lowest arithmetic
        and memory scaling, i.e., if possible only 2 tensors are contracted
        simultaneously. (default: True)
    """
    if not isinstance(expr, Expr):
        raise Inputerror("The expression needs to be provided as 'Expr'.")

    # try to reduce the number of terms by exploiting permutational symmetry
    expr_with_perm_sym = exploit_perm_sym(
        expr=expr, target_indices=target_indices, target_spin=target_spin,
        bra_ket_sym=bra_ket_sym,
        antisymmetric_result_tensor=antisymmetric_result_tensor
    )
    # remove the bra-ket separator in target indices and target spin
    if "," in target_indices:
        target_indices = target_indices.replace(",", "")
    if target_spin is not None and "," in target_spin:
        target_spin = target_spin.replace(",", "")

    code = []
    for perm_symmetry, sub_expr in expr_with_perm_sym.items():
        perm_str = format_perm_symmetry(perm_symmetry)

        # generate the contrations for each of the terms
        contraction_code = []
        for term in sub_expr.terms:
            prefactor = format_prefactor(term, backend)

            if not term.idx:  # term is just a prefactor
                contraction_code.append(prefactor)
                continue
            if len({idx.spin for idx in term.idx}) > 1:
                logger.warning("Found more than one spin in the indices of "
                               f"term {term}. Indices with different spin "
                               "might not be distinguishable in the "
                               "generated contractions, because only the name "
                               "of the indices is considered.")

            # generate the contractions for the term
            if optimize_contraction_scheme:
                contractions = optimize_contractions(
                    term=term, target_indices=target_indices,
                    target_spin=target_spin,
                    max_itmd_dim=max_itmd_dim
                )
            else:
                contractions = unoptimized_contraction(
                    term=term, target_indices=target_indices,
                    target_spin=target_spin
                )
            # build a comment describing the scaling of the contraction
            # scheme
            scaling_comment = format_scaling_comment(
                term=term, contractions=contractions, backend=backend
            )
            # contractions are sorted in the way they need to be performed,
            # i.e., inner contractions are listed first. Therefore,
            # we need to cache the strings from the inner contractions
            # using the unique contraction id.
            contraction_cache = {}
            for contr in contractions[:-1]:
                contr_str = format_contraction(contr, contraction_cache,
                                               backend=backend)
                contraction_cache[contr.id] = contr_str
            contr_str = format_contraction(contractions[-1], contraction_cache,
                                           backend=backend)
            contraction_code.append(
                f"{prefactor} * {contr_str}  {scaling_comment}"
            )
        contraction_code = '\n'.join(contraction_code)
        code.append(
            "The scaling comment is given as: [comp_scaling] / [mem_scaling]\n"
            f"Apply {perm_str} to:\n{contraction_code}"
        )
    return "\n\n".join(code)


def format_contraction(contraction: Contraction,
                       contraction_cache: dict[int, str],
                       backend: str) -> str:
    """
    Builds a backend specific string for the given contraction.
    """
    # split the objects in tensors and factors
    # and transform the indices of the tensors to string
    tensors: list[str] = []
    factors: list[str] = []
    idx_str: list[str] = []
    for name, indices in zip(contraction.names, contraction.indices):
        # check the cache for the contraction string of the other contraction
        if name.startswith("contraction"):
            id = int(name.split("_")[-1])
            name = contraction_cache.get(id, None)
            if name is None:
                raise KeyError("Could not find contraction string for inner "
                               f"contraction {contraction}.")
        else:  # translate eri and fock matrix
            if backend == "einsum":
                name = translate_adcc_names(name, indices)
            elif backend == "libtensor":
                name = translate_libadc_names(name, indices)
                name = format_libtensor_tensor(name, indices)

        if indices:  # we have a tensor
            tensors.append(name)
            # build a string for the indices
            idx_str.append("".join(idx.name for idx in indices))
        else:  # we have a factor without indices
            factors.append(name)
    # also transform the target indices to string
    target = "".join(idx.name for idx in contraction.target)

    if backend == "einsum":
        return format_einsum_contraction(tensors=tensors, factors=factors,
                                         indices=idx_str, target=target)
    elif backend == "libtensor":
        return format_libtensor_contraction(tensors=tensors, factors=factors,
                                            target=target,
                                            contracted=contraction.contracted)
    else:
        raise NotImplementedError("Contraction not implemented for backend "
                                  f"{backend}.")


def format_einsum_contraction(tensors: list[str], factors: list[str],
                              indices: list[str], target: str) -> str:
    """
    Builds a contraction string for the given contraction using Python
    numpy einsum syntax.
    """

    components = [*factors]
    # special case: single tensor with the correct target indices
    # -> no einsum needed
    if len(tensors) == 1 and indices[0] == target:
        components.append(tensors[0])
    elif tensors:  # we need a einsum: reorder or contraction or outer
        contr_str = f"\"{','.join(indices)}->{target}\""
        components.append(
            f"einsum({contr_str}, {', '.join(tensors)})"
        )
    return " * ".join(components)


def format_libtensor_contraction(tensors: list[str], factors: list[str],
                                 target: str, contracted: tuple[Index]) -> str:
    """
    Builds a contraction string for the given contraction using libtensor
    C++ syntax.
    """

    components = [*factors]
    if len(tensors) == 1:  # single tensor
        assert not contracted  # trace
        components.append(tensors[0])
    elif len(tensors) > 1:  # multipe tensors
        # hyper-contraction only implemented for 3 tensors i think
        if contracted and target:  # contract
            components.append(
                f"contract({'|'.join(s.name for s in contracted)}, "
                f"{', '.join(tensors)})"
            )
        elif not contracted and target:  # outer product
            components.extend(tensors)
        elif contracted and not target:  # inner product
            components.append(f"dot_product({', '.join(tensors)})")
        else:
            raise NotImplementedError("No target and contracted indices in "
                                      f"contraction of {tensors} and "
                                      f"{factors}.")
    return " * ".join(components)


def format_libtensor_tensor(name: str, indices: tuple[Index]):
    """Formats a single tensor object using libtensor syntax."""

    if any(n > 1 for n in Counter(indices).values()):
        raise NotImplementedError("Libtensor can not handle a partial trace, "
                                  "i.e., a trace with a tensor as result. "
                                  f"Found {indices} on tensor {name}.")
    return f"{name}({'|'.join(idx.name for idx in indices)})"


def translate_adcc_names(name: str, indices: tuple[Index]) -> str:
    """Translates tensor names specifically for adcc."""
    if name.startswith(tensor_names.eri):
        space = "".join(s.space[0] for s in indices)
        return f"hf.{space}"
    elif name.startswith(tensor_names.fock):
        space = "".join(s.space[0] for s in indices)
        return f"hf.f{space}"
    return name


def translate_libadc_names(name: str, indices: tuple[Index]) -> str:
    if name.startswith(tensor_names.eri):
        space = "".join(s.space[0] for s in indices)
        return f"i_{space}"
    elif name.startswith("t2eri"):
        _, n = name.split("_")
        return f"pi{n}"
    return name


def format_scaling_comment(term: Term, contractions: list[Contraction],
                           backend: str) -> str:
    """
    Builds a backend specific comment describing the scaling of the
    contraction scheme.
    """
    max_comp_scaling = max(contr.scaling.computational
                           for contr in contractions)
    max_mem_scaling = max(contr.scaling.memory for contr in contractions)
    max_mem_scaling = max(max_mem_scaling, term_memory_requirements(term))
    comp = [f"N^{max_comp_scaling.total}: "]
    mem = [f"N^{max_mem_scaling.total}: "]
    for space in ["general", "occ", "virt"]:
        if (n := getattr(max_comp_scaling, space)):
            comp.append(f"{space[0].capitalize()}^{n}")
        if (n := getattr(max_mem_scaling, space)):
            mem.append(f"{space[0].capitalize()}^{n}")
    if backend == "einsum":
        comment_token = "#"
    elif backend == "libtensor":
        comment_token = "//"
    else:
        raise NotImplementedError("Comment token not implemented for backend "
                                  f"{backend}.")
    return f"{comment_token} {''.join(comp)} / {''.join(mem)}"


def format_prefactor(term: Term, backend: str) -> str:
    """Formats the prefactor for Python (einsum) or C++ (libtensor)."""
    # extract number and symbolic prefactor
    number_pref = term.prefactor
    symbol_pref = " * ".join(
        [obj.name for obj in term.objects if isinstance(obj.base, Symbol)
         for _ in range(obj.exponent)]
    )
    # extract the sign
    if number_pref < 0:
        sign = "-"
        number_pref *= -1
    else:
        sign = "+"
    # format the number prefactor (depends on the backend)
    if backend == "einsum":  # python
        number_pref = _format_python_prefactor(number_pref)
    elif backend == "libtensor":  # C++
        number_pref = _format_cpp_prefactor(number_pref)
    else:
        raise NotImplementedError(f"Prefactor for backend {backend} not "
                                  "implemented.")
    # combine the contributions
    if symbol_pref:
        return f"{sign} {number_pref} * {symbol_pref}"
    else:
        return f"{sign} {number_pref}"


def _format_python_prefactor(prefactor) -> str:
    """Formats a prefactor using Python syntax."""

    if prefactor == int(prefactor):  # natural number
        return str(prefactor)
    elif prefactor in [Rational(1, 2), Rational(1, 4)]:  # simple Rational
        return str(float(prefactor))
    elif isinstance(prefactor, Rational):  # mor ecomplex rational
        return f"{prefactor.p} / {prefactor.q}"
    elif isinstance(prefactor, Pow) and prefactor.args[1] == 0.5:  # sqrt
        return f"sqrt({prefactor.args[0]})"
    elif isinstance(prefactor, Mul):
        return " * ".join(
            _format_python_prefactor(pref) for pref in prefactor.args
        )
    raise NotImplementedError(
        f"Formatting of prefactor {prefactor}, {type(prefactor)} "
        "not implemented."
    )


def _format_cpp_prefactor(prefactor) -> str:
    """Formats a prefactor using C++ syntax."""

    if prefactor == int(prefactor) or \
            prefactor in [Rational(1, 2), Rational(1, 4)]:
        return str(float(prefactor))
    elif isinstance(prefactor, Rational):
        return f"{float(prefactor.p)} / {float(prefactor.q)}"
    elif isinstance(prefactor, Pow) and prefactor.args[1] == 0.5:
        return f"constants::sq{prefactor.args[0]}"
    elif isinstance(prefactor, Mul):
        return " * ".join(
            _format_cpp_prefactor(pref) for pref in prefactor.args
        )
    raise NotImplementedError(
        f"Formatting of prefactor {prefactor}, {type(prefactor)} "
        "not implemented."
    )


def format_perm_symmetry(perm_symmetry: tuple[Permutation]):
    """Formats the permutational symmetry."""
    perm_sym = ["1"]
    for permutations, factor in perm_symmetry:
        assert factor in [1, -1]
        contrib = ["+ "] if factor == 1 else ["- "]
        for perm in permutations:
            contrib.append(str(perm))
        perm_sym.append("".join(contrib))
    if len(perm_sym) == 1:
        return "1"
    return f"({' '.join(perm_sym)})"
