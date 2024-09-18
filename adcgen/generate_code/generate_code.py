from .. import expr_container as e
from ..misc import Inputerror
from ..sort_expr import exploit_perm_sym
from ..indices import sort_idx_canonical, get_symbols
from ..sympy_objects import SymbolicTensor, KroneckerDelta
from ..tensor_names import tensor_names

from .optimize_contractions import optimize_contractions

from collections import namedtuple
from sympy import Symbol

scaling = namedtuple('scaling', ['total', 'g', 'v', 'o', 'mem'])
mem_scaling = namedtuple('mem_scaling', ['total', 'g', 'v', 'o'])
contraction_data = namedtuple('contraction_data',
                              ['obj_idx', 'indices', 'obj_names',
                               'contracted', 'target', 'scaling'])


def generate_code(expr: e.Expr, target_indices: str, backend: str,
                  bra_ket_sym: int = 0, max_tensor_dim: int = None,
                  optimize_contraction_scheme: bool = True) -> str:
    """Transforms an expression to contractions using either einsum (python)
       or libtensor (C++) syntax. Additionally, the computational and the
       memory scaling of each term is given as comment after each contraction
       string in the form '{comp_scaling} / {mem_scaling}'.
       """

    def unoptimized_contraction(term: e.Term, target_indices: str):
        # construct a contraction_data object for the simulötaneous,
        # unoptimized contraction of all objects of a term.
        target = get_symbols(target_indices)
        o_idx = []
        indices = []
        names = []
        contracted = set()
        for i, o in enumerate(term.objects):
            # nonsym_tensor / antisym_tensor / delta
            # -> extract relevant data
            base, exp = o.base_and_exponent
            if isinstance(base, (SymbolicTensor, KroneckerDelta)):
                o_idx.extend(i for _ in range(exp))
                names.extend(o.longname() for _ in range(exp))
                idx = o.idx
                indices.extend(idx for _ in range(exp))
                contracted.update(s for s in idx if s not in target)
            elif o.sympy.is_number or isinstance(base, Symbol):  # prefactor
                continue
            else:  # polynom / create / annihilate / NormalOrdered
                raise NotImplementedError("Contractions not implemented for "
                                          "polynoms, creation and annihilation"
                                          f"operators: {term}")
        contracted = sorted(contracted, key=sort_idx_canonical)
        # determine the scaling
        target_sp = [s.space[0] for s in target]
        contracted_sp = [s.space[0] for s in contracted]

        occ = target_sp.count('o') + contracted_sp.count('o')
        virt = target_sp.count('v') + contracted_sp.count('v')
        general = target_sp.count('g') + contracted_sp.count('g')
        total = occ + virt + general
        mem = mem_scaling(len(target_indices), target_sp.count('g'),
                          target_sp.count('v'), target_sp.count('o'))
        scal = scaling(total, general, virt, occ, mem)
        return [contraction_data(tuple(o_idx), tuple(indices), tuple(names),
                                 tuple(contracted), tuple(target), scal)]

    if not isinstance(expr, e.Expr):
        raise Inputerror("The expression needs to be provided as an instance "
                         "of the expr container.")

    if backend not in ['einsum', 'libtensor']:
        raise Inputerror(f"Unknown backend {backend}. Available:"
                         "'einsum' and 'libtensor'.")

    backend_specifics = {
        'einsum': {'prefactor': _einsum_prefactor, 'comment': '#',
                   'contraction': _einsum_contraction},
        'libtensor': {'prefactor': _libtensor_prefactor, 'comment': '//',
                      'contraction': _libtensor_contraction}
    }
    backend_specifics = backend_specifics[backend]

    # try to reduce the number of terms by exploiting permutational symmetry
    expr_with_perm_sym = exploit_perm_sym(
        expr=expr, target_indices=target_indices, bra_ket_sym=bra_ket_sym)
    if ',' in target_indices:  # remove the separator in target indices
        target_indices = "".join(target_indices.split(','))

    ret = []
    for perm_symmetry, expr in expr_with_perm_sym.items():
        # make symmetry more readable
        perm_symmetry = _pretty_perm_symmetry(perm_symmetry)

        # generate contractions for each term
        contraction_code = []
        for term in expr.terms:
            # 1) generate a string for the prefactor
            if (pref := term.prefactor) < 0:
                pref_str = '- '
                pref *= -1  # abs(pref)
            else:
                pref_str = '+ '
            pref_str += backend_specifics['prefactor'](pref)

            # add symbols to the prefactor
            symbol_pref = " * ".join(
                [o.name for o in term.objects if isinstance(o.base, Symbol)
                 for _ in range(o.exponent)]
            )
            if symbol_pref:
                pref_str += f" * {symbol_pref}"

            if not term.idx:  # the term just consists of numbers
                contraction_code.append(pref_str)
                continue

            if optimize_contraction_scheme:
                contractions = optimize_contractions(term, target_indices,
                                                     max_tensor_dim)
            else:  # just contract all objects at once
                contractions = unoptimized_contraction(term, target_indices)

            # 2) construct a comment string that describes the scaling:
            #    computational and memory
            max_scal: scaling = max(contr.scaling for contr in contractions)
            max_itmd_mem = max(contr.scaling.mem for contr in contractions)
            max_mem: mem_scaling = max(max_itmd_mem, term.memory_requirements)
            comp_scaling = ""
            mem_scal = ""
            for sp in ['o', 'v', 'g']:
                if (n := getattr(max_scal, sp)):
                    comp_scaling += f"{sp.capitalize()}^{n}"
                if (n := getattr(max_mem, sp)):
                    mem_scal += f"{sp.capitalize()}^{n}"
            scaling_comment = f"{backend_specifics['comment']} "
            scaling_comment += f"N^{max_scal.total}: {comp_scaling} / "
            scaling_comment += f"N^{max_mem.total}: {mem_scal}"

            # 3) construct a string for the contraction
            contraction_strings = {}  # cache for not final contractions
            last_idx = len(contractions) - 1
            for i, c_data in enumerate(contractions):
                c_str = backend_specifics['contraction'](c_data,
                                                         contraction_strings)
                if i == last_idx:  # only the last contraction is relevant
                    contraction_code.append(
                        f"{pref_str} * {c_str}  {scaling_comment}"
                    )
                else:  # save the contraction -> need it in the final
                    contraction_strings[c_data.obj_idx] = (c_str, c_data)
        contraction_code = "\n".join(contraction_code)
        res_string = (
            "The scaling comment is given as: [comp_scaling] / "
            f"[mem_scaling]\nApply {perm_symmetry} to:\n{contraction_code}"
        )
        ret.append(res_string)
    return "\n\n".join(ret)


def _einsum_contraction(c_data: contraction_data, c_strings: dict) -> str:
    """Generate a contraction string using the einsum syntax."""

    translate_tensor_names = {
        tensor_names.eri: lambda indices: (  # eri
            f"hf.{''.join(s.space[0] for s in indices)}"
        ),
        tensor_names.fock: lambda indices: (  # fock
            f"hf.f{''.join(s.space[0] for s in indices)}"
        )
    }

    # special case:
    #  we only have a single object that has exactly matching indices
    if len(c_data.obj_idx) == 1 and c_data.indices[0] == c_data.target:
        name = c_data.obj_names[0]
        if name.split('_')[0] in translate_tensor_names:
            name = translate_tensor_names[name.split('_')[0]]
            name = name(c_data.indices[0])
        return name

    obj_strings = []
    idx_strings = []
    factors = []
    for o_idx, indices, name in \
            zip(c_data.obj_idx, c_data.indices, c_data.obj_names):
        if name == 'contraction':  # nested contraction
            try:  # get the contraction string and its target indices
                c_str, other_c_data = c_strings[o_idx]
                if indices:
                    obj_strings.append(c_str)
                    idx_strings.append(
                        "".join(s.name for s in other_c_data.target)
                    )
                else:  # the object has no indices -> its a number
                    factors.append(c_str)
            except KeyError:
                raise KeyError(f"Could not find the contraction {o_idx} to "
                               f"use in the nested contraction {c_data}. "
                               "Should have been constructed ealier.")
        else:  # contraction of two tensors
            if name.split('_')[0] in translate_tensor_names:
                name = translate_tensor_names[name.split('_')[0]](indices)

            if indices:
                obj_strings.append(name)
                idx_strings.append("".join(s.name for s in indices))
            else:  # no indices -> its a number
                raise RuntimeError("An object that is no contraction should "
                                   "hold indices. Did we miss a prefactor?",
                                   c_data)

    contraction_str = ""
    if factors:
        contraction_str += " * ".join(factors)
    if obj_strings:
        if contraction_str:
            contraction_str += " * "
        target_str = "".join(s.name for s in c_data.target)
        # if we only have a single tensor with correct target indices
        # -> factor * tensor
        if len(idx_strings) == 1 and idx_strings[0] == target_str:
            contraction_str += obj_strings[0]
        else:
            # build the einsum contraction string
            contraction_str += (
                f"einsum('{','.join(idx_strings)}->{target_str}', "
            )
            contraction_str += f"{', '.join(obj_strings)})"
    if not contraction_str:
        raise RuntimeError(f"Could not translate {c_data} to a contraction "
                           "string.")
    return contraction_str


def _libtensor_contraction(c_data: contraction_data, c_strings: dict) -> str:
    """Generate a contraction string using the libtensor syntax."""
    from collections import Counter

    def libtensor_object(name: str, indices):
        # if no index is repeating -> return name(i|j)
        # else name(i|i) -> diag(i, i|j, name(i|j)

        # get the basic name of the tensor (special case for ERI)
        if name.split('_')[0] in translate_tensor_names:
            name = translate_tensor_names[name.split('_')[0]](indices)

        # count indices
        if not indices:
            raise ValueError("An object is expected to hold indices. "
                             f"Found: {name} with indices {indices}.")
        counted_idx = Counter(indices)
        # all indices occur exactly once -> everything is fine
        if all(n == 1 for n in counted_idx.values()):
            return f"{name}({'|'.join(s.name for s in indices)})"
        # indices repeat on a tensor -> problem for libtensor
        raise NotImplementedError("Libtensor can not handle repeating indices"
                                  " on 1 tensor, i.e., contract(i, tensor) is"
                                  f" not implemented.\n{c_data}.")

    translate_tensor_names = {
        tensor_names.eri: lambda indices: (  # eri
            f"i_{''.join(s.space[0] for s in indices)}"
        ),
    }

    # contraction of a single object -> trace/transpose/just add the object
    if len(c_data.obj_idx) == 1:
        if c_data.contracted:  # trace
            raise NotImplementedError(f"trace or sth like that {c_data}")
        # just return the object?
        # TODO: what about transpose?
        name = c_data.obj_names[0]
        indices = c_data.indices[0]
        return libtensor_object(name, indices)

    if c_data.contracted and c_data.target:  # contract
        start = f"contract({'|'.join(s.name for s in c_data.contracted)}, "
        end = ")"
        separator = ", "
    elif not c_data.contracted and c_data.target:  # outer product
        start = ""
        end = ""
        separator = ' * '
    elif c_data.contracted and not c_data.target:  # inner product
        start = "dot_product("
        end = ")"
        separator = ", "
    else:  # neither target nor contracted indices -> a number
        raise ValueError("Numbers should have been catched ealier: "
                         f"{c_data}")

    obj_strings = []
    factors = []
    for o_idx, indices, name in \
            zip(c_data.obj_idx, c_data.indices, c_data.obj_names):
        if name == 'contraction':
            try:
                c_str, _ = c_strings[o_idx]
                if indices:
                    obj_strings.append(c_str)
                else:
                    factors.append(c_str)
            except KeyError:
                raise KeyError(f"The contraction {o_idx} has not been "
                               "constructed prior to the current contration "
                               f"{c_data}.")
        else:
            obj_strings.append(libtensor_object(name, indices))

    contraction_str = ""
    if factors:
        contraction_str += " * ".join(factors)
    if obj_strings:
        if contraction_str:
            contraction_str += " * "
        contraction_str += start + separator.join(obj_strings) + end
    return contraction_str


def _einsum_prefactor(pref):
    """Transforms the prefactor of a term to a string for python/einsum."""
    from sympy import Rational, Pow

    if pref == int(pref):  # natural numbers
        return str(pref)
    elif pref in [0.5, 0.25]:
        return str(float(pref))
    elif isinstance(pref, Rational):
        return f"{pref.p} / {pref.q}"
    elif isinstance(pref, Pow) and pref.args[1] == 0.5:  # sqrt -> import math
        return f"sqrt({pref.args[0]})"
    else:
        raise NotImplementedError(f"{pref}, {type(pref)}")


def _libtensor_prefactor(pref):
    """Transforms the prefactor to string for C++/libtensor."""
    from sympy import Rational, S, Pow, Mul

    if pref == int(pref) or pref in [S.Half, 0.25]:
        return str(float(pref))
    elif isinstance(pref, Rational):
        num = float(pref.p)
        denom = float(pref.q)
        return f"{num} / {denom}"
    elif isinstance(pref, Pow) and pref.args[1] == 0.5:
        return f"constants::sq{pref.args[0]}"
    elif isinstance(pref, Mul):
        return " * ".join(_libtensor_prefactor(p) for p in pref.args)
    else:
        raise NotImplementedError(f"{pref}, {type(pref)}")


def _pretty_perm_symmetry(perm_sym: tuple) -> str:
    if not perm_sym:  # trivial case
        return "1"

    perm_sym_str = "(1"
    for perms, factor in perm_sym:
        perm_str = " + " if factor == 1 else " - "
        for perm in perms:
            perm_str += f"P_{{{''.join(s.name for s in perm)}}}"
        perm_sym_str += perm_str
    return perm_sym_str + ")"
