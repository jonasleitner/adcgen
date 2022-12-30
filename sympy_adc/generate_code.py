from . import expr_container as e
from .misc import Inputerror
from .sort_expr import exploit_perm_sym
from collections import namedtuple

scaling = namedtuple('scaling', ['o', 'v', 'g'])
contraction_data = namedtuple('contraction_data',
                              ['obj_idx', 'indices', 'obj_names',
                               'contracted', 'target', 'scaling'])


def generate_code(expr: e.expr, target_indices: str, backend: str,
                  target_bra_ket_sym: int = 0, max_tensor_dim: int = None,
                  optimize_contractions: bool = True) -> str:

    def unoptimized_contraction(term: e.term, target_indices: str = None):
        from .indices import index_space, get_symbols
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
            if 'tensor' in (o_type := o.type) or o_type == 'delta':
                o_idx.append(i)
                names.append(o.pretty_name)
                idx = o.idx
                indices.append(idx)
                contracted.update(s for s in idx if s not in target)
            elif o_type == 'prefactor':
                continue
            else:  # polynom / create / annihilate / NormalOrdered
                raise NotImplementedError("Contractions not implemented for "
                                          "polynoms, creation and annihilation"
                                          f"operators: {term}")
        contracted = sorted(contracted, key=lambda s:
                            (index_space(s.name)[0],
                             int(s.name[1:]) if s.name[1:] else 0,
                             s.name[0]))
        # determine the scaling
        target_sp = [index_space(s.name)[0] for s in target]
        contracted_sp = [index_space(s.name)[0] for s in contracted]

        occ = target_sp.count('o') + contracted_sp.count('o')
        virt = target_sp.count('v') + contracted_sp.count('v')
        general = target_sp.count('g') + contracted_sp.count('g')
        return [contraction_data(tuple(o_idx), tuple(indices), tuple(names),
                                 tuple(contracted), tuple(target),
                                 scaling(occ, virt, general))]

    if not isinstance(expr, e.expr):
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
    if ',' in target_indices:
        if target_indices.count(',') != 1:
            raise Inputerror("Found more than 1 ',' in target_indices str "
                             f"{target_indices}. Since it should separate "
                             "bra and ket spaces, there should only be 1.")
        upper, lower = target_indices.split(',')
        expr_with_perm_sym = exploit_perm_sym(expr, upper, lower,
                                              target_bra_ket_sym)
        target_indices = upper + lower  # recombine the indices without ','
    else:
        expr_with_perm_sym = exploit_perm_sym(expr)

    ret = ''
    for perm_symmetry, expr in expr_with_perm_sym.items():
        # make symmetry more readable
        perm_symmetry = _pretty_perm_symmetry(perm_symmetry)

        # generate contractions for each term
        contraction_code = ''
        for term in expr.terms:
            # 1) generate a string for the prefactor
            if (pref := term.prefactor) < 0:
                pref_str = '- '
                pref *= -1  # abs(pref)
            else:
                pref_str = '+ '
            pref_str += backend_specifics['prefactor'](pref)

            if term.sympy.is_number:  # term just consists of a number
                contraction_code += pref_str
                continue

            if optimize_contractions:  # only two objects at once - min scaling
                contractions = term.optimized_contractions(target_indices,
                                                           max_tensor_dim)
            else:  # just contract all objects at once
                contractions = unoptimized_contraction(term, target_indices)

            # 2) construct a comment string that describes the scaling
            max_scal = max([contr.scaling for contr in contractions],
                           key=lambda sc: (sum(sc), sc.g, sc.v, sc.o))
            scaling_comment = \
                f"{backend_specifics['comment']} N^{sum(max_scal)}: "
            for sp in ['o', 'v', 'g']:
                if (n := getattr(max_scal, sp)):
                    scaling_comment += f"{sp.capitalize()}^{n}"

            # 3) construct a string for the contraction
            contraction_strings = {}  # cache for not final contractions
            last_idx = len(contractions) - 1
            for i, c_data in enumerate(contractions):
                c_str = backend_specifics['contraction'](c_data,
                                                         contraction_strings)
                if i == last_idx:  # only the last contraction is relevant
                    contraction_code += \
                        f"{pref_str} * {c_str}  {scaling_comment}\n"
                else:  # save the contraction -> need it in the final
                    contraction_strings[c_data.obj_idx] = (c_str, c_data)
        ret += f"Apply {perm_symmetry} to:\n{contraction_code}\n\n"
    return ret


def _einsum_contraction(c_data: contraction_data, c_strings: dict) -> str:
    """Generate a contraction string using the einsum syntax."""
    from .indices import index_space

    translate_tensor_names = {
        'V': lambda indices: (  # eri
            f"hf.{''.join(index_space(s.name)[0] for s in indices)}"
        ),
        'f': lambda indices: (  # fock
            f"hf.f{''.join(index_space(s.name)[0] for s in indices)}"
        )
    }

    # special case:
    #  we only have a single object that has exactly matching indices
    if len(c_data.indices) == 1 and c_data.indices[0] == c_data.target:
        name = c_data.obj_names[0]
        if name in translate_tensor_names:
            name = translate_tensor_names[name](c_data.indices[0])
        return name

    obj_strings = []
    idx_strings = []
    for o_idx, indices, name in \
            zip(c_data.obj_idx, c_data.indices, c_data.obj_names):
        if name == 'contraction':  # nested contraction
            try:  # get the contraction string and its target indices
                c_str, other_c_data = c_strings[o_idx]
                obj_strings.append(c_str)
                idx_strings.append(
                    "".join(s.name for s in other_c_data.target)
                )
            except KeyError:
                raise KeyError(f"Could not find the contraction {o_idx} to "
                               f"use in the nested contraction {c_data}. "
                               "Should have been constructed ealier.")
        else:  # contraction of two tensors
            if name in translate_tensor_names:
                name = translate_tensor_names[name](indices)

            obj_strings.append(name)
            idx_strings.append("".join(s.name for s in indices))
    target_str = "".join(s.name for s in c_data.target)
    # build the einsum contraction string
    contraction_str = f"einsum('{','.join(idx_strings)}->{target_str}', "
    contraction_str += f"{', '.join(obj_strings)})"
    return contraction_str


def _libtensor_contraction(c_data: contraction_data, c_strings: dict) -> str:
    """Generate a contraction string using the libtensor syntax."""
    from .indices import index_space

    translate_tensor_names = {
        'V': lambda indices: (  # eri
            f"i_{''.join(index_space(s.name)[0] for s in indices)}"
        ),
        'f': lambda indices: (  # fock
            f"f_{''.join(index_space(s.name)[0] for s in indices)}"
        )
    }

    # contraction of a single object -> trace/transpose/just add the object
    if len(c_data.obj_idx) == 1:
        if c_data.contracted:  # trace
            raise NotImplementedError(f"trace or sth like that {c_data}")
        # just return the object?
        # TODO: what about transpose?
        name = c_data.obj_names[0]
        indices = c_data.indices[0]
        if name in translate_tensor_names:
            name = translate_tensor_names[name](indices)
        return f"{name}({'|'.join(s.name for s in indices)})"

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
    else:
        raise NotImplementedError(f"{c_data}")

    obj_strings = []
    for o_idx, indices, name in \
            zip(c_data.obj_idx, c_data.indices, c_data.obj_names):
        if name == 'contraction':
            try:
                c_str, _ = c_strings[o_idx]
                obj_strings.append(c_str)
            except KeyError:
                raise KeyError(f"The contraction {o_idx} has not been "
                               "constructed prior to the current contration "
                               f"{c_data}.")
        else:
            if name in translate_tensor_names:
                name = translate_tensor_names[name](indices)
            obj_strings.append(f"{name}({'|'.join(s.name for s in indices)})")
    return start + separator.join(obj_strings) + end


def _einsum_prefactor(pref):
    """Transforms the prefactor of a term to a string for python/einsum."""
    from sympy import Rational

    if pref == int(pref):  # natural numbers
        return str(pref)
    elif pref in [0.5, 0.25]:
        return str(float(pref))
    elif isinstance(pref, Rational):
        return f"{pref.p} / {pref.q}"
    else:
        raise NotImplementedError(f"{pref}, {type(pref)}")


def _libtensor_prefactor(pref):
    """Transforms the prefactor to string for C++/libtensor."""
    from sympy import Rational, S

    if pref == int(pref) or pref in [S.Half, 0.25]:
        return str(float(pref))
    elif isinstance(pref, Rational):
        num = float(pref.p)
        denom = float(pref.q)
        return f"{num} / {denom}"
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
