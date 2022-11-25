from sympy import S
import expr_container as e
from misc import Inputerror
from indices import index_space


def by_delta_types(expr):
    """Sort the terms in an expression according to the space of the deltas
       contained in the term."""
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)
    ret = {}
    for term in expr.terms:
        d_spaces = tuple(sorted([o.space for o in term.deltas
                         for i in range(o.exponent)]))
        if not d_spaces:
            d_spaces = ('none',)
        if d_spaces not in ret:
            ret[d_spaces] = e.compatible_int(0)
        ret[d_spaces] += term
    return ret


def by_delta_indices(expr):
    """Sort the terms in an expression according to the indices on the
       KroneckerDeltas in the term."""
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)
    ret = {}
    for term in expr.terms:
        d_idx = [o.idx for o in term.deltas for i in range(o.exponent)]
        for i, d in enumerate(d_idx):
            d_idx[i] = "".join([s.name for s in d])
        d_idx = tuple(sorted(d_idx))
        if not d_idx:
            d_idx = ('none',)
        if d_idx not in ret:
            ret[d_idx] = e.compatible_int(0)
        ret[d_idx] += term
    return ret


def by_tensor_block(expr, t_string, symmetric=False):
    """Sorts the terms in an expression according to the blocks of a tensor,
       e.g. collect all terms that define the 'oo' block of the density matrix.
       If the desired tensor occures more than once in a term, the sorted list
       of blocks will be used."""
    if not isinstance(t_string, str):
        raise Inputerror("Tensor name must be provided as string.")
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)
    # if the tensor is symmetric -> add to symmetric tensors of the expr
    if symmetric and t_string not in expr.sym_tensors:
        sym_tensors = expr.sym_tensors
        sym_tensors.add(t_string)
        expr.set_sym_tensors(sym_tensors)
    elif not symmetric and t_string in expr.sym_tensors:
        sym_tensors = expr.sym_tensors
        sym_tensors.remove(t_string)
        expr.set_sym_tensors(sym_tensors)

    ret = {}
    for term in expr.terms:
        # the space is automatically adjusted for a symmetric tensor
        blocks = tuple(sorted(
            [t.space for t in term.tensors if t.name == t_string]
        ))
        if not blocks:
            blocks = ("none",)
        if blocks not in ret:
            ret[blocks] = e.compatible_int(0)
        ret[blocks] += term
    return ret


def by_tensor_target_idx(expr, t_string):
    """Sorts the terms in an expression according to the number and type of
       target indices on a specified tensor, e.g. f_cc Y_ij^ac -> if sorting
       according to Y: oov; if sorting acording to f: none.
       """
    from simplify import filter_tensor

    if not isinstance(t_string, str):
        raise Inputerror("Tensor name must be provided as string.")
    expr = expr.expand()
    if not isinstance(expr, e.expr):
        expr = e.expr(expr)
    # filter to only obtain terms that contain the requested tensor
    tensor_terms = filter_tensor(expr, t_string, strict='low')

    ret = {}
    # collect all terms that do not contain the tensor
    remaining_terms = expr - tensor_terms
    if remaining_terms.sympy is not S.Zero:
        ret[(f'no_{t_string}',)] = remaining_terms
        # if there are no terms that contain the tensor
        if tensor_terms.sympy is S.Zero:
            return ret
    del remaining_terms

    for term in tensor_terms.terms:
        key = []
        target = term.target
        for o in term.tensors:
            if o.name == t_string:
                obj_target_space = [
                    index_space(s.name)[0] for s in o.idx if s in target
                ]
                if o.type == 'antisym_tensor':
                    obj_target_space = sorted(obj_target_space)
                obj_target_space = "".join(obj_target_space)
                if obj_target_space:
                    key.append(obj_target_space)
        key = tuple(sorted(key))
        if not key:
            key = ('none',)
        if key not in ret:
            ret[key] = e.compatible_int(0)
        ret[key] += term
    return ret
