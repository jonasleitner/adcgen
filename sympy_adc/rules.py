from . import expr_container as e


class Rules:
    """Rule to apply to expressions. Currently it is only possible to
       define forbidden tensor blocks, i.e., only allow a certain subset
       of blocks in the expression and remove all terms that contain a
       forbidden block. Forbidden tensor blocks can be defined as:
       {'tensor_name': [block1, block2,...]}"""

    def __init__(self, forbidden_tensor_blocks=None):
        self.forbidden_blocks = forbidden_tensor_blocks

    def apply(self, expr):
        if not isinstance(expr, e.expr):
            raise TypeError(f"Expression needs to be provided as {e.expr}")
        # is there anything to do?
        if self.forbidden_blocks is None:
            return expr

        res = e.expr(0, **expr.assumptions)
        for term in expr.terms:
            # remove the forbidden blocks of tensors
            if any(obj.name in self.forbidden_blocks
                   and obj.space in self.forbidden_blocks[obj.name]
                   for obj in term.objects):
                continue
            res += term
        return res
