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
        # is there anything to do?
        if self.forbidden_blocks is None:
            return expr

        if not isinstance(expr, e.Expr):
            raise TypeError(f"Expression needs to be provided as {e.Expr}")

        res = e.Expr(0, **expr.assumptions)
        for term in expr.terms:
            # remove the forbidden blocks of tensors
            if any(obj.name in self.forbidden_blocks
                   and obj.space in self.forbidden_blocks[obj.name]
                   for obj in term.objects):
                continue
            res += term
        return res

    def __eq__(self, other):
        if not isinstance(other, Rules):
            return False

        # self is empty
        if self.forbidden_blocks is None or not self.forbidden_blocks:
            # both are empty
            if other.forbidden_blocks is None or not other.forbidden_blocks:
                return True
            else:  # only self is emtpy
                return False
        else:  # only other is empty
            if other.forbidden_blocks is None or not other.forbidden_blocks:
                return False

        # both not empty -> compare forbidden blocks
        for k, v in self.forbidden_blocks.items():
            if k not in other.forbidden_blocks:
                return False
            if sorted(v) != sorted(other.forbidden_blocks[k]):
                return False
        return True
