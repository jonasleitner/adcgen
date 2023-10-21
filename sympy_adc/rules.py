from . import expr_container as e


class Rules:
    """Rule to apply to expressions. Currently it is only possible to
       define forbidden tensor blocks, i.e., only allow a certain subset
       of blocks in the expression and remove all terms that contain a
       forbidden block. Forbidden tensor blocks can be defined as:
       {'tensor_name': [block1, block2,...]}"""

    def __init__(self, forbidden_tensor_blocks=None):
        self._forbidden_blocks = forbidden_tensor_blocks

    def apply(self, expr):
        """Applies the rules to the provided expression."""

        if not isinstance(expr, e.Expr):
            raise TypeError(f"Expression needs to be provided as {e.Expr}")
        if self.is_empty:  # nothing to do
            return expr

        res = e.Expr(0, **expr.assumptions)
        for term in expr.terms:
            # remove the forbidden blocks of tensors
            if any(obj.name in self._forbidden_blocks
                   and obj.space in self._forbidden_blocks[obj.name]
                   for obj in term.objects):
                continue
            res += term
        return res

    @property
    def is_empty(self) -> bool:
        return not bool(self._forbidden_blocks)

    def __eq__(self, other):
        if not isinstance(other, Rules):
            return False

        empty, other_empty = self.is_empty, other.is_empty
        if empty and other_empty:  # both are empty
            return True
        elif empty:  # only self is empty
            return False
        elif other_empty:  # only other is empty
            return False

        # both not empty -> compare forbidden blocks
        for k, v in self._forbidden_blocks.items():
            if k not in other._forbidden_blocks:
                return False
            if sorted(v) != sorted(other._forbidden_blocks[k]):
                return False
        return True
