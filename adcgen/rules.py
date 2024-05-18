from . import expr_container as e


class Rules:
    """
    Rules to apply to expressions.

    Parameters
    ----------
    forbidden_tensor_blocks : dict, optional
        Tensor blocks to remove from an expression, i.e., only allow
        a certain subset of blocks in the expression. A dictionary of the form
        {tensor_name: [block1, block2, ...]}
        is expected.
    """

    def __init__(self, forbidden_tensor_blocks: dict[str, list[str]] = None):
        self._forbidden_blocks = forbidden_tensor_blocks

    def apply(self, expr: e.Expr) -> e.Expr:
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
        elif empty or other_empty:  # only self or other is empty
            return False

        # both not empty -> compare forbidden blocks (keys and values)
        if self._forbidden_blocks.keys() != other._forbidden_blocks.keys():
            return False
        if any(sorted(v) != sorted(other._forbidden_blocks[k])
               for k, v in self._forbidden_blocks.items()):
            return False
        return True
