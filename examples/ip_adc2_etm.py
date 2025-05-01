from adcgen import (
    Operators, GroundState, IntermediateStates, Properties, simplify,
    remove_tensor, ExprContainer, tensor_names
)

# init the class structure.
h = Operators()
mp = GroundState(h)
isr = IntermediateStates(mp, variant="ip")
prop = Properties(isr)

# compute the transition moment X_I <I|p^+ q|0> d^p_q
expec = prop.trans_moment(adc_order=2)
# in a real orbital basis and simplify the expression
expec = ExprContainer(expec, real=True).substitute_contracted()
expec = simplify(expec)

# remove the operator matrix to obtain the transition dm
# The one particle operator is by named 'd' by default. The tensor names can
# be modified through 'adcgen/tensor_names.json' and accessed through the
# TensorNames class.
dm = remove_tensor(expec, tensor_names.operator)
for dm_block, dm_expr in dm.items():
    assert len(dm_block) == 1
    dm_block = dm_block[0]
    # remove the (left) adc vector to obtain the effective transition moments
    etm = remove_tensor(dm_expr, tensor_names.left_adc_amplitude)
    for etm_block, etm_expr in etm.items():
        assert len(etm_block) == 1
        etm_block = etm_block[0]
        print("\n", "#"*80, sep="")
        print(f"{len(etm_expr)} terms in etm block {etm_block}/{dm_block}:")
        print(etm_expr)
