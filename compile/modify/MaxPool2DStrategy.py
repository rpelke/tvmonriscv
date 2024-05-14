from tvm import te
from tvm.topi.x86 import schedule_pool
from tvm.target import get_native_generic_func
import tvm.relay.op.op as _op
import tvm.relay.op.strategy as _strategy


def _pool_mod(outs, layout) :
    """TODO: Modify pooling scheduling function
    Fallback function: Default scheduling function 'schedule_pool' in 'tvm.topi.x86.py'.
    """
    s = te.create_schedule([x.op for x in outs])
    pool_max = [stge for stge in s.stages if stge.op.tag == 'pool_max']
    if len(pool_max) != 1 :
        raise Exception("Expected one pool_max operation in 'stages'.")
    pool_max = pool_max[0]
    return schedule_pool(outs, layout)

    
@_strategy.x86.schedule_pool.register("cpu", override=True)
def schedule_pool_cpu(attrs, outs, target):
    """Register scheduling strategy for pooling function and CPU target.
    The 'schedule_pool' function in 'tvm.relay.op.strategy.generic.py' has to be called first.
    The 'override' parameter allows to override the default scheduling function 'schedule_pool_cpu' in 'tvm.relay.op.strategy.x86.py'.
    """
    with target:
        return _pool_mod(outs, attrs.layout)


def _create_fstrategy_from_schedule(op_name, schedule):
    assert hasattr(schedule, "dispatch_dict")
    compute = _op.get(op_name).get_attr("FTVMCompute")
    assert compute is not None, f"FTVMCompute is not registered for op {op_name}"
    fstrategy = get_native_generic_func(f"{op_name}_strategy")
    name_pfx = schedule.__name__
    name_pfx = name_pfx[name_pfx.index("_") + 1 :]
    fstrategy.set_default(
        _op._wrap_default_fstrategy(compute, schedule.fdefault, f"{name_pfx}.generic"), allow_override=True
    )
    for key, sch in schedule.dispatch_dict.items():
        fstrategy.register(_op._wrap_default_fstrategy(compute, sch, f"{name_pfx}.{key}"), [key], allow_override=True)
    return fstrategy


def register_schedule_override(op_name, schedule, level=10):
    """This function registers a schedule but overrides an already registered schedule.
    A similar functionality can be seen in 'tvm.relay.op.op.py' (but without the possibility to rewrite).
    """
    fstrategy = _create_fstrategy_from_schedule(op_name, schedule)
    return _op.register_strategy(op_name, fstrategy, level)


"""Register schedule for relay operation 'nn.max_pool2d'.
The level must be >10 since the default schedule has been registered with level 10 in 'tvm.relay.op.nn._nn'.
"""
register_schedule_override("nn.max_pool2d", _strategy.x86.schedule_pool, level=11)
