import tvm
from tvm import te
from tvm import topi
import tvm.relay.op as _op
import tvm.relay.op.strategy as _strategy
from tvm.relay.op.strategy.generic import is_depthwise_conv2d


def default_compute(op) :
    return _strategy.generic.wrap_compute_conv2d(op)
def default_topi_schedule(sched) :
    return _strategy.generic.wrap_topi_schedule(sched)
def default_depthwise_compute(op) :
    return _strategy.generic.wrap_compute_conv2d(op, need_kernel_layout=True)
    

def xbar_compute_conv2d_nhwc(op) :
    return default_compute(op)
def xbar_compute_depthwise_conv2d_nhwc(op) :
    return default_depthwise_compute(op)


def xbar_topi_schedule_conv2d_nhwc(topi_schedule) :
    def wrapper(attrs, outs, target):
        with target:
            s = te.create_schedule([x.op for x in outs])
            
            conv_op = [stge for stge in s.stages if stge.op.tag == 'conv2d_nhwc']
            if len(conv_op) != 1 :
                raise Exception("Expected one conv2d operation in 'stages'.")
            conv_op = conv_op[0]
            
            if len(conv_op.all_iter_vars) == 7 :
                n, oh, ow, oc, kw, kh, ki = conv_op.all_iter_vars
                conv_op.pragma(var=n, pragma_type="outerloop")
            else :
                print("Conv2D NHWC has not enough axes due to optimization.")
            return s
    return wrapper


def xbar_topi_schedule_depthwise_conv2d_nhwc(topi_schedule) :
    def wrapper(attrs, outs, target):
        with target:
            s = te.create_schedule([x.op for x in outs])
            
            dw_conv_op = [stge for stge in s.stages if stge.op.tag == 'depthwise_conv2d_nhwc']
            if len(dw_conv_op) != 1 :
                raise Exception("Expected one depthwise_conv2d_nhwc operation in 'stages'.")
            dw_conv_op = dw_conv_op[0]
            
            if len(dw_conv_op.all_iter_vars) == 6 :
                n, oh, ow, oc, kw, kh = dw_conv_op.all_iter_vars
                dw_conv_op.pragma(var=n, pragma_type="outerloop")
            else :
                print("Depthwise Conv2D NHWC has not enough axes due to optimization.")
            return s
    return wrapper


@_strategy.generic.conv2d_strategy.register(["cpu"])
def conv2d_strategy_xbar(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    layout = attrs.data_layout
    groups = attrs.groups
    kernel_layout = attrs.kernel_layout
    data, kernel = inputs
    
    if (groups == 1) and (layout == "NHWC") :
        assert kernel_layout == "HWIO"
        
        strategy.add_implementation(
            xbar_compute_conv2d_nhwc(topi.nn.conv2d_nhwc),
            xbar_topi_schedule_conv2d_nhwc(topi.generic.schedule_conv2d_nhwc),
            name="conv2d_nhwc.generic",
        )
    
    elif (layout == "NHWC") and _strategy.generic.is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups) :
        strategy.add_implementation(
            xbar_compute_depthwise_conv2d_nhwc(topi.nn.depthwise_conv2d_nhwc),
            xbar_topi_schedule_depthwise_conv2d_nhwc(topi.generic.schedule_depthwise_conv2d_nhwc),
            name="depthwise_conv2d_nhwc.generic",
        )
    else :
        return _strategy.x86.conv2d_strategy_cpu(attrs, inputs, out_type, target)
    return strategy
