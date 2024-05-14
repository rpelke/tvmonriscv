from tvm import relay
from tvm.relay.dataflow_pattern import *


class MaxPool2DCallback(DFPatternCallback):
    """This is an example on how to replace patterns on a relay level.
    Call this class using: mod["main"] = rewrite(MaxPool2DCallback(), mod["main"])
    """
    def __init__(self, require_type=False, rewrite_once=False):
        super().__init__(require_type, rewrite_once)
        self.pool2d_input = wildcard()
        self.attr_dict = dict(
            pool_size=[2,2],
            strides=[2,2],
            dilation=[1, 1],
            padding=[0,0,0,0],
            layout="NHWC",
            out_layout="",
            ceil_mode=False
        )
        self.pattern = is_op("nn.max_pool2d")(self.pool2d_input).has_attr(attrs=self.attr_dict)

    def callback(self, pre, post, node_map):
        input = node_map[self.pool2d_input][0]
        return relay.op.nn.max_pool2d(data=input, **self.attr_dict)
