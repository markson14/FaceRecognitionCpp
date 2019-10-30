import numpy as np
import tvm
from tvm import relay
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.contrib.quantization import quantize_model
import nnvm
import json

MODEL = "mnet.25"
PREFIX, EPOCH = "../models/%s/mnet.25" % (MODEL), 0
SIZE = (320, 320)


def main():
    """
    Load model from MXNet and complier to TVM model
    """
    sym, arg_params, aux_params = mx.model.load_checkpoint(PREFIX, EPOCH)
    opt_level = 3
    shape_dict = {'data': (1, 3, *SIZE)}

    # "target" means your target platform you want to compile.
    # better than "llvm" on MacOS
    target = tvm.target.create("llvm -mcpu=haswell")

    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(
        sym, arg_params, aux_params)
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(
            nnvm_sym, target, shape_dict, params=nnvm_params)
    print(type(graph), type(lib), type(params))
    lib.export_library("./deploy_lib_%s_%d.so" % (MODEL, EPOCH))
    print('lib export succeefully')
    with open("./deploy_graph_%s_%d.json" % (MODEL, EPOCH), "w") as fo:
        fo.write(graph.json())
    with open("./deploy_param_%s_%d.params" % (MODEL, EPOCH), "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))


if if __name__ == "__main__":
    main()
