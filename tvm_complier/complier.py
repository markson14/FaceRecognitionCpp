import mxnet as mx
import tvm
import tvm.relay as relay
PREFIX, EPOCH = "./model/mnet.25", 0
SIZE = (480, 640)


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
    relay_sym, relay_params = relay.frontend.from_mxnet(
        symbol=sym,
        shape=shape_dict,
        dtype="float32",
        arg_params=arg_params,
        aux_params=aux_params)
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build(
            relay_sym,
            target,
            params=relay_params)
    print(type(graph), type(lib), type(params))
    lib.export_library("./mnet.25.x86.cpu.so")
    print('lib export succeefully')
    with open("./mnet.25.x86.cpu.json", "w") as fo:
        fo.write(graph)
    with open("./mnet.25.x86.cpu.params", "wb") as fo:
        fo.write(relay.save_param_dict(params))


if __name__ == "__main__":
    main()
    print("-------------convert done!!!-------------")
