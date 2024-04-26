import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory"""
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(
        batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
    )
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


def deserializing_engine(engine_file):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file, "rb") as f:
        serialized_engine = f.read()
    return runtime.deserialize_cuda_engine(serialized_engine)


def load_tensorrt(engine_path, info=None):
    engine = deserializing_engine(engine_path)
    return engine, info

def infer_tensorrt(inps, engine, info):

    shape_of_output = info["output_shape"]
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(
        engine
    ) 
    inputs[0].host = inps[0].reshape(-1)
    trt_outputs = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
    feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)
    return feat, info
