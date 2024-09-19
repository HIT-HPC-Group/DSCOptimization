import tvm
from tvm import relay
import numpy as np
import pandas as pd
from tvm.contrib import graph_executor

# Set the target to ROCm (AMD GPU)
target = "rocm"
dev = tvm.device(target, 0)

# Define the batch sizes
batch_sizes = [1, 8, 16, 32, 64]

# Define the layer configurations for pointwise convolution
parameterList = [
    [32, 112, 16], [16, 112, 96], [96, 56, 24], [24, 56, 144],
    [144, 56, 24], [144, 28, 32], [32, 28, 192], [192, 28, 32],
    [144, 28, 40], [40, 28, 240], [240, 28, 40], [192, 14, 64],
    [64, 14, 384], [384, 14, 64], [384, 14, 96], [96, 14, 576],
    [576, 14, 96], [240, 14, 80], [80, 14, 480], [480, 14, 80],
    [480, 14, 112], [112, 14, 672], [672, 14, 112], [576, 7, 160],
    [160, 7, 960], [960, 7, 160], [960, 7, 320], [320, 7, 1280],
    [672, 7, 192], [192, 7, 1152], [1152, 7, 192], [1152, 7, 320],
    [16, 112, 48], [48, 56, 24], [24, 56, 72], [72, 56, 24],
    [72, 28, 40], [40, 28, 120], [120, 28, 40], [480, 14, 96],
    [576, 7, 192], [24, 28, 24], [48, 14, 48], [96, 7, 96],
    [192, 7, 1024]
]

# Create a list to store the results
results = []

# Run tests for each configuration and batch size
for params in parameterList:
    in_channels, height, out_channels = params

    # Dictionary to store the configuration and profiling results for the current row
    result_dict = {
        "Input Channel": in_channels,
        "Input Height/Width": f"{height}x{height}",
        "Output Channel": out_channels
    }

    for batch_size in batch_sizes:
        print("Batch Size = " + str(batch_size) + ", Input Channel = " + str(in_channels) + ", Input Height = " + str(height) + ", Output Channel = " + str(out_channels))

        # Define input and weights
        input_shape = (batch_size, in_channels, height, height)
        weight_shape = (out_channels, in_channels, 1, 1)
        data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        weight_data = np.random.uniform(-1, 1, size=weight_shape).astype("float32")

        # Define pointwise convolution using Relay (1x1 kernel, no padding, stride 1)
        data_var = relay.var("data", shape=input_shape, dtype="float32")
        weight_var = relay.var("weight", shape=weight_shape, dtype="float32")

        pointwise_conv = relay.nn.conv2d(
            data_var, weight_var,
            strides=(1, 1),
            padding=(0, 0),  # No padding
            channels=out_channels,
            kernel_size=(1, 1),  # 1x1 kernel
            out_dtype="float32"
        )

        # Create a function for the Relay computation graph
        func = relay.Function([data_var, weight_var], pointwise_conv)

        # Build the Relay module
        mod = tvm.IRModule.from_expr(func)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target)

        # Create a graph executor
        module = graph_executor.GraphModule(lib["default"](dev))

        # Set inputs
        module.set_input("data", tvm.nd.array(data, dev))
        module.set_input("weight", tvm.nd.array(weight_data, dev))

        # Measure execution time using time_evaluator (in microseconds)
        evaluator = module.module.time_evaluator("run", dev, number=100)
        time_taken = evaluator().mean * 1e6  # Convert to microseconds
        result_dict[f"Input Batch = {batch_size} (us)"] = f"{time_taken:.2f}"

    # Append the result dictionary to the results list
    results.append(result_dict)

# Convert the results to a pandas DataFrame
df = pd.DataFrame(results)

# Specify the desired column order
columns_order = [
    "Input Channel", "Input Height/Width", "Output Channel",
    "Input Batch = 1 (us)", "Input Batch = 8 (us)", "Input Batch = 16 (us)",
    "Input Batch = 32 (us)", "Input Batch = 64 (us)"
]

# Reorder the DataFrame columns
df = df[columns_order]

# Save the DataFrame to an Excel file
df.to_excel("Pointwise_TVM_default_Result.xlsx", index=False)

print("Profiling results have been saved to 'Pointwise_TVM_default_Result.xlsx'.")