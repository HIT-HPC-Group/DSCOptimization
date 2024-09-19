import tvm
from tvm import relay
import numpy as np
import pandas as pd
from tvm.contrib import graph_executor

# Set the target device for ROCm (AMD GPU backend)
target = "rocm"  # Use "rocm" for AMD GPUs
dev = tvm.device(target, 0)

# Define the batch sizes and layer configurations
batch_sizes = [1, 8, 16, 32, 64]
paramList = [
    [32, 112, 112, 3, 1],
    [144, 56, 56, 3, 1],
    [192, 28, 28, 3, 1],
    [240, 28, 28, 5, 1],
    [384, 14, 14, 3, 1],
    [480, 14, 14, 3, 1],
    [480, 14, 14, 5, 1],
    [576, 14, 14, 3, 1],
    [672, 14, 14, 5, 1],
    [960, 7, 7, 3, 1],
    [1152, 7, 7, 3, 1],
    [1152, 7, 7, 5, 1],
    [96, 112, 112, 3, 2],
    [144, 56, 56, 3, 2],
    [144, 56, 56, 5, 2],
    [192, 28, 28, 3, 2],
    [240, 28, 28, 3, 2],
    [576, 14, 14, 3, 2],
    [672, 14, 14, 5, 2],
    [72, 56, 56, 3, 1],
    [120, 28, 28, 5, 1],
    [24, 28, 28, 3, 1],
    [48, 14, 14, 3, 1],
    [96, 7, 7, 3, 1],
    [48, 112, 112, 3, 2],
    [72, 56, 56, 5, 2],
    [576, 14, 14, 5, 2],
    [24, 56, 56, 3, 2],
    [48, 28, 28, 3, 2],
    [96, 14, 14, 3, 2]
]

# Create a list to store the results
results = []

# Run tests for each configuration and batch size
for params in paramList:
    in_channels, height, width, kernel_size, stride = params

    # Dictionary to store the configuration and profiling results for the current row
    result_dict = {
        "Input Channel": in_channels,
        "Input Height/Width": f"{height}x{width}",
        "Filter Height/Width": f"{kernel_size}x{kernel_size}",
        "Stride": stride
    }

    # Iterate over batch sizes
    for batch_size in batch_sizes:
        # Create input shapes and random data
        input_shape = (batch_size, in_channels, height, width)
        weight_shape = (in_channels, 1, kernel_size, kernel_size)
        data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
        weight_data = np.random.uniform(-1, 1, size=weight_shape).astype("float32")

        # Define depthwise convolution using Relay
        data_var = relay.var("data", shape=input_shape, dtype="float32")
        weight_var = relay.var("weight", shape=weight_shape, dtype="float32")

        depthwise_conv = relay.nn.conv2d(
            data_var, weight_var,
            strides=(stride, stride),
            padding=(kernel_size // 2, kernel_size // 2),  # Same padding for simplicity
            groups=in_channels,  # Set groups to the number of input channels for depthwise convolution
            channels=in_channels,
            kernel_size=(kernel_size, kernel_size),
            out_dtype="float32"
        )

        # Create a function for the Relay computation graph
        func = relay.Function([data_var, weight_var], depthwise_conv)

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
    "Input Channel", "Input Height/Width", "Filter Height/Width", "Stride",
    "Input Batch = 1 (us)", "Input Batch = 8 (us)", "Input Batch = 16 (us)",
    "Input Batch = 32 (us)", "Input Batch = 64 (us)"
]

# Reorder the DataFrame columns
df = df[columns_order]

# Save the DataFrame to an Excel file
df.to_excel("Depthwise_TVM_default_Result.xlsx", index=False)

print("Profiling results have been saved to 'Depthwise_TVM_default_Result.xlsx'.")