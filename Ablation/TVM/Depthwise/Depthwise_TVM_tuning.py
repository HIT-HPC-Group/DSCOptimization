import tvm
from tvm import relay, autotvm
import numpy as np
import pandas as pd
from tvm.contrib import graph_executor

# Set the target to ROCm (AMD GPU)
target = "rocm"
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

# File for tuning logs
log_file = "conv2d_tuning.log"

# Define tuning options
tuning_option = {
    "tuner": "xgb",  # Tuner algorithm
    "trials": 1000,  # Number of tuning trials (adjust based on hardware/resources)
    "early_stopping": 300,  # Stop early if no better schedule is found after a number of trials
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1, min_repeat_ms=1000, timeout=10)
    ),
}

# Function to run the tuning
def tune_tasks(tasks):
    for i, task in enumerate(tasks):
        print(f"Tuning task {i+1}/{len(tasks)}")
        # Dynamically select the tuner based on tuning_option['tuner']
        if tuning_option['tuner'] == 'random':
            tuner = autotvm.tuner.RandomTuner(task)
        elif tuning_option['tuner'] == 'xgb':
            tuner = autotvm.tuner.XGBTuner(task, verbose=0)
        elif tuning_option['tuner'] == 'ga':
            tuner = autotvm.tuner.GATuner(task)
        elif tuning_option['tuner'] == 'gridsearch':
            tuner = autotvm.tuner.GridSearchTuner(task)
        else:
            raise ValueError(f"Unknown tuner: {tuning_option['tuner']}")
        tuner.tune(
            n_trial=tuning_option["trials"],
            early_stopping=tuning_option["early_stopping"],
            measure_option=tuning_option["measure_option"],
            callbacks=[
                autotvm.callback.progress_bar(tuning_option["trials"]),
                autotvm.callback.log_to_file(log_file),
            ],
        )

# Function to test and profile performance after tuning
def profile_and_save_to_excel():
    results = []  # Store results

    for params in paramList:
        in_channels, height, width, kernel_size, stride = params

        result_dict = {
            "Input Channel": in_channels,
            "Input Height/Width": f"{height}x{width}",
            "Filter Height/Width": f"{kernel_size}x{kernel_size}",
            "Stride": stride
        }

        for batch_size in batch_sizes:
            # Define input and weights
            input_shape = (batch_size, in_channels, height, width)
            weight_shape = (in_channels, 1, kernel_size, kernel_size)
            data = np.random.uniform(-1, 1, size=input_shape).astype("float32")
            weight_data = np.random.uniform(-1, 1, size=weight_shape).astype("float32")

            # Define depthwise convolution in Relay
            data_var = relay.var("data", shape=input_shape, dtype="float32")
            weight_var = relay.var("weight", shape=weight_shape, dtype="float32")

            depthwise_conv = relay.nn.conv2d(
                data_var, weight_var,
                strides=(stride, stride),
                padding=(kernel_size // 2, kernel_size // 2),  # Same padding
                groups=in_channels,  # Depthwise convolution
                channels=in_channels,
                kernel_size=(kernel_size, kernel_size),
                out_dtype="float32"
            )

            # Create Relay module
            func = relay.Function([data_var, weight_var], depthwise_conv)
            mod = tvm.IRModule.from_expr(func)

            # Extract tasks for tuning
            tasks = autotvm.task.extract_from_program(mod["main"], target=target, params={})

            # Tune the tasks if not already tuned
            tune_tasks(tasks)

            # Apply the best tuning results
            with autotvm.apply_history_best(log_file):
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(mod, target=target)

            # Create graph executor and set inputs
            module = graph_executor.GraphModule(lib["default"](dev))
            module.set_input("data", tvm.nd.array(data, dev))
            module.set_input("weight", tvm.nd.array(weight_data, dev))

            # Measure performance using time_evaluator
            evaluator = module.module.time_evaluator("run", dev, number=100)
            time_taken = evaluator().mean * 1e6  # Convert to microseconds
            result_dict[f"Input Batch = {batch_size} (us)"] = f"{time_taken:.2f}"

        # Append result to list
        results.append(result_dict)

    # Create pandas DataFrame from results
    df = pd.DataFrame(results)

    # Specify the desired column order
    columns_order = [
        "Input Channel", "Input Height/Width", "Filter Height/Width", "Stride",
        "Input Batch = 1 (us)", "Input Batch = 8 (us)", "Input Batch = 16 (us)",
        "Input Batch = 32 (us)", "Input Batch = 64 (us)"
    ]
    df = df[columns_order]

    # Save to Excel
    df.to_excel("Depthwise_TVM_Result_Tuned.xlsx", index=False)
    print("Tuned profiling results have been saved to 'Depthwise_TVM_Result_Tuned.xlsx'.")

# Run the tuning and profiling process
profile_and_save_to_excel()