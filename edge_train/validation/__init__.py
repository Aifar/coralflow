"""TFLite model conversion and validation."""

from pathlib import Path


def convert_to_tflite(model_path: Path, output: Path) -> Path:
    """Convert a TensorFlow SavedModel to TFLite format."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tflite_model)
    return output


def check_size_constraint(size_mb: float, limit_mb: float = 10.0) -> bool:
    """Check model size is under the limit."""
    return size_mb <= limit_mb


def estimate_latency(tflite_path: Path, num_runs: int = 50) -> float:
    """Run dummy inference to estimate average latency in ms."""
    import time
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    dtype = input_details[0]["dtype"]
    if dtype == np.int32:
        dummy_input = np.zeros(input_shape, dtype=np.int32)
    elif dtype == np.uint8:
        dummy_input = np.zeros(input_shape, dtype=np.uint8)
    else:
        dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # warmup
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        timings.append((time.perf_counter() - start) * 1000)

    return float(np.median(timings))
