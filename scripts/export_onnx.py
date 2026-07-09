"""
scripts/export_onnx.py

Export the best Optuna trial model to ONNX format for production deployment.

ONNX benefits:
- Platform-independent inference (Python, C++, Java, mobile)
- 2-10x faster inference than sklearn on CPU
- Compatible with ONNX Runtime, TensorRT, CoreML, TFLite

Usage:
    python scripts/export_onnx.py --model-path models/best_model.pkl --output models/best_model.onnx

Research note:
    ONNX export enables deployment benchmarks (latency, throughput)
    which are a common evaluation metric in applied ML papers.
"""
import argparse
import pickle
import numpy as np
import os

def export_to_onnx(model_path: str, output_path: str, n_features: int = 5000):
    try:
        import skl2onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        print("Install skl2onnx: pip install skl2onnx onnxruntime")
        return False

    # Load sklearn model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model type: {type(model).__name__}")

    # Convert to ONNX
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type, verbose=0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_kb = os.path.getsize(output_path) / 1024
    print(f"✅ ONNX model saved: {output_path} ({size_kb:.1f} KB)")
    return True


def benchmark_inference(model_path: str, onnx_path: str, n_features: int = 5000):
    """Compare sklearn vs ONNX inference speed."""
    import time, pickle
    try:
        import onnxruntime as rt
    except ImportError:
        print("Install onnxruntime: pip install onnxruntime")
        return

    X_test = np.random.rand(1000, n_features).astype(np.float32)

    # sklearn
    with open(model_path, "rb") as f:
        sk_model = pickle.load(f)
    t0 = time.perf_counter()
    sk_model.predict(X_test)
    sk_time = time.perf_counter() - t0

    # ONNX Runtime
    sess = rt.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    sess.run(None, {input_name: X_test})
    onnx_time = time.perf_counter() - t0

    speedup = sk_time / onnx_time
    print(f"\n── Inference Benchmark (n=1000 samples) ──")
    print(f"sklearn:      {sk_time*1000:.1f} ms")
    print(f"ONNX Runtime: {onnx_time*1000:.1f} ms")
    print(f"Speedup:      {speedup:.1f}x {'🚀 faster' if speedup > 1 else '(no speedup)'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  default="models/best_model.pkl")
    parser.add_argument("--output",      default="models/best_model.onnx")
    parser.add_argument("--n-features",  type=int, default=5000)
    parser.add_argument("--benchmark",   action="store_true")
    args = parser.parse_args()

    ok = export_to_onnx(args.model_path, args.output, args.n_features)
    if ok and args.benchmark:
        benchmark_inference(args.model_path, args.output, args.n_features)


if __name__ == "__main__":
    main()
