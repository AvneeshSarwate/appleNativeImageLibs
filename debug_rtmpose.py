"""Debug RTMPose conversion: check onnx2torch accuracy and CoreML output ordering."""

import onnx
import numpy as np
import torch
import onnxruntime as ort
import coremltools as ct
from onnx2torch import convert as onnx2torch_convert

ONNX_PATH = "rtmpose_fixed.onnx"

# Create a test input
np.random.seed(42)
test_input = np.random.randn(1, 3, 384, 288).astype(np.float32)

# 1. Run ONNX
print("=== ONNX Runtime ===")
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
onnx_outs = sess.run(None, {"input": test_input})
print(f"  simcc_x shape: {onnx_outs[0].shape}, range: [{onnx_outs[0].min():.4f}, {onnx_outs[0].max():.4f}]")
print(f"  simcc_y shape: {onnx_outs[1].shape}, range: [{onnx_outs[1].min():.4f}, {onnx_outs[1].max():.4f}]")

# 2. Run onnx2torch PyTorch model
print("\n=== onnx2torch PyTorch ===")
torch_model = onnx2torch_convert(ONNX_PATH)
torch_model.eval()
with torch.no_grad():
    torch_outs = torch_model(torch.from_numpy(test_input))
    if isinstance(torch_outs, (list, tuple)):
        for i, t in enumerate(torch_outs):
            print(f"  output[{i}] shape: {t.shape}, range: [{t.min():.4f}, {t.max():.4f}]")
            err = np.abs(t.numpy() - onnx_outs[i]).max()
            print(f"    vs ONNX max abs error: {err:.6f}")
    else:
        print(f"  shape: {torch_outs.shape}")

# 3. Run CoreML and check output mapping
print("\n=== CoreML ===")
cml = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.ALL)
spec = cml.get_spec()
cml_input_name = spec.description.input[0].name
cml_out_names = [o.name for o in spec.description.output]
print(f"  Input: {cml_input_name}")
print(f"  Outputs: {cml_out_names}")

cml_outs = cml.predict({cml_input_name: test_input})
for name in sorted(cml_outs.keys()):
    val = cml_outs[name]
    print(f"\n  {name}: shape={val.shape}, range=[{val.min():.4f}, {val.max():.4f}]")
    # Try to match with ONNX outputs by shape
    for i, oname in enumerate(["simcc_x", "simcc_y"]):
        if val.shape == onnx_outs[i].shape:
            err = np.abs(val - onnx_outs[i]).max()
            # Also check argmax agreement
            onnx_argmax = np.argmax(onnx_outs[i][0], axis=-1)
            cml_argmax = np.argmax(val[0], axis=-1)
            kpt_err = np.abs(onnx_argmax.astype(float) - cml_argmax.astype(float))
            print(f"    matches {oname} shape — max_abs_err: {err:.6f}, "
                  f"max_kpt_err: {kpt_err.max():.0f}, mean_kpt_err: {kpt_err.mean():.1f}")

# 4. Check if float32 CoreML does better
print("\n=== CoreML (recheck with CPU_ONLY) ===")
cml_cpu = ct.models.MLModel("rtmpose.mlpackage", compute_units=ct.ComputeUnit.CPU_ONLY)
cml_cpu_outs = cml_cpu.predict({cml_input_name: test_input})
for name in sorted(cml_cpu_outs.keys()):
    val = cml_cpu_outs[name]
    for i, oname in enumerate(["simcc_x", "simcc_y"]):
        if val.shape == onnx_outs[i].shape:
            err = np.abs(val - onnx_outs[i]).max()
            cml_argmax = np.argmax(val[0], axis=-1)
            onnx_argmax = np.argmax(onnx_outs[i][0], axis=-1)
            kpt_err = np.abs(onnx_argmax.astype(float) - cml_argmax.astype(float))
            print(f"  {name} matches {oname} — max_abs_err: {err:.6f}, "
                  f"max_kpt_err: {kpt_err.max():.0f}")
