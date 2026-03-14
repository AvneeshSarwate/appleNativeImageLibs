"""Convert RTMPose with mixed precision: fp16 conv layers, fp32 everything else.

Conv layers are the backbone (122 ops, 95%+ of compute) — safe in fp16.
All other ops (mul, sigmoid, matmul, div, etc.) stay fp32 to preserve
SimCC argmax accuracy.
"""

import torch
import numpy as np
import coremltools as ct
import onnxruntime as ort
from onnx2torch import convert as onnx2torch_convert

ONNX_PATH = "rtmpose_fixed.onnx"
OUTPUT_PATH = "rtmpose_mixed.mlpackage"

print("Step 1: ONNX → PyTorch → Trace...")
torch_model = onnx2torch_convert(ONNX_PATH)
torch_model.eval()
dummy = torch.randn(1, 3, 384, 288)
with torch.no_grad():
    traced = torch.jit.trace(torch_model, dummy)

print("Step 2: Converting with mixed precision (fp16 conv, fp32 rest)...")


def op_selector(op):
    """Only conv ops go to fp16. Everything else stays fp32."""
    return op.op_type == "conv"


mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(1, 3, 384, 288))],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS14,
    compute_precision=ct.transform.FP16ComputePrecision(
        op_selector=op_selector
    ),
)
mlmodel.save(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")

# Quick accuracy check
print("\nStep 3: Accuracy check (random input)...")
onnx_sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
mixed = ct.models.MLModel(OUTPUT_PATH, compute_units=ct.ComputeUnit.ALL)
spec = mixed.get_spec()
in_name = spec.description.input[0].name
out_names = sorted([o.name for o in spec.description.output])

np.random.seed(42)
test_input = np.random.randn(1, 3, 384, 288).astype(np.float32)
onnx_outs = onnx_sess.run(None, {"input": test_input})
mixed_outs = mixed.predict({in_name: test_input})

for i, name in enumerate(["simcc_x", "simcc_y"]):
    onnx_am = np.argmax(onnx_outs[i][0], axis=-1)
    mixed_am = np.argmax(mixed_outs[out_names[i]][0], axis=-1)
    kpt_err = np.abs(onnx_am.astype(float) - mixed_am.astype(float))
    exact = (kpt_err == 0).sum()
    print(f"  {name}: exact={exact}/{len(kpt_err)} ({100*exact/len(kpt_err):.1f}%)  max_err={kpt_err.max():.0f}px")

print("\nDone.")
