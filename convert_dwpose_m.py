"""Convert DWPose-m (133 keypoints, 256x192) to CoreML fp32."""

import onnx
import onnxsim
import torch
import coremltools as ct
from onnx2torch import convert as onnx2torch_convert

ONNX_PATH = "dwpose_m.onnx"
OUTPUT_PATH = "dwpose_m.mlpackage"

# Fix Clip nodes (same issue as the larger model)
print("Fixing and simplifying ONNX...")
model = onnx.load(ONNX_PATH)
graph = model.graph
fixed = 0
for node in graph.node:
    if node.op_type == "Clip":
        while len(node.input) < 3:
            node.input.append("")
        if node.input[2] == "":
            max_name = f"{node.name}_max_const"
            max_tensor = onnx.helper.make_tensor(max_name, onnx.TensorProto.FLOAT, [], [3.4028235e+38])
            graph.initializer.append(max_tensor)
            node.input[2] = max_name
            fixed += 1
        if node.input[1] == "":
            min_name = f"{node.name}_min_const"
            min_tensor = onnx.helper.make_tensor(min_name, onnx.TensorProto.FLOAT, [], [0.0])
            graph.initializer.append(min_tensor)
            node.input[1] = min_name
            fixed += 1
print(f"  Fixed {fixed} Clip inputs")

model_sim, ok = onnxsim.simplify(model)
if ok:
    model = model_sim
FIXED_PATH = "dwpose_m_fixed.onnx"
onnx.save(model, FIXED_PATH)

# Convert ONNX → PyTorch → CoreML
print("ONNX → PyTorch...")
torch_model = onnx2torch_convert(FIXED_PATH)
torch_model.eval()

print("Tracing...")
dummy = torch.randn(1, 3, 256, 192)
with torch.no_grad():
    traced = torch.jit.trace(torch_model, dummy)
    test_out = traced(dummy)
    if isinstance(test_out, (list, tuple)):
        print(f"  Outputs: {[t.shape for t in test_out]}")

print("Converting to CoreML (fp32)...")
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(1, 3, 256, 192))],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.macOS14,
    compute_precision=ct.precision.FLOAT32,
)
mlmodel.save(OUTPUT_PATH)
print(f"Saved: {OUTPUT_PATH}")

# Quick accuracy check
print("\nAccuracy check...")
import numpy as np
import onnxruntime as ort

onnx_sess = ort.InferenceSession(FIXED_PATH, providers=["CPUExecutionProvider"])
cml = ct.models.MLModel(OUTPUT_PATH, compute_units=ct.ComputeUnit.ALL)
spec = cml.get_spec()
cml_in = spec.description.input[0].name
cml_outs = sorted([o.name for o in spec.description.output])

np.random.seed(42)
test = np.random.randn(1, 3, 256, 192).astype(np.float32)
onnx_outs = onnx_sess.run(None, {"input": test})
cml_preds = cml.predict({cml_in: test})

for i, name in enumerate(["simcc_x", "simcc_y"]):
    o_am = np.argmax(onnx_outs[i][0], axis=-1)
    c_am = np.argmax(cml_preds[cml_outs[i]][0], axis=-1)
    err = np.abs(o_am.astype(float) - c_am.astype(float))
    print(f"  {name}: exact={int((err==0).sum())}/{len(err)}  max_err={err.max():.0f}")

# Speed check
import time
for label, units in [("ALL", ct.ComputeUnit.ALL), ("CPU+GPU", ct.ComputeUnit.CPU_AND_GPU),
                      ("CPU+NE", ct.ComputeUnit.CPU_AND_NE)]:
    m = ct.models.MLModel(OUTPUT_PATH, compute_units=units)
    s = m.get_spec()
    n = s.description.input[0].name
    for _ in range(10): m.predict({n: test})
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        m.predict({n: test})
        times.append(time.perf_counter() - t0)
    print(f"  {label:8s}: {np.median(times)*1000:.2f}ms")
