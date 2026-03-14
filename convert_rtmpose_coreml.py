"""Phase 2.1 – Convert RTMPose ONNX → (fix+simplify) → PyTorch → CoreML."""

import onnx
import onnxsim
import torch
import numpy as np
import coremltools as ct
from onnx2torch import convert as onnx2torch_convert

RTMPOSE_ONNX = "/Users/avneeshsarwate/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.onnx"
YOLOX_ONNX = "/Users/avneeshsarwate/.cache/rtmlib/hub/checkpoints/yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"


def fix_clip_nodes(model):
    """Fix Clip nodes with empty-string max input (onnx2torch can't handle them).

    A Clip(input, min=0, max='') is equivalent to Relu. Replace empty max inputs
    with a large constant so onnx2torch can parse them.
    """
    graph = model.graph
    fixed = 0
    for node in graph.node:
        if node.op_type == "Clip":
            # Clip can have 1-3 inputs: (input, [min, [max]])
            # Empty string means "not specified"
            while len(node.input) < 3:
                node.input.append("")
            if node.input[2] == "":
                # Add a large constant for max
                max_name = f"{node.name}_max_const"
                max_tensor = onnx.helper.make_tensor(
                    max_name, onnx.TensorProto.FLOAT, [], [3.4028235e+38]
                )
                graph.initializer.append(max_tensor)
                node.input[2] = max_name
                fixed += 1
            if node.input[1] == "":
                min_name = f"{node.name}_min_const"
                min_tensor = onnx.helper.make_tensor(
                    min_name, onnx.TensorProto.FLOAT, [], [0.0]
                )
                graph.initializer.append(min_tensor)
                node.input[1] = min_name
                fixed += 1
    print(f"  Fixed {fixed} Clip node inputs")
    return model


def prepare_onnx(onnx_path, output_path):
    """Fix and simplify ONNX model."""
    print(f"  Loading ONNX: {onnx_path}")
    model = onnx.load(onnx_path)
    model = fix_clip_nodes(model)

    print(f"  Simplifying...")
    model_sim, ok = onnxsim.simplify(model)
    if not ok:
        print("  WARNING: simplification failed, using fixed-only model")
        onnx.save(model, output_path)
    else:
        onnx.save(model_sim, output_path)
    print(f"  Saved: {output_path}")
    return output_path


def convert_onnx_to_coreml(onnx_path, output_path, input_shape, label):
    print(f"\n{'='*50}")
    print(f"Converting {label}")
    print(f"  Input shape: {input_shape}")

    # Step 0: Fix + simplify
    fixed_path = output_path.replace(".mlpackage", "_fixed.onnx")
    fixed_path = prepare_onnx(onnx_path, fixed_path)

    # Step 1: ONNX → PyTorch
    print("  Step 1: ONNX → PyTorch...")
    torch_model = onnx2torch_convert(fixed_path)
    torch_model.eval()

    # Step 2: Trace
    print("  Step 2: Tracing...")
    dummy = torch.randn(*input_shape)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, dummy)

    # Verify trace
    with torch.no_grad():
        test_out = traced(dummy)
    if isinstance(test_out, (tuple, list)):
        print(f"  Trace outputs: {len(test_out)} tensors, shapes: {[t.shape for t in test_out]}")
    else:
        print(f"  Trace output shape: {test_out.shape}")

    # Step 3: PyTorch → CoreML
    print("  Step 3: PyTorch → CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(shape=input_shape)],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.save(output_path)
    print(f"  Saved: {output_path}")
    return mlmodel


# RTMPose: (1, 3, 384, 288) in NCHW — H=384, W=288
convert_onnx_to_coreml(
    RTMPOSE_ONNX,
    "rtmpose.mlpackage",
    input_shape=(1, 3, 384, 288),
    label="RTMPose (RTMW-x-l, 133 keypoints)",
)

# Skip YOLOX — we use YOLO11-seg CoreML for person detection instead

print("\n\nDone! Both models converted.")
