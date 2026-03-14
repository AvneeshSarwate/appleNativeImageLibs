"""Inspect RTMPose ONNX graph to understand backbone vs SimCC head structure."""

import onnx
import numpy as np

model = onnx.load("rtmpose_fixed.onnx")
graph = model.graph

print(f"Total nodes: {len(graph.node)}")
print(f"Inputs: {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in graph.input]}")
print(f"Outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in graph.output]}")

# Find the SimCC head — the outputs are simcc_x (1,133,576) and simcc_y (1,133,768)
# Trace backwards from outputs to find where the head starts
output_names = {o.name for o in graph.output}

# Build a map of node outputs → nodes
output_to_node = {}
for node in graph.node:
    for out in node.output:
        output_to_node[out] = node

# Trace back from outputs
def trace_back(name, depth=0):
    if name not in output_to_node:
        return
    node = output_to_node[name]
    prefix = "  " * depth
    shapes = ""
    print(f"{prefix}{node.op_type} [{node.name}] → {list(node.output)}")
    if depth < 15:  # limit depth
        for inp in node.input:
            if inp in output_to_node:
                trace_back(inp, depth + 1)

print("\n=== Trace back from simcc_x ===")
trace_back("simcc_x", 0)

print("\n=== Trace back from simcc_y ===")
trace_back("simcc_y", 0)

# Also list all op types and their counts
from collections import Counter
op_counts = Counter(n.op_type for n in graph.node)
print("\n=== Op type counts ===")
for op, count in op_counts.most_common():
    print(f"  {op}: {count}")

# Find the last conv/matmul ops — these are likely the SimCC head
print("\n=== Last 30 nodes (near output) ===")
for node in graph.node[-30:]:
    print(f"  {node.op_type:15s} {node.name:40s} inputs={list(node.input)[:2]}... → {list(node.output)}")
