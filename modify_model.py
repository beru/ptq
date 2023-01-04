import sys
import onnx
import onnxruntime
from onnx import numpy_helper

argc = len(sys.argv)

if argc < 3:
  print("usage: modify_model.py load_onnx_path save_onnx_path")
  sys.exit()

load_onnx_path = sys.argv[1]
save_onnx_path = sys.argv[2]

model = onnx.load(load_onnx_path)

ort_session = onnxruntime.InferenceSession(load_onnx_path)
org_outputs = [x.name for x in ort_session.get_outputs()]

model = onnx.shape_inference.infer_shapes(model)

value_info_map = {}
for value_info in model.graph.value_info:
  value_info_map[value_info.name] = value_info

for node in model.graph.node:
  for output in node.output:
    if output in org_outputs:
      continue
    
    value_info = value_info_map[output]
    tensor_type = value_info.type.tensor_type
    shape = []
    for dim in tensor_type.shape.dim:
      if getattr(dim, "dim_param"):
        shape.append(dim.dim_param)
      elif getattr(dim, "dim_value"):
        shape.append(dim.dim_value)
    
    model.graph.output.extend([
      onnx.helper.make_tensor_value_info(
        name=output,
        elem_type=tensor_type.elem_type,
        shape=shape,
      )
    ])

onnx.save(model, save_onnx_path)

