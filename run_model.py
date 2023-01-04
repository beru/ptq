import sys
import numpy as np
import onnxruntime 
import onnx
from onnx import numpy_helper
from PIL import Image
import ast

def preprocess(image):
  image = image.astype('float32')
  image = (image / 127.5) - 2
  image = np.expand_dims(image, axis=0)
  return image

def softmax(x):
  x = x.reshape(-1)
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)
  
def postprocess(result):
  return softmax(np.array(result)).tolist()

model_file = sys.argv[1]
input_file = sys.argv[2]
image = Image.open(input_file)
image = np.array(image).transpose(2, 0, 1)
X = preprocess(image)

ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X)
session = onnxruntime.InferenceSession(model_file)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
raw_result = session.run([output_name], {input_name: ortvalue})

res = postprocess(raw_result)
sort_idx = np.flip(np.squeeze(np.argsort(res)))

with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
  categories = ast.literal_eval(f.read())

for idx in sort_idx[:5]:
  print(idx, res[idx], categories[idx])

