import sys
import os
import numpy as np
import onnxruntime 
import onnx
from onnx import numpy_helper
from pprint import pprint
import mnist

def softmax(x):
  x = x.reshape(-1)
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def postprocess(result):
  return softmax(np.array(result)).tolist()

model_file_path = "./models/mnist-12.onnx"

session = onnxruntime.InferenceSession(model_file_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

#np.set_printoptions(linewidth=100000)
#pprint(test_set_images[0])

data = mnist.Data()
data.load()

data.test_set_images = data.test_set_images.astype(np.float32) / 255

matched_cnt = 0

hdr = data.test_set_image_file_header

for i in range(hdr.num_images):
  
  input = data.test_set_images[i]
  ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input)
  raw_result = session.run([output_name], {input_name: ortvalue})
  res = postprocess(raw_result)
  res = np.array(res)
  sorted_idx = np.argsort(res)[::-1]
  if sorted_idx[0] == data.test_set_labels[i]:
    matched_cnt += 1
  
print(f"{matched_cnt*100/hdr.num_images}")

