import sys
import os
import numpy as np
import onnxruntime 
import onnx
from onnx import numpy_helper
import ctypes
from pprint import pprint
import mnist

import matplotlib as mpl
import matplotlib.pyplot as plt

def softmax(x):
  x = x.reshape(-1)
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def postprocess(result):
  return softmax(np.array(result)).tolist()

if len(sys.argv) < 2:
  print("usage: mnist_ptq.py model_file_path")
  sys.exit()

model_file_path = sys.argv[1]

session = onnxruntime.InferenceSession(model_file_path)
input_name = session.get_inputs()[0].name
outputs = session.get_outputs()
output_names = []
for output in outputs:
  output_names.append(output.name)

#np.set_printoptions(linewidth=100000)
#pprint(test_set_images[0])

data = mnist.Data()
data.load()

data.test_set_images = data.test_set_images.astype(np.float32) / 255
hdr = data.test_set_image_file_header

minmax_map = {}

float_min = sys.float_info.min
float_max = sys.float_info.max

class MinMax:
  def __init__(self):
    self.min = float_max
    self.max = float_min

  def __repr__(self):
    return f"min:{self.min}, max:{self.max}"

# find min max
for i in range(hdr.num_images):
  
  input = data.test_set_images[i]
  ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input)
  results = session.run(output_names, {input_name: ortvalue})
  
  for j in range(len(output_names)):
    name = output_names[j]
    result = results[j]
    shape = result.shape
    if len(shape) != 4 or shape[0] != 1:
      continue
    arr = result.ravel()
    
    if name not in minmax_map:
      minmax_map[name] = MinMax()
    
    minmax = minmax_map[name]
    minmax.min = min(minmax.min, arr.min())
    minmax.max = max(minmax.max, arr.max())

hist_map = {}

class HistResult:
  def __init__(self, hist, bin_edges):
    self.hist = hist
    self.bin_edges = bin_edges

num_bins = 32

# collect histograms
for i in range(hdr.num_images):
  
  input = data.test_set_images[i]
  ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(input)
  results = session.run(output_names, {input_name: ortvalue})
  
  for j in range(len(output_names)):
    name = output_names[j]
    if not name in minmax_map:
      continue
    
    result = results[j]
    arr = result.ravel()
    minmax = minmax_map[name]
    
    hist, bin_edges = np.histogram(arr, bins=num_bins, range=(minmax.min, minmax.max))
    
    if not name in hist_map:
      hist_map[name] = HistResult(hist, bin_edges)
    else:
      hist_map[name].hist += hist

mpl.rcParams['axes.linewidth'] = 0.1 #set the value globally

for j in range(len(output_names)):
  name = output_names[j]
  if not name in minmax_map:
    continue

  minmax = minmax_map[name]
  hist = hist_map[name]
  print(f"{name}: {minmax}")
  #print(hist)
  
  width = (minmax.max - minmax.min) / num_bins * 0.7
  plt.clf()
  plt.title(name)
  plt.bar(hist.bin_edges[1:], hist.hist, align='center', width=width)
  plt.savefig(f"{name}.svg", dpi=300)
  
