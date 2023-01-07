import sys
import os
import numpy as np
import onnxruntime 
import onnx
from onnx import numpy_helper
from PIL import Image
import ast
import imagenet

def preprocess(image):
  image = image.astype('float32')
  image = image / 255.0
  image = image - np.array([0.485, 0.456, 0.406], dtype = 'float32', ndmin=3).T
  image = image / np.array([0.229, 0.224, 0.225], dtype = 'float32', ndmin=3).T
  
  image = np.expand_dims(image, axis=0)
  return image

def softmax(x):
  x = x.reshape(-1)
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)
  
def postprocess(result):
  return softmax(np.array(result)).tolist()

def crop_center(pil_img, crop_width, crop_height):
  img_width, img_height = pil_img.size
  return pil_img.crop(((img_width - crop_width) // 2,
                       (img_height - crop_height) // 2,
                       (img_width + crop_width) // 2,
                       (img_height + crop_height) // 2))

def crop_max_square(pil_img):
  return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

if len(sys.argv) < 2:
  print(f"usage: python mobilenet_imagenet_accuracy.py model_file_path")
  sys.exit()

model_file_path = sys.argv[1]

session = onnxruntime.InferenceSession(model_file_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

top1_success_filenames = []
top5_success_filenames = []

cnt = 0

data = imagenet.Data()

for filename, image, correct_class in data:

  image = crop_max_square(image)
  image = image.resize((224, 224))
  image = image.convert("RGB")
  image = np.array(image).transpose(2, 0, 1)
  X = preprocess(image)
  
  ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X)
  raw_result = session.run([output_name], {input_name: ortvalue})

  res = postprocess(raw_result)
  res = np.array(res)
  sort_idx = np.argsort(res)[::-1]
  
  top5_idx = sort_idx[:5]
  print(f"{cnt} {filename} { *top5_idx, *res[top5_idx], }")

  for i in range(5):
    idx = sort_idx[i]
    class_label = data.class_label_map[str(idx)]
    if class_label[0] == correct_class:
      if i == 0:
        top1_success_filenames.append(filename)
      else:
        top5_success_filenames.append(filename)
      break

  cnt += 1
  
  if cnt > 1000:
    break
  
total_cnt = cnt
top1_cnt = len(top1_success_filenames)
top5_cnt = len(top5_success_filenames) + top1_cnt
  
print(f"Top1 Accuracy : {top1_cnt * 100 / total_cnt} %")
print(f"Top5 Accuracy : {top5_cnt * 100 / total_cnt} %")

