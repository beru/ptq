
import numpy
import onnxruntime
import os
import sys
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationMethod, quantize_static
import imagenet
import numpy as np

print(onnxruntime.__version__)

def crop_center(pil_img, crop_width, crop_height):
  img_width, img_height = pil_img.size
  return pil_img.crop(((img_width - crop_width) // 2,
                       (img_height - crop_height) // 2,
                       (img_width + crop_width) // 2,
                       (img_height + crop_height) // 2))

def crop_max_square(pil_img):
  return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def preprocess(image):
  image = crop_max_square(image)
  image = image.resize((224, 224))
  image = image.convert("RGB")

  image = np.array(image).astype('float32')
  image -= [127.0, 127.0, 127.0]
  image /= [128.0, 128.0, 128.0]
  image = np.expand_dims(image, axis=0)
  return image

class ImageNetDataReader(CalibrationDataReader):
  def __init__(self, input_name: str):
    self.input_name = input_name
    self.data = imagenet.Data()
    
  def get_next(self):
    try:
      ret = next(self.data)
      filename, image, correct_class = *ret,
      image = preprocess(image)
      return {self.input_name: image}
    except:
      return

# python -m onnxruntime.quantization.preprocess --input models\efficientnet-lite4-11.onnx --output models\efficientnet-lite4-11-infer.onnx
input_model_path = "./models/efficientnet-lite4-11-infer.onnx"
output_model_path = "./models/efficientnet-lite4-11_quantized.onnx"

session = onnxruntime.InferenceSession(input_model_path, None)
input_name = session.get_inputs()[0].name

dr = ImageNetDataReader(input_name)

quantize_static(
  input_model_path,
  output_model_path,
  dr,
  quant_format=QuantFormat.QOperator,
  per_channel=True,
  activation_type=QuantType.QUInt8,
  weight_type=QuantType.QInt8,
  optimize_model=True,
  calibrate_method=CalibrationMethod.Entropy
)

