
import numpy
import onnxruntime
import os
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import QuantFormat, QuantType, CalibrationMethod, quantize_static
import mnist
import numpy as np

class MnistDataReader(CalibrationDataReader):
  def __init__(self, input_name: str):
    self.enum_data = None

    self.input_name = input_name
    self.data = mnist.Data()
    self.data.load()
    self.data.test_set_images = self.data.test_set_images.astype(np.float32)
    
  def get_next(self):
    if self.enum_data is None:
      self.enum_data = iter(
        [{self.input_name: data} for data in self.data.test_set_images]
      )
    return next(self.enum_data, None)

  def rewind(self):
    self.enum_data = None

input_model_path = "./models/mnist-12-infer.onnx"
output_model_path = "./models/mnist-12_quantized.onnx"

session = onnxruntime.InferenceSession(input_model_path, None)
input_name = session.get_inputs()[0].name

dr = MnistDataReader(input_name)

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

