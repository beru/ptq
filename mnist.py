import ctypes
import numpy as np

class TestSetLabelFileHeader(ctypes.BigEndianStructure):
  _fields_ = [
    ("magic_number", ctypes.c_uint32),
    ("num_items", ctypes.c_uint32),
  ]

class TestSetImageFileHeader(ctypes.BigEndianStructure):
  _fields_ = [
    ("magic_number", ctypes.c_uint32),
    ("num_images", ctypes.c_uint32),
    ("num_rows", ctypes.c_uint32),
    ("num_cols", ctypes.c_uint32),
  ]

test_set_image_file_path = "./mnist/t10k-images.idx3-ubyte"
test_set_label_file_path = "./mnist/t10k-labels.idx1-ubyte"

class Data:

  def __init__(self):
    self.test_set_label_file_header = TestSetLabelFileHeader()
    self.test_set_image_file_header = TestSetImageFileHeader()
    
  def load(self):
    with open(test_set_label_file_path, "rb") as f:
      f.readinto(self.test_set_label_file_header)
      if self.test_set_label_file_header.magic_number != 0x00000801:
        raise ValueError(f"magic number of set image file must be 0x00000801 but {self.test_set_label_file_header.magic_number} is written.")

      self.test_set_labels = np.fromfile(f, dtype=np.ubyte, count=self.test_set_label_file_header.num_items)

    with open(test_set_image_file_path, "rb") as f:
      f.readinto(self.test_set_image_file_header)
      if self.test_set_image_file_header.magic_number != 0x00000803:
        raise ValueError(f"magic number of set image file must be 0x00000803 but {self.test_set_image_file_header.magic_number} is written.")

      hdr = self.test_set_image_file_header
      count = hdr.num_images * hdr.num_rows * hdr.num_cols
      self.test_set_images = np.fromfile(f, dtype=np.ubyte, count=count)
      self.test_set_images = self.test_set_images.reshape((hdr.num_images, 1, 1, hdr.num_rows, hdr.num_cols))

