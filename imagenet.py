
import os
import json
from PIL import Image
import xml.etree.ElementTree as ET

images_folder = "./imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
annotation_folder = "./imagenet-object-localization-challenge/ILSVRC/Annotations/CLS-LOC/val"

class Data:
  def __init__(self):
    self._filenames = os.listdir(images_folder)
    self._i = 0
    
    with open("imagenet/imagenet_class_index.json", "r") as f:
      self.class_label_map = json.load(f)
    
  def __iter__(self):
    return self
    
  def __next__(self):
    
    if self._i >= (len(self._filenames) // 100):
      raise StopIteration
    
    filename = self._filenames[self._i]
    filename_wo_ext, ext = os.path.splitext(filename)
    image_path = os.path.join(images_folder, filename)
    image = Image.open(image_path)
    
    xml_path = os.path.join(annotation_folder, filename_wo_ext + ".xml")
    correct_class = ET.parse(xml_path).getroot().find(".//object/name").text
    
    self._i += 1
    return (filename, image, correct_class)

