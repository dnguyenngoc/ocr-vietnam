"""@Author by Duy Nguyen Ngoc - email: duynguyenngoc@hotmail.com/duynn_1@digi-texx.vn"""


"""@Author by Duy Nguyen Ngoc - email: duynguyenngoc@hotmail.com/duynn_1@digi-texx.vn"""


import cv2
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from detector import Detector
from detector import DetectorTF2
from recognition import TextRecognition

from helpers import corner_utils
from helpers import ocr_helpers
from helpers.image_utils import align_image, sort_text
from helpers import load_label_map


class CompletedModel(object):
    def __init__(self):
        self.text_recognition_model = TextRecognition(path_to_checkpoint='./models/text_recogintion/transformerocr.pth')
    
    def predict(self, image):
        result = self.text_recognition_model.predict(image)
        return result
