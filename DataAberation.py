# DataAberation.py
# Author: Stefan Mocko <https://github.com/>
# Licensed

import os
import glob
import numpy as np
import cv2
import ImageOperations
import matplotlib.pyplot as plt
from math import pi
from skimage.color import rgb2gray
from PIL import Image, ImageOps


class AberrationProcess(object):
    def __init__(self, data_path="./data/trainme", output_path="./data/trainmeaberated", label_path="./data/labelme",
                 output_label_path="./data/labelaberated", img_type="png"):

        self.output_label_path = output_label_path
        self.label_path = label_path
        self.data_path = data_path
        self.output_path = output_path
        self.img_type = img_type
        self.image_paths = ""
        self.label_paths = ""

    def find_data_paths(self):
        print('-' * 30)
        print('Looking for data paths...')
        print('-' * 30)

        self.image_paths = glob.glob(self.data_path + "/*." + self.img_type)
        self.label_paths = glob.glob(self.label_path + "/*." + self.img_type)

    def process_data(self):
        for img_name in self.image_paths:
            mid_name = img_name[img_name.rindex("\\")+1:]
            actual_image = cv2.imread(self.data_path + "/" + mid_name, 0)
            actual_label = cv2.imread(self.label_path + "/" + mid_name, 0)

            # perform operations - call functions

            # write output
            cv2.imwrite(self.output_path + "/" + "aberated_" + mid_name, actual_image)
            cv2.imwrite(self.output_label_path + "/" + "aberated_" + mid_name, actual_label)

    def process_one(self):
        img_name = self.image_paths[0]
        mid_name = img_name[img_name.rindex("\\")+1:]

        actual_image = cv2.imread(self.data_path + "/" + mid_name, 0)
        actual_label = cv2.imread(self.label_path + "/" + mid_name, 0)
        # perform operations - call functions

        p1image = ImageOperations.create_aberation_data(actual_image)

        # write output
        cv2.imwrite(self.output_path + "/" + "aberated_" + mid_name, p1image)
        cv2.imwrite(self.output_label_path + "/" + "aberated_" + mid_name, actual_label)


if __name__ == "__main__":
    aberration_data = AberrationProcess()
    aberration_data.find_data_paths()
    # aberration_data.process_data()
    aberration_data.process_one()

    print("Done")
