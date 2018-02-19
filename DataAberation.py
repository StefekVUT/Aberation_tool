# DataAberation.py
# Author: Stefan Mocko <https://github.com/>
# Licensed Apache License Version 2.0

import os
import glob
import numpy as np
import cv2
import ImageOperations as ImOp
import matplotlib.pyplot as plt
import sys
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

    def load_image_and_label(self, mid_name):
        actual_image = Image.open(self.data_path + "/" + mid_name)
        actual_label = Image.open(self.label_path + "/" + mid_name)

        # actual_image = cv2.imread(self.data_path + "/" + mid_name, 0)
        # actual_label = cv2.imread(self.label_path + "/" + mid_name, 0)
        return [actual_image, actual_label]

    def write_image_and_label(self, processed_data, mid_name, oper):
        operation_list = '-'.join(oper)

        image = processed_data[0]
        label = processed_data[1]
        image.save(self.output_path + "/" + operation_list + mid_name)
        label.save(self.output_label_path + "/" + operation_list + mid_name)

        #cv2.imwrite(self.output_path + "/" + "aberated_" + mid_name, processed_data[0])
        #cv2.imwrite(self.output_label_path + "/" + "aberated_" + mid_name, processed_data[1])

    def process_data(self):
        i = 0
        for img_name in self.image_paths:
            mid_name = img_name[img_name.rindex("\\") + 1:]
            operations = []

            image_data = self.load_image_and_label(mid_name)
            # perform operations - call functions
            data, oper = ImOp.RotateRange(25, 25).perform_operation(image_data, operations)

            # temp_processed_data = ImageOperations.create_aberation_data(actual_image)

            # write output
            self.write_image_and_label(data, mid_name, oper)
            i = i+1
            sys.stdout.write('\r[{0}{1}] {2}'.format('#' * (i / 10), ' ' * (10 - i / 10), i))

    def process_one(self):
        img_name = self.image_paths[0]
        mid_name = img_name[img_name.rindex("\\")+1:]
        operations = []

        image_data = self.load_image_and_label(mid_name)
        # perform operations - call functions
        data, oper = ImOp.RotateRange(25, 25).perform_operation(image_data, operations)

        # temp_processed_data = ImageOperations.create_aberation_data(actual_image)

        # write output
        self.write_image_and_label(data, mid_name, oper)


if __name__ == "__main__":
    aberration_data = AberrationProcess()
    aberration_data.find_data_paths()
    aberration_data.process_data()
    #aberration_data.process_one()

    print("\n Aberration Done")
