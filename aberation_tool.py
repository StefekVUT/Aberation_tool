

# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import pi
from skimage.color import rgb2gray


class AberationProcess(object):

    def __init__(self, data_path="./data/trainme", output_path="./data/trainmeaberated", label_path="./data/labelme",
                 output_label_path="./data/labelmeout", img_type="png"):

        self.output_label_path = output_label_path
        self.label_path = label_path
        self.data_path = data_path
        self.output_path = output_path
        self.img_type = img_type

    @staticmethod
    def pipe_convert_normalize(image):
        gray_image = rgb2gray(image)
        normalized = cv2.normalize(gray_image.astype('float'),
                                   None, 0.0, 256.0, cv2.NORM_MINMAX)
        sqrt_image = np.sqrt(normalized)
        return sqrt_image

    @staticmethod
    def create_for_compare(image, filename):
        normalized = cv2.normalize(image.astype('float'), None,
                                   0.0, 256.0, cv2.NORM_MINMAX)
        cv2.imwrite(filename, normalized)

    @staticmethod
    def find_centre(image):
        x = np.size(image, 1) / 2
        y = np.size(image, 0) / 2
        return x, y

    @staticmethod
    def fft_image(image):
        img_fft = np.fft.fft2(image)
        img_shift = np.fft.fftshift(img_fft)
        return img_shift

    @staticmethod
    def get_meshxy(image):
        meshx, meshy = np.meshgrid(np.arange(0, np.size(image, 0)),
                                   np.arange(0, np.size(image, 1)))
        return meshx, meshy

    @staticmethod
    def compute_mesh_axis(mesh_value, centre_axis_value):
        temp_mesh = mesh_value - centre_axis_value
        return temp_mesh.astype(float)

    @staticmethod
    def aberate_image(image):
        temp_aber_image = np.fft.ifft2(image)
        abs_abber_image = abs(temp_aber_image) ** 2
        return cv2.normalize(abs_abber_image.astype('float'), None,
                             0.0, 256.0, cv2.NORM_MINMAX)

    def create_aberation_data(self):
        print('-' * 30)
        print('Creating aberated images...')
        print('-' * 30)
        imgs = glob.glob(self.data_path + "/*." + self.img_type)
        print(str(len(imgs))+" image(s) inside directory")

        for imgname in imgs:
            midname = imgname[imgname.rindex("\\")+1:]
            img = cv2.imread(self.data_path + "/" + midname, 0)

            """
            Parameters
            ----------
            E0          eV | electron rest energy
            E           eV | electron kinetic energy
            deltaE      eV | for FEG
            llambda     nm | e wavelength for 200 keV
            px_size     nm | pixel size for magnification 500kx
            C30         nm | Spherical Aberration coefficient
            Cc          nm | Chromatic Aberration coefficient
            deltaf0     nm | overfocus = + | underfocus = - | base value = 0
            C22         nm | Astigmatic Aberration coeff. | base value = -200
            Fi_astig    rad| base value = 0
            fr             | coeff. from REIMER []
            H_Rei       nm | Reimer
            H_Vul       nm | Vulovic
            """
            param = {'E0': 0.5109989461 * 10 ** 6,
                     'E': 200 * 10 ** 3,
                     'deltaE': 0.5,
                     'llambda': 2.51 * 10 ** (-3),
                     'px_size': 17.1 * 10 ** (-3),
                     'C30': 1.2 * 10 ** 6,
                     'Cc': 1.2 * 10 ** 6,
                     'deltaf0': 0,
                     'C22': -200,
                     'Fi_astig': (pi / 180) * 0}

            param['fr'] = (1 + param['E'] / param['E0']) / \
                          (1 + param['E'] / param['E0'] / 2)

            param['H_Rei'] = param['Cc'] * (param['deltaE'] / param['E']) * param['fr']

            param['H_Vul'] = param['Cc'] * (param['deltaE'] / param['E'])

            # code
            out_image, image_check = self.pipe_convert_normalize(img)
            # create_for_compare(image_check, 'astig_processed.png')
            centre1, centre2 = self.find_centre(out_image)
            shifted_image = self.fft_image(out_image)

            # coordinate system
            meshx, meshy = self.get_meshxy(image_check)
            meshx_f = self.compute_mesh_axis(meshx, centre1)
            meshy_f = self.compute_mesh_axis(meshy, centre2)

            # parameters
            param['Phi'] = np.arctan(meshy_f / (meshx_f + 1 * 10 ** -10))
            param['q'] = param['px_size'] * np.sqrt(np.abs(meshx_f) ** 2 + np.abs(meshy_f) ** 2)
            param['deltaf'] = param['deltaf0'] + param['C22'] + np.cos(2 * (param['Phi'] - param['Fi_astig']))
            param['W'] = 0.5 * param['C30'] * pi * (param['llambda'] ** 3) * \
                         param['q'] ** 4 + pi * param['llambda'] * np.multiply(param['deltaf'], param['q'] ** 2)
            param['CTF'] = np.cos(2 * pi * param['llambda'] * param['W']) + np.sin(2 * pi * param['llambda'] *
                                                                                   param['W']) * complex(0, 1)
            param['Kc'] = np.exp(
                -((pi * param['llambda'] * param['q'] ** 2 * param['H_Rei']) ** 2 / (16 * np.log(2)))) + 0j

            # aberate image
            multiply_image = np.multiply(shifted_image, np.multiply(param['CTF'], param['Kc']))
            aberated_image = self.aberate_image(multiply_image)
            cv2.imwrite(self.output_path + "/" + "aberated_" + midname, aberated_image)

if __name__ == "__main__":
    aberdata = AberationProcess()
    aberdata.create_aberation_data()
    print("Done")
