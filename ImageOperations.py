# List of function operations to enhance training_dataset
# Author Stefan Mocko
# Licensed

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import pi
from skimage.color import rgb2gray


def pipe_convert_normalize(image):
    gray_image = rgb2gray(image)
    normalized = cv2.normalize(gray_image.astype('float'),
                               None, 0.0, 256.0, cv2.NORM_MINMAX)
    sqrt_image = np.sqrt(normalized)
    return sqrt_image

def find_centre(image):
    x = np.size(image, 1) / 2
    y = np.size(image, 0) / 2
    return x, y

def fft_image(image):
    img_fft = np.fft.fft2(image)
    img_shift = np.fft.fftshift(img_fft)
    return img_shift

def get_meshxy(image):
    meshx, meshy = np.meshgrid(np.arange(0, np.size(image, 0)),
                               np.arange(0, np.size(image, 1)))
    return meshx, meshy

def compute_mesh_axis(mesh_value, centre_axis_value):
    temp_mesh = mesh_value - centre_axis_value
    return temp_mesh.astype(float)

def aberate_image(image):
    temp_aber_image = np.fft.ifft2(image)
    abs_abber_image = abs(temp_aber_image) ** 2
    return cv2.normalize(abs_abber_image.astype('float'), None,
                         0.0, 256.0, cv2.NORM_MINMAX)

def create_aberation_data(image):
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
    out_image = pipe_convert_normalize(image)
    # create_for_compare(image_check, 'astig_processed.png')
    centre1, centre2 = find_centre(out_image)
    shifted_image = fft_image(out_image)

    # coordinate system
    meshx, meshy = get_meshxy(image)
    meshx_f = compute_mesh_axis(meshx, centre1)
    meshy_f = compute_mesh_axis(meshy, centre2)

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
    return aberate_image(multiply_image)



