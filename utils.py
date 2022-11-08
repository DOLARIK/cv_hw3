import cv2
import matplotlib.pyplot as plt
import imageio
import math
import numpy as np
from hashlib import sha1
import json
from matplotlib.patches import ConnectionPatch

from numpy import all, array, uint8


def load_image(image_path):
    img = cv2.imread(image_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def visualize_image(image):
    """
    image: an output of load_png function
    """
    plt.imshow(image)
    
def save_array_as_image(array, name):
    imageio.imwrite(name, array)

def _filter(image, kernel, 
             stride=(1, 1),
             padding=None,):
    
    num_channels = image.shape[2]
    kernel_extended = np.asarray([kernel for i in range(num_channels)]).reshape(kernel.shape[0], 
                                                                                        kernel.shape[1], 
                                                                                        num_channels)
    
    image_height, image_width, image_channels = image.shape
    kernel_height, kernel_width, kernel_channels = kernel_extended.shape
    
    stride_height, stride_width = stride
       
    if padding is None:
        width_padding = int(((image_width - 1)*stride_width + kernel_width - image_width)/2)
        height_padding = int(((image_height - 1)*stride_height + kernel_height - image_height)/2)
    else:
        height_padding, width_padding = padding
        
    
    input_image = np.zeros((height_padding + image_height + height_padding, width_padding + image_width + width_padding, image_channels))
    input_image[height_padding:image_height + height_padding, width_padding:image_width + width_padding, :] = image.copy()  
    
    input_height, input_width, input_channels = input_image.shape
    
    output_height = int((input_height - kernel_height)/stride_height) + 1
    output_width = int((input_width - kernel_width)/stride_width) + 1
    
    output_image = np.zeros((output_height, output_width, input_channels))
    
    for x in range(0, output_width):
        for y in range(0, output_height):
            output_image[y, x, :] = np.einsum('ij,ijk->k', 
                                              kernel, 
                                              input_image[y*stride_height:y*stride_height + kernel_height, 
                                                          x*stride_width:x*stride_width + kernel_width, :])
    
    return output_image

def convolve(image, kernel, 
             stride=(1, 1),
             padding=None,):

    kernel_flipped = np.rot90(np.flipud(kernel.T))
    
    return _filter(image, kernel_flipped, stride, padding)

class hashable(object):
    r'''Hashable wrapper for ndarray objects.
        Instances of ndarray are not hashable, meaning they cannot be added to
        sets, nor used as keys in dictionaries. This is by design - ndarray
        objects are mutable, and therefore cannot reliably implement the
        __hash__() method.
        The hashable class allows a way around this limitation. It implements
        the required methods for hashable objects in terms of an encapsulated
        ndarray object. This can be either a copied instance (which is safer)
        or the original object (which requires the user to be careful enough
        not to modify it).
    '''
    def __init__(self, wrapped, tight=False):
        r'''Creates a new hashable object encapsulating an ndarray.
            wrapped
                The wrapped ndarray.
            tight
                Optional. If True, a copy of the input ndaray is created.
                Defaults to False.
        '''
        self.__tight = tight
        self.__wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(self, other):
        return all(self.__wrapped == other.__wrapped)

    def __hash__(self):
        return self.__hash

    def unwrap(self):
        r'''Returns the encapsulated ndarray.
            If the wrapper is "tight", a copy of the encapsulated ndarray is
            returned. Otherwise, the encapsulated ndarray itself is returned.
        '''
        if self.__tight:
            return array(self.__wrapped)

        return self.__wrapped

def extract_keypoints(corner_image, thresh=(30,30,30)):
    indices = np.where(np.all(corner_image > thresh, axis=-1))
    return list(zip(indices[1], indices[0]))

def visualize_keypoints(image, keypoints, set_inches = (18.5, 10.5)):
    figure, ax = plt.subplots(1, 1)
    ax.scatter(np.asarray(keypoints)[:, 0], 
               np.asarray(keypoints)[:, 1], 
               s=50, 
               c='r', 
               marker = 'x')
    ax.imshow(image)
    figure.set_size_inches(set_inches[0], set_inches[1])

def save_list(data, name):
    with open(name, 'w') as f:
        json.dump(data, f, default='str')

def save_keypoints(keypoints, name):
    save_list(np.asarray(keypoints).tolist(), name)

def load_keypoints(name):
    return [tuple(kp) for kp in json.load(open(name))]


def draw_matches(left, right, pairs, set_inches = (18.5, 10.5)):
    
    figure, ax = plt.subplots(2, 1)
    
    ax[0].imshow(left)
    
    ax[1].imshow(right)
    for pair in pairs:
        ax[0].scatter(pair[0][0], 
               pair[0][1], 
               s=50, 
               c='r', 
               marker = 'x')
        ax[1].scatter(pair[1][0], 
               pair[1][1], 
               s=50, 
               c='r', 
               marker = 'x')
        con = ConnectionPatch(xyA=pair[0], xyB=pair[1], coordsA="data", coordsB="data",
                      axesA=ax[0], axesB=ax[1], color="red")
        ax[1].add_artist(con)
    
    
    figure.set_size_inches(set_inches[0], set_inches[1])

def save_pairs(pairs, name):
    pairs_json = []
    for pair in pairs:
        element = [[int(pair[0][0]), int(pair[0][1])], [int(pair[1][0]), int(pair[1][1])], float(pairs[pair])]
        pairs_json.append(element)
    
    # print(pairs_json)

    with open(name, 'w') as f:
        json.dump(pairs_json, f)

def load_pairs(name):
    pairs_json = json.load(open(name))

    pairs = {}
    for element in pairs_json:
        lkp = element[0]
        rkp = element[1]
        score = element[2]
        pairs[(tuple(lkp), tuple(rkp))] = score

    return pairs

def get_top_pairs(pairs, top_n=50):
    top_pairs = {k:pairs[k] for k in list(pairs.keys())[:top_n]}
    return top_pairs
