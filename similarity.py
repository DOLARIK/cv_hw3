import numpy as np
import cv2

def ssd(patch1, patch2):
    patch1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
    patch2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)
    
    ssd = np.sum(np.square(patch1 - patch2))
    return ssd

def norm_data(data):
    mean_data=np.mean(data)
    std_data=np.std(data)
    return (data-mean_data)/(std_data)

def ncc(data_0, data_1):
    data0 = cv2.cvtColor(data_0, cv2.COLOR_BGR2GRAY)
    data1 = cv2.cvtColor(data_1, cv2.COLOR_BGR2GRAY)
    return (1/(len(data0)*len(data0[0]))) * np.sum(norm_data(data0)*norm_data(data1))

