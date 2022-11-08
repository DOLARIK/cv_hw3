import math
import numpy as np
from gauss import GaussFilter
from utils import _filter, convolve
import cv2

class Hessian:
	"""docstring for Hessian"""
	def __init__(self, smoothing_std = 1):
		super(Hessian, self).__init__()
		sobel = np.asarray([[-1, 0, 1],
							[-2, 0, 2], 
							[-1, 0, 1]])
		self.sobel_x = sobel
		self.sobel_y = sobel.T

		self.smoothing_std = smoothing_std


	def __call__(self, image, threshold = 2):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_extend = np.dstack([gray for i in range(3)])
		Ix, Iy = self._get_first_order_derivatives(gray_extend)
		Ixx, Ixy, Iyx, Iyy = self._get_second_order_derivative(Ix, Iy)

		hessian_det = Ixx*Iyy - Ixy*Iyx

		hessian_norm = 255*hessian_det/hessian_det.max()

		hessian = hessian_norm*np.asarray(hessian_norm > threshold, dtype='float')

		return hessian

	def _get_first_order_derivatives(self, image):
		image = convolve(image, GaussFilter(self.smoothing_std).kernel)
		Ix = _filter(image, self.sobel_x)
		Iy = _filter(image, self.sobel_y)
		return Ix, Iy

	def _get_second_order_derivative(self, Ix, Iy):
		Ixx = _filter(Ix, self.sobel_x)
		Ixy = _filter(Ix, self.sobel_y)
		Iyx = _filter(Iy, self.sobel_x)
		Iyy = _filter(Iy, self.sobel_y)
		return Ixx, Ixy, Iyx, Iyy