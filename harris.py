import math
import numpy as np
from gauss import GaussFilter
from utils import _filter, convolve, hashable
import cv2
import functools

class Harris:
	"""docstring for Harris"""
	def __init__(self,
				 threshold = 0,
				 num_top = -1,
				 corner_gauss_dim = None, 
				 corner_gauss_std = 1,
				 response_alpha =  0.05,
				 smoothing_std = 1):
		super(Harris, self).__init__()
		sobel = np.asarray([[-1, 0, 1],
							[-2, 0, 2], 
							[-1, 0, 1]])
		self.sobel_x = sobel
		self.sobel_y = sobel.T

		self.smoothing_std = smoothing_std
		self.corner_gauss_dim = corner_gauss_dim
		self.corner_gauss_std = corner_gauss_std

		self.response_alpha = response_alpha
		self.threshold = threshold
		self.num_top = 3*num_top # multiplying for all 3 colour channels

	def __call__(self, image, 
				 threshold = 0,
				 num_top = -1,):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_extend = np.dstack([gray for i in range(3)])

		self.threshold = threshold
		self.num_top = 3*num_top # multiplying for all 3 colour channels

		M = self._second_moment_matrix(hashable(gray_extend))
		R = self._response_function(M)

		R = self._select_top_pixels(R, self.num_top)

		R_normalized = R*np.asarray(R/R.max() > self.threshold, 
								    dtype='uint8')/R.max()

		return R_normalized

	def _select_top_pixels(self, matrix, num_pixels):
		idxs = np.argsort(matrix.ravel())[-(num_pixels):]
		h,w,c = np.unravel_index(idxs, (matrix.shape[0], matrix.shape[1], 3))
		top_n = matrix*0
		for x, y, z in zip(list(h), list(w), list(c)):
		    top_n[x, y, z] = matrix[x, y, z]

		return top_n

	def _get_first_order_derivatives(self, image):
		Ix = _filter(image, self.sobel_x)
		Iy = _filter(image, self.sobel_y)
		return Ix, Iy

	# The main bottleneck here is while computing
	# the second moment matrices of the images.
	# Hence, chaching this would speed up experimentation.
	# 'functools' is used to cache the second moment matrices
	# of the previously fed images.
	@functools.lru_cache(maxsize=32)
	def _second_moment_matrix(self, image_hash):

		image = image_hash.unwrap()

		smooth = convolve(image, GaussFilter(self.smoothing_std).kernel)
		Ix, Iy = self._get_first_order_derivatives(smooth)


		corner_gauss = GaussFilter(self.corner_gauss_std,
								   self.corner_gauss_dim).kernel
		
		M = [[convolve(Ix*Ix, corner_gauss), convolve(Ix*Iy, corner_gauss)],
			 [convolve(Iy*Ix, corner_gauss), convolve(Iy*Iy, corner_gauss)]]

		return M

	def _response_function(self, M):
		M_determinant = M[0][0]*M[1][1] - M[0][1]*M[1][0]
		M_trace = M[0][0] + M[1][1]

		R = M_determinant - self.response_alpha*(M_trace**2)

		return R


		