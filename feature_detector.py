from nms import non_max_suppression
from utils import extract_keypoints

class FeatureDetector(object):
	"""docstring for FeatureDetector"""
	def __init__(self, corner_detector):
		super(FeatureDetector, self).__init__()
		self.corner_detector = corner_detector
	
	def __call__(self, image, 
				 nms_bool=True,
				 kp_thresh=(30,30,30),
				 **corner_detector_kwargs,
				 ):
		corner_image = self.corner_detector(image, **corner_detector_kwargs)
		nms = non_max_suppression(corner_image, bool=nms_bool)
		keypoints = extract_keypoints(nms*255, thresh=kp_thresh)
		return keypoints



