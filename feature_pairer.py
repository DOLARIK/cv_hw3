from similarity import ssd, ncc
import numpy as np

class FeaturePairer:
    """docstring for FeaturePairer"""
    def __init__(self, window_size=3, similarity='ncc'):
        super(FeaturePairer, self).__init__()
        self.similarity = similarity
        if similarity == 'ncc':
            self.similarity_scorer = ncc
        elif similarity == 'ssd':
            self.similarity_scorer = ssd

        self.window_size = window_size


    def __call__(self, left, left_keypoints,
                       right, right_keypoints):

        pairs = {}

        for lkp in left_keypoints:
            for rkp in right_keypoints:
                left_patch = self._capture_patch(left, lkp, self.window_size)
                right_patch = self._capture_patch(right, rkp, self.window_size)
                pairs[(lkp,rkp)] = self._compare_patches(left_patch, 
                                                         right_patch)
        
        if self.similarity == 'ncc':
            sorted_pairs = {k: v for k, v in sorted(pairs.items(), 
                                                    key=lambda item: item[1],
                                                    reverse=True)}
        elif self.similarity == 'ssd':
            sorted_pairs = {k: v for k, v in sorted(pairs.items(), 
                                                    key=lambda item: item[1],
                                                    reverse=False)}

        return sorted_pairs    
    
    def _capture_patch(self, image, keypoint, window_size=3):
        top_left, bottom_right = self._get_extreme_coords(image, 
                                                          keypoint, 
                                                          window_size)

        patch = image[top_left[0]: bottom_right[0], 
                      top_left[1]: bottom_right[1], :]

        return patch
    
    def _get_extreme_coords(self, image, keypoint, window_size=3):
        x, y = keypoint
        
        x_max = image.shape[1] - 1
        y_max = image.shape[0] - 1
        
        if max(0, y - (window_size//2)) == 0:
            y_top = 0
            y_bottom = y_top + window_size
        elif min(y + window_size, y_max) == y_max:
            y_top = y_max - window_size + 1
            y_bottom = y_max + 1
        else:
            y_top = y - (window_size//2)
            y_bottom = y + (window_size//2) + 1
        
        if max(0, x - (window_size//2)) == 0:
            x_left = 0
            x_right = x_left + window_size
        elif min(x + window_size, x_max) == x_max:
            x_left = x_max - window_size + 1
            x_right = x_max + 1
        else:
            x_left = x - (window_size//2)
            x_right = x + (window_size//2) + 1
            
        top_left = (y_top, x_left)
        bottom_right = (y_bottom, x_right)
        
        return top_left, bottom_right
        

    def _compare_patches(self, left_patch, right_patch):
        if self.similarity=='ncc':
            return self.similarity_scorer(left_patch, right_patch)
        elif self.similarity=='ssd':  
            return self.similarity_scorer(left_patch, right_patch)


