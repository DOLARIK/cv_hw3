import numpy as np

class Panorama:
    """docstring for Panorama

    Arguments
        left: left image (numpy array)
        right: right image (numpy array)
        affine: affine transformation applied to left coordinates (numpy array)
        transform: image that needs to undergo transformation (str)
    """
    def __init__(self, left, right, affine,
                       background_dims = (900,2000,3)):
        super(Panorama, self).__init__()
        self.left = left
        self.right = right
        self.affine = affine

        self.background_dims = background_dims

    def _get_first_and_second_image(self, transform):
        if transform == 'left':
            return self.right, self.left
        elif transform == 'right':
            return self.left, self.right

    def _get_adjusted_affine(self, transform):
        if transform == 'left':
            return self.affine
        elif transform == 'right':
            return np.linalg.inv(self.affine)

    def stitch(self, transform, edge_blend = False):
        background = np.zeros(self.background_dims)

        first, second = self._get_first_and_second_image(transform)
        affine = self._get_adjusted_affine(transform)         

        del_first_x = background.shape[1]//2 - first.shape[1]//2
        del_first_y = background.shape[0]//2 - first.shape[0]//2

        background[del_first_y : del_first_y + first.shape[0], 
             del_first_x : del_first_x + first.shape[1], :] = first

        del_second_x = background.shape[1]//2 - second.shape[1]//2
        del_second_y = background.shape[0]//2 - second.shape[0]//2


        for x in range(second.shape[1]):
            for y in range(second.shape[0]):
                proj = np.dot(affine, np.asarray([[x],[y],[1]]))
                x_new = proj[0,0]/proj[2,0]
                y_new = proj[1,0]/proj[2,0]
                        
                x_proj = x_new + del_second_x
                y_proj = y_new + del_second_y
                
 
                if background[int(y_proj), int(x_proj), 0] > 0:


                    if edge_blend:
                        a = ((x/second.shape[1])**2 + (y/second.shape[0])**2)**0.5
                        b = (((first.shape[1] - (int(x_proj) - del_first_x))/first.shape[1])**2 + ((first.shape[0] - (int(y_proj) - del_first_y))/first.shape[0])**2)**.5
                    else:
                        a = 1
                        b = 1

                    if transform == 'left':
                        background[int(y_proj), int(x_proj), :] = (b*second[y, x, :] + a*background[int(y_proj), int(x_proj), :])/(a+b)

                    elif transform == 'right':
                        background[int(y_proj), int(x_proj), :] = (a*second[y, x, :] + b*background[int(y_proj), int(x_proj), :])/(a+b)
                else:
                    background[int(y_proj), int(x_proj), :] = second[y, x, :]

        y, x, z = np.where(background>0)

        Ys = []
        Xs = []

        for b, a, c in zip(y.tolist(), x.tolist(), z.tolist()):
            Ys.append(b)
            Xs.append(a)    
            
        y_min = min(Ys)
        x_min = min(Xs)
        y_max = max(Ys)
        x_max = max(Xs)

        panorama = np.asarray(255*background[y_min:y_max+1, x_min:x_max+1, :]/background[y_min:y_max+1, x_min:x_max+1, :].max(),
                              dtype='uint8')

        return panorama


        
