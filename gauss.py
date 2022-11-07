import math
import numpy as np

class GaussFilter:
    def __init__(self, 
                 std_dev, 
                 kernel_dimension=None):
        self.std_dev = std_dev
        if not kernel_dimension:
            self.kernel_dimension = self.compute_kernel_dimensions(std_dev)
        else:
            self.kernel_dimension = kernel_dimension 
        self.kernel = self.generate_gaussian_kernel(std_dev)
    
    @staticmethod
    def compute_kernel_dimensions(std_dev):
        return int(1 + 2*(3*std_dev))

    def get_gaussian_coordiantes(self, x, y):
        center = int(self.kernel_dimension/2) + 1
        origin = center - 1
        gauss_x = x - origin
        gauss_y = y - origin
        
        return gauss_x, gauss_y
    
    def compute_gaussian_pixel_value(self, gauss_x, gauss_y):
        std_squared = self.std_dev**2
        pi = math.pi
        pixel_value = math.exp(-(gauss_x**2 + gauss_y**2)/(2*std_squared))/(2*pi*std_squared)
        return pixel_value

    def generate_gaussian_kernel(self, std_dev):
        """
        std_dev: value of standard deviation in terms of pixels
        """
        k = self.kernel_dimension
        kernel = np.zeros((k, k))
        
        for x, y in np.ndindex((k, k)):
            gauss_x, gauss_y = self.get_gaussian_coordiantes(x, y)
            kernel[x, y] = self.compute_gaussian_pixel_value(gauss_x, gauss_y)
            
        return kernel/sum(sum(kernel))