import numpy as np

def get_comparison_filters():
    top = np.asarray([[0,-1,0],[0,1,0],[0,0,0]])
    top_left = np.asarray([[-1,0,0],[0,1,0],[0,0,0]])
    top_right = np.asarray([[0,0,-1],[0,1,0],[0,0,0]])
    left = np.asarray([[0,0,0],[-1,1,0],[0,0,0]])
    right = np.asarray([[0,0,0],[0,1,-1],[0,0,0]])
    bottom = np.asarray([[0,0,0],[0,1,0],[0,-1,0]])
    bottom_left = np.asarray([[0,0,0],[0,1,0],[-1,0,0]])
    bottom_right = np.asarray([[0,0,0],[0,1,0],[0,0,-1]])
    
    return top_left, top, top_right, left, right, bottom_left, bottom, bottom_right



def non_max_suppression(gradient, bool=False):
    height, width, channels = gradient.shape
    
    nms_gradient = np.zeros((height, width, channels))

    for x in range(1, width-1):
        for y in range(1, height-1):
            filters = get_comparison_filters()
            values = [np.einsum('ij,ijk->k', _filter_, gradient[y-1:y+1+1, x-1:x+1+1, :]) for _filter_ in filters]            
            
            value = values[0]
            for i in range(1, len(values)):
                value = (values[i] > 0) & (value > 0)
            
            comparison = np.asarray(value, dtype = 'int')
            
            if not bool:
                nms_gradient[y,x,:] = gradient[y, x, :]*comparison
            
            # if np.array_equal(comparison, np.asarray([1, 1, 1])):
            #     print('\nlocation', y, x)
            #     print('nms', nms_gradient[y,x,:])
            #     print('hess', gradient[y,x,:])
            else:
                nms_gradient[y,x,:] = comparison
    
    return nms_gradient