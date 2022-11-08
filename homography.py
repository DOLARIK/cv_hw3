import numpy as np
import random
from itertools import permutations
import math

class RANSAC:
    def __init__(self, pairs):
        self.pairs = pairs
        
        self.fits = []
        
    def select_group(self, num_pairs, previously_chosen_groups):
        random.seed(42)
        
        new_group = random.sample(list(self.pairs.keys()), num_pairs)
        while tuple(new_group) in previously_chosen_groups:
            new_group = random.sample(list(self.pairs.keys()), num_pairs)
        
        return new_group
    
    def _generate_source_destination_matrix(self, group):
        source = []
        destination = []
        
        for pair in group:
            xs, ys = pair[0]
            xd, yd = pair[1]
            
            row_s = [[xs, ys, 0, 0, 1, 0],
                     [0, 0, xs, ys, 0, 1]]
            
            source.extend(row_s)
            
            row_d = [[xd],
                     [yd]]
            
            destination.extend(row_d)
            
        source_matrix = np.asarray(source)
        destination_matrix = np.asarray(destination)
        
        return source_matrix, destination_matrix
    
    def _get_affine_parameters(self, source_matrix, destination_matrix):
        s = source_matrix
        d = destination_matrix
        
        m = 10^-9
        # affine = np.dot(np.linalg.inv(s), d)
        affine = np.dot(np.dot(np.linalg.inv(np.dot(s.T, s) + np.eye(np.dot(s.T, s).shape[1])*m), s.T), d)
        
        # affine = s \ d
        
        affine_homogeneous = [[affine[0,0], affine[1,0], affine[4,0]],
                              [affine[2,0], affine[3,0], affine[5,0]],
                              [0          , 0          , 1          ]]
        
        return np.asarray(affine_homogeneous)
    
    def _convert_to_homogeneous(self, coord):
        x, y = coord
        return np.asarray([[x],[y],[1]])
    
    def _convert_from_homogeneous(self, coord):
        x = coord[0,0]/coord[2,0]
        y = coord[1,0]/coord[2,0]
        
        return (x,y)
    
    def _get_inliers(self, affine, threshold):
        inliers = []
        
        for pair in self.pairs:
            distance = self._calculate_distance(pair, affine)
            
            if distance <= threshold:
                inliers.append(pair)
        
        inliers_percentage = len(inliers)/len(self.pairs)
        
        return inliers, inliers_percentage
            
            
    def _calculate_distance(self, pair, affine):
        source, destination = pair
        
        source_hom = self._convert_to_homogeneous(source)
        projection_hom = np.dot(affine, source_hom)
        projection = self._convert_from_homogeneous(projection_hom)
                
        distance = ((destination[0] - projection[0])**2 + (destination[1] - projection[1])**2)**.5
        
        return distance
    
    def adaptively_fit(self, 
                       probability, 
                       threshold, 
                       num_inliers = 15,
                       num_pairs = 4):
        print('\nFITTING!!')
        
        N = float('inf')
        sample_count = 0
        
        inliers_count_milestone = num_inliers
        inliers_milestone = []
        
        print(N, inliers_count_milestone)
        
        fit = ()
        
        previously_chosen_groups = []
        
        while N > sample_count:
            group = self.select_group(num_pairs, previously_chosen_groups)
            source, destination = self._generate_source_destination_matrix(group)
            affine = self._get_affine_parameters(source, destination)
            inliers, inliers_percentage = self._get_inliers(affine, threshold)
            
            for group in list(permutations(group)):
                previously_chosen_groups.append(group)
            
            if len(inliers) > inliers_count_milestone:
                inliers_count_milestone = len(inliers)
                inliers_milestone = inliers.copy()
                
                fit = affine               


                e = 1 - inliers_count_milestone/len(self.pairs) # outleir percentage
                N = math.log(1 - probability)/(.0001 + math.log(1 - (1 - e)**num_pairs))
                print(N, inliers_count_milestone)
            
            
            sample_count += 1
            
        
        print('----------------------------------------------------------')
        
        return fit, inliers_milestone