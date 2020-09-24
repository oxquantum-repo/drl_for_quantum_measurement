import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from mock_pygor import mock_pygor

class double_dot_2d:

    def __init__(self, block_size , file_name, offset=0):

        # D is the dimension of each patch
        # self.dim is the dimension of blocks in the images
        self.block_size = block_size

        self.pygor = mock_pygor(file_name, block_size, offset)

        self.image = self.pygor.scan

        self.allowed_n_blocks = self.pygor.allowed_n_blocks

        self.D = 9
        self.K = 6
        self.dim = [self.allowed_n_blocks,self.allowed_n_blocks]
        
        # create an empty visit_map
        self.visit_map = np.zeros((self.allowed_n_blocks,self.allowed_n_blocks))
        self.isquantum = np.zeros((self.allowed_n_blocks,self.allowed_n_blocks))
        self.cnn_prediction = np.zeros((self.allowed_n_blocks,self.allowed_n_blocks))
        self.pre_classification_prediction = np.zeros((self.allowed_n_blocks,self.allowed_n_blocks))
        self.total_measurement = np.zeros((self.allowed_n_blocks*self.block_size,self.allowed_n_blocks*self.block_size))

        self.starting_pos = self.pygor.get_location(self.pygor.start_params)

        self.current_pos = self.pygor.get_location(self.pygor.get_current_params())

        # based on this location, construct the data
        self.small_window_measurements = [0] * self.allowed_n_blocks
        self.small_window_statistics = [0] * self.allowed_n_blocks

        for ii in range(self.allowed_n_blocks):
            self.small_window_measurements[ii] = [0] * self.allowed_n_blocks
            self.small_window_statistics[ii] = [0] * self.allowed_n_blocks

        self.pre_classify()

        self.model_binary_classifier = self.load_cnn()
        self.pixels_measured = 0

        measurement, statistics = self.random_point_measurement([self.starting_pos[0], self.starting_pos[1]])

        self.small_window_measurements[self.starting_pos[0]][self.starting_pos[1]], self.small_window_statistics[self.starting_pos[0]][self.starting_pos[1]] = measurement, statistics

        self.visit_map[self.starting_pos[0], self.starting_pos[1]] += 1

        classification = self.check_for_bias_triangle(self.starting_pos[0],self.starting_pos[1])

        if classification == 1:
            self.isquantum[self.starting_pos[0], self.starting_pos[1]] = 1

    def pre_classify(self):

        self.trace_0 = self.pygor.do1d(0,0,0,self.pygor.env_size)
        self.trace_1 = self.pygor.do1d(0,1,0,self.pygor.env_size)

        self.environment_max_current = max([max(self.trace_0),max(self.trace_1)])
        self.environment_min_current = min([min(self.trace_0),min(self.trace_1)])

        self.standard_deviation_trace_0 = np.std(self.trace_0)
        self.standard_deviation_trace_1 = np.std(self.trace_1)

        self.standard_deviation_for_normalisation = max([min([self.standard_deviation_trace_0,self.standard_deviation_trace_1]),abs(self.standard_deviation_trace_0-self.standard_deviation_trace_1)])

        self.trace_range_max = self.normalise_mean_function(abs(self.environment_max_current - self.environment_min_current))

        self.threshold_upper = self.trace_range_max * 0.4
        self.threshold_lower = self.trace_range_max * 0.005

        self.pygor.setvals(self.pygor.start_params)

    def get_neightborMapIndividual(self, location):

        id1, id2 = location

        norm_factor = 5.0
        output = []
        # return a 6 dimensional vector
        if id1 == 0:  # decrease d1
            output.append(0)
        else:
            output.append(self.visit_map[id1 - 1, id2] / norm_factor)

        if id1 == self.dim[0] - 1:  # increase d1
            output.append(0)
        else:
            output.append(self.visit_map[id1 + 1, id2] / norm_factor)

        if id2 == 0:  # decrease d2
            output.append(0)
        else:
            output.append(self.visit_map[id1, id2 - 1] / norm_factor)

        if id2 == self.dim[1] - 1:  # increase d2
            output.append(0)
        else:
            output.append(self.visit_map[id1, id2 + 1] / norm_factor)

        if id1 < self.dim[0] - 1 and id2 < self.dim[1] - 1:  # decrease d1 and decrease d2
            output.append(self.visit_map[id1 + 1, id2 + 1] / norm_factor)
        else:
            output.append(0)

        if id1 > 0 and id2 > 0:  # increase d1 and increase d2
            output.append(self.visit_map[id1 - 1, id2 - 1] / norm_factor)
        else:
            output.append(0)

        # replace zero by -1
        output2 = [-1 / norm_factor if o == 0 else o * 1 for o in output]
        return output2

    def get_neighborMap(self, locations):
        locations = np.asarray(locations)
        if len(locations.shape) == 1:  # 1 data point
            output = self.get_neightborMapIndividual(locations)
        else:
            output = np.apply_along_axis(self.get_neightborMapIndividual, 1, locations)

        return output

    def get_statistics(self,location):

        stats = self.small_window_statistics[location[0]][location[1]]

        return stats

    def get_location(self):

        location = np.copy(self.current_pos)

        return location

    def take_measurement(self,location):

        initial_x, final_x, intial_y, final_y = self.pygor.get_params([location[0]]),self.pygor.get_params([location[0]+1]), self.pygor.get_params([location[1]]),self.pygor.get_params([location[1]+1])

        scan = self.pygor.do2d(initial_x,final_x,intial_y,final_y)

        small_window_measurements = scan
        
        small_window_statistics = self.block_splitting(small_window_measurements)

        self.total_measurement[initial_x:final_x,intial_y:final_y] = small_window_measurements

        small_window_measurements = list(map(self.normalise_mean_function,small_window_measurements))

        return small_window_measurements, small_window_statistics

    def random_point_measurement(self,location):

        small_window_measurements = np.zeros((self.block_size,self.block_size))

        log_stats_mean = []
        log_stats_std = []

        stats = np.zeros(18)

        i = 0
        while any(stats == 0):
            small_window_measurements = self.sample_random_pixels(small_window_measurements,location)
            stats = self.block_splitting_statistics(small_window_measurements)
            log_stats_mean.append(stats[0])
            log_stats_std.append(stats[8])
            i+=1
        small_window_measurements = self.sample_random_pixels(small_window_measurements,location)
        new_stats = self.block_splitting_statistics(small_window_measurements)

        while sum(abs((stats - new_stats)/stats)) > 0.1:
            i+=1
            small_window_measurements = self.sample_random_pixels(small_window_measurements,location)
            stats = new_stats
            new_stats = self.block_splitting_statistics(small_window_measurements)

            log_stats_mean.append(new_stats[0])
            log_stats_std.append(new_stats[8])

        self.pixels_measured += i
        small_window_statistics = self.block_splitting_statistics(small_window_measurements)

        return small_window_measurements, small_window_statistics

    def sample_random_pixels(self,small_window_measurements,location):

        x, y = random.randint(0, self.block_size - 1), random.randint(0, self.block_size - 1)

        loc_x,loc_y = self.pygor.get_params([location[0]])+x, self.pygor.get_params([location[1]])+y

        point_measurement = self.pygor.do0d(loc_x,loc_y)
        small_window_measurements[x, y] = point_measurement

        return small_window_measurements

    def block_splitting_statistics(self, measurement):

        measurement_size = np.shape(measurement)[0]
        n_over_2 = math.floor(measurement_size / 2.0)
        n_over_4 = math.floor(measurement_size / 4.0)
        n_3_over_4 = math.floor(3 * measurement_size / 4.0)

        # Split into blocks based:
        block_1 = measurement[0:n_over_2, 0:n_over_2]
        block_2 = measurement[0:n_over_2, n_over_2:measurement_size]
        block_3 = measurement[n_over_2:measurement_size, 0:n_over_2]
        block_4 = measurement[n_over_2:measurement_size, n_over_2:measurement_size]
        block_5 = measurement[n_over_4:n_3_over_4, n_over_4:n_3_over_4]
        block_6 = measurement[n_over_4:n_3_over_4, 0:n_over_2]
        block_7 = measurement[n_over_4:n_3_over_4, n_over_2:measurement_size]
        block_8 = measurement[0:n_over_2, n_over_4:n_3_over_4]
        block_9 = measurement[n_over_2:measurement_size, n_over_4:n_3_over_4]

        blocks = [block_1, block_2, block_3, block_4, block_5, block_6, block_7, block_8, block_9]
        mean_current = []
        stds_current = []
        for block in blocks:
            data_set = []
            for row in block:
                for element in row:
                    if element != 0.0:
                        data_set.append(element)
                        #print("Element",element)
            
            if data_set == []:
                data_set = 0
                
            mean_current.append(np.mean(data_set))
            stds_current.append(np.std(data_set))

        normalised_mean = list(map(self.normalise_mean_function,mean_current))
        normalised_stds = list(map(self.normalise_std_function,stds_current))
        # Concatenate data into single 18-feature array:
        current_statistics = np.concatenate((normalised_mean, normalised_stds))

        return current_statistics  # mean_current, stds_current

    def normalise_mean_function(self, mean):
        normalised_mean = (mean - self.environment_min_current)/(self.environment_max_current - self.environment_min_current)
        return normalised_mean

    def normalise_std_function(self, std):
        normalised_std = (std)/(self.standard_deviation_for_normalisation)
        return normalised_std

    def block_splitting(self,measurement):

        measurement_size = np.shape(measurement)[0]
        n_over_2 = math.floor(measurement_size / 2.0)
        n_over_4 = math.floor(measurement_size / 4.0)
        n_3_over_4 = math.floor(3 * measurement_size / 4.0)

        # Split into blocks based:
        block_1 = measurement[0:n_over_2, 0:n_over_2]
        block_2 = measurement[0:n_over_2, n_over_2:measurement_size]
        block_3 = measurement[n_over_2:measurement_size, 0:n_over_2]
        block_4 = measurement[n_over_2:measurement_size, n_over_2:measurement_size]
        block_5 = measurement[n_over_4:n_3_over_4, n_over_4:n_3_over_4]
        block_6 = measurement[n_over_4:n_3_over_4, 0:n_over_2]
        block_7 = measurement[n_over_4:n_3_over_4, n_over_2:measurement_size]
        block_8 = measurement[0:n_over_2, n_over_4:n_3_over_4]
        block_9 = measurement[n_over_2:measurement_size, n_over_4:n_3_over_4]

        # Concatenate data into single 18-feature array:
        mean_current = np.array(
            [np.mean(block_1), np.mean(block_2), np.mean(block_3), np.mean(block_4), np.mean(block_5), np.mean(block_6),
             np.mean(block_7), np.mean(block_8), np.mean(block_9)])
        stds_current = np.array(
            [np.std(block_1), np.std(block_2), np.std(block_3), np.std(block_4), np.std(block_5), np.std(block_6),
             np.std(block_7), np.std(block_8), np.std(block_9)])

        normalised_mean = list(map(self.normalise_mean_function,mean_current))
        normalised_stds = list(map(self.normalise_std_function,stds_current))

        current_statistics = np.concatenate((normalised_mean, normalised_stds))

        return current_statistics

    def possible_actions_from_location(self, location=None):

        if location is None:
            location = self.current_pos
        irow, icol = location

        possible_actions = []

        if irow > 0:  # decrease d1
            possible_actions.append(0)
        if irow < self.dim[0] - 1:  # increase d1
            possible_actions.append(1)
        if icol > 0:  # decrease d2
            possible_actions.append(2)
        if icol < self.dim[1] - 1:  # increase d2
            possible_actions.append(3)
        if irow < (self.dim[0] - 1) and icol < (self.dim[1] - 1):  # decrease d1 and d2
            possible_actions.append(4)
        if (irow > 0) and (icol > 0):  # increase d1 and  d2
            possible_actions.append(5)

        # possible_actions=[0,1,2,3,4,5]

        return possible_actions

    def step(self, action):
        # perform an action to move to the next state

        # 0: Decrease dim 1
        # 1: Increase dim 1
        # 2: Decrease dim 2
        # 3: Increase dim 2
        # 4: Decrease both
        # 5: Increase both
        flagoutside = 0
        # flagRepeat=0

        if action == 0:
            if self.current_pos[0] == 0:
                flagoutside = 1
                print("cannot decrease d1")
            else:
                self.current_pos[0] = self.current_pos[0] - 1
        elif action == 1:
            if self.current_pos[0] == self.dim[0] - 1:
                flagoutside = 1
                print("cannot increase d1")
            else:
                self.current_pos[0] = self.current_pos[0] + 1
        elif action == 2:
            if self.current_pos[1] == 0:
                flagoutside = 1
                print("cannot decrease d2")
            else:
                self.current_pos[1] = self.current_pos[1] - 1
        elif action == 3:
            if self.current_pos[1] == self.dim[1] - 1:
                flagoutside = 1
                print("cannot decrease d2")
            else:
                self.current_pos[1] = self.current_pos[1] + 1
        elif action == 4:
            if self.current_pos[0] < self.dim[0] - 1 and self.current_pos[1] < self.dim[1] - 1:
                self.current_pos[1] = self.current_pos[1] + 1
                self.current_pos[0] = self.current_pos[0] + 1
            else:
                flagoutside = 1
                print("cannot increase both d1 and d2")

        elif action == 5:
            if self.current_pos[0] > 0 and self.current_pos[1] > 0:
                self.current_pos[1] = self.current_pos[1] - 1
                self.current_pos[0] = self.current_pos[0] - 1
            else:
                flagoutside = 1
                print("cannot decrease both d1 and d2")
        else:
            print("action is 0-6")

        id1, id2 = self.current_pos

        if self.visit_map[id1, id2] == 1:
            reward = 0
            statistics =self.small_window_statistics[id1][id2]
            done = False
            location = np.copy(self.current_pos)
            revisited = True
            return statistics, reward, done, location, revisited

        measurement, statistics = self.random_point_measurement([id2, id1])

        self.small_window_measurements[id1][id2], self.small_window_statistics[id1][id2] = measurement, statistics

        self.visit_map[id1, id2] += 1
        reward = -1

        done = False
        revisited = False

        classification = self.check_for_bias_triangle(id1,id2)

        if classification == 1:
            reward += 100
            done = True

        location = np.copy(self.current_pos)

        return statistics, reward, done, location, revisited

    def normalise(self,x):
        x_max = np.amax(x)
        x_min = np.amin(x)
        y = (x - x_min) / (x_max - x_min)
        return y


    def load_cnn(self):
        #model_binary_classifier = models.load_model('../../classifier/bias_triangle_binary_classifier.h5')
        model_binary_classifier = models.load_model('../../classifier/bias_triangle_binary_classifier.h5')
        return model_binary_classifier

    def predict_cnn(self,ii,jj):
        
        large_patch = self.small_window_measurements[ii][jj]
        x, y = np.shape(large_patch)
        test_image = tf.image.resize(self.normalise(np.array(large_patch)).reshape(-1, x, y, 1), (32, 32))

        self.cnn_prediction[ii,jj] = self.model_binary_classifier.predict(test_image, steps=1)

        return
        
    def check_for_bias_triangle(self, ii, jj):
        
        statistics = self.small_window_statistics[ii][jj]

        self.small_window_measurements[ii][jj], statistics = self.take_measurement([ii, jj])

        self.pre_classification_prediction[ii, jj] = 0

        means = statistics[:9]
        for mean in means:
            if (abs(mean) > self.threshold_lower) and (abs(mean) < self.threshold_upper):
                self.pre_classification_prediction[ii, jj] += 1

        if self.pre_classification_prediction[ii,jj] == 0:
            return 0

        measurement, statistics = self.take_measurement([ii, jj])

        self.pixels_measured += self.block_size*self.block_size
        self.small_window_measurements[ii][jj], self.small_window_statistics[ii][jj] = measurement, statistics

        self.predict_cnn(ii,jj)

        if self.cnn_prediction[ii,jj] > 0.5:
            self.isquantum[ii,jj] = 1
            return 1
        else:
            self.isquantum[ii, jj] = 0
            return 0

    def where_is_quantum(self):
        self.model_binary_classifier = self.load_cnn()
        # return a map telling the quantum location
        ndim1, ndim2 = self.dim
        self.isquantum = np.zeros(self.dim)
        self.threshold_test = np.zeros(self.dim)
        self.prediction = np.zeros(self.dim)
        for ii in tqdm(range(ndim1)):
            for jj in range(ndim2):
                # self.isquantum[ii,jj]=self.check_where_is_quantum_2d(ii,jj)
                self.isquantum[ii, jj] = self.check_for_bias_triangle(ii, jj)

        return self.isquantum

