import sys

sys.path.append('../')
sys.path.append('../../binary_classifier')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import itertools
from tqdm import tqdm
import pickle
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras import models

reward_table = []


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


class Quantum_T4_2D:  # 2D
    # in the test environment, we start at a particular location
    # then, we start from there, we can not pre-define the blocks.

    def possible_actions_from_location(self, location=None):

        if location is None:
            location = self.current_pos
        irow, icol = location

        possible_actions = []

        if self.isRepeat is True:
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
        else:

            if irow > 0 and self.visit_map[irow - 1, icol] == 0:  # up
                possible_actions.append(0)
            if irow < self.dim[0] - 1 and self.visit_map[irow + 1, icol] == 0:  # down
                possible_actions.append(1)
            if icol > 0 and self.visit_map[irow, icol - 1] == 0:  # left
                possible_actions.append(2)
            if icol < self.dim[1] - 1 and self.visit_map[irow, icol + 1] == 0:  # right
                possible_actions.append(3)
            if irow < (self.dim[0] - 1) and icol < (self.dim[1] - 1) and self.visit_map[
                irow + 1, icol + 1] == 0:  # decrease d1 and d2
                possible_actions.append(4)
            if (irow > 0) and (icol > 0) and self.visit_map[irow - 1, icol - 1] == 0:  # increase d1 and  d2
                possible_actions.append(5)

        # possible_actions=[0,1,2,3,4,5]
        return possible_actions

    def construct_block_given_starting_locations(self, starting_pixel_locs):

        w1 = self.bh
        w2 = self.bw

        [idx1, idx2] = starting_pixel_locs

        img = self.image

        # img=self.image
        extended_img = self.extended_image

        self.max_d1, self.max_d2 = img.shape
        # print('Shape for patching',self.imgheight,self.imgwidth)

        self.current_pos = np.copy([np.int(idx1 / w1), np.int(idx2 / w2)])
        self.starting_loc = np.copy(self.current_pos)

        # scale the data to 0-1
        range_data = [np.min(extended_img), np.max(extended_img)]

        nDim1 = math.ceil(self.max_d1 / w1)
        nDim2 = math.ceil(self.max_d2 / w2)

        count = 0

        patch_data = [0] * nDim1
        image_largepatch_data = [0] * nDim1
        image_smallpatch_data = [0] * nDim1

        MaxExtImg_d1 = extended_img.shape[0]
        MaxExtImg_d2 = extended_img.shape[1]

        maxMu, minMu, maxSig, minSig = 0, 10, 0, 10

        for ii in range(nDim1):
            patch_data[ii] = [0] * nDim2
            image_largepatch_data[ii] = [0] * nDim2
            image_smallpatch_data[ii] = [0] * nDim2

            for jj in range(nDim2):

                # expand to 64x64
                patch = extended_img[max(0, ii * w1 - w1):min(ii * w1 + w1 + w1, MaxExtImg_d1),
                        max(0, jj * w2 - w2):min(jj * w2 + w2 + w2, MaxExtImg_d2)]  # 2D
                count += 1

                size_patch = patch.shape

                # find p1,p2,p3 equally between l1,l2,l3
                mypp = [0] * 3
                pp_block = [0] * 2  # 2D
                for dd in range(2):  # number of dimension
                    mypp[dd] = [0] * 5
                    temp = np.linspace(0, size_patch[dd], num=5)
                    temp = temp.astype(int)
                    mypp[dd] = temp.tolist()

                    pp_block[dd] = [[mypp[dd][0], mypp[dd][2]], [mypp[dd][1], mypp[dd][3]],
                                    [mypp[dd][2], mypp[dd][4]]]

                pp_blocks = list(itertools.product(*pp_block))
                # 27 elements
                temp_patch = []
                image_smallpatch_data[ii][jj] = [0] * len(pp_blocks)

                for kk, mypp in enumerate(pp_blocks):  # 27 items

                    temp = patch[mypp[0][0]:mypp[0][1],
                           mypp[1][0]:mypp[1][1]]

                    temp2 = temp
                    temp_patch += [np.mean(temp2), np.std(temp2)]
                    image_smallpatch_data[ii][jj][kk] = np.copy(temp2)

                    minMu = min(minMu, np.mean(temp2))
                    maxMu = max(maxMu, np.mean(temp2))
                    minSig = min(minSig, np.std(temp2))
                    maxSig = max(maxSig, np.std(temp2))

                patch_data[ii][jj] = temp_patch
                image_largepatch_data[ii][jj] = patch
                patch_data[ii][jj] = self.block_splitting(image_largepatch_data[ii][jj],minMu,maxMu,minSig,maxSig)

        return patch_data, [nDim1, nDim2], range_data, image_largepatch_data, image_smallpatch_data

    def block_splitting(self,measurement,minMu,maxMu,minSig,maxSig):

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

        normalised_mean = np.zeros_like(mean_current)
        normalised_stds = np.zeros_like(stds_current)

        for i in range(9):
            normalised_mean[i] = mean_current[i] / (maxMu - minMu)
            normalised_stds[i] = stds_current[i]/ (maxSig - minSig)

        current_statistics = np.concatenate((normalised_mean, normalised_stds))

        return current_statistics

    def take_reward_table(self):
        reward_table = (-0.05) * np.ones((self.dim))
        [id1, id2] = np.where(self.isquantum == 1)
        reward_table[id1, id2] = 5
        return reward_table

    def __init__(self, file_name="",image = None, file = True, starting_pixel_loc=[0, 0], bh=18, bw=18, isRepeat=True, offset=0.0):
        # img_row_idx and img_col_idx are the pixel indices, not the block indices
        self.isRepeat = isRepeat  # allow reselect visited location
        self.bw = bw  # block width
        self.bh = bh  # block height
        # action space
        self.K = 6

        self.offset = offset
        if file == False:
            self.image = image

        else:
            # load multiple data scan into the memory
            try:
                strFile = "../data/{}.p".format(file_name)

                self.image = pickle.load(open(strFile, "rb"))

            except:
                strFile = "../data/{}.npy".format(file_name)
                self.image = np.load(strFile)

        self.image = self.image + self.offset
        # find min positive
        idxPos = np.where(self.image > 0)
        min_pos = np.min(self.image[idxPos])
        idxNegative = np.where(self.image < 0)
        self.image[idxNegative] = min_pos
        self.image = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))
        self.img_dim = self.image.shape
        # padding to four direction
        # self.extended_image=np.pad(self.image, (16, 16), 'constant',constant_values=0)
        self.extended_image = np.pad(self.image, (int(self.bh / 2), int(self.bw / 2)), 'edge')
        # base on this location, construct the data
        self.data, self.dim, self.range_data, self.image_largepatch_data, self.image_smallpatch_data = \
            self.construct_block_given_starting_locations(starting_pixel_loc)
        # D is the dimension of each patch
        # self.dim is the dimension of blocks in the images
        self.D = len(self.data[0][0])
        self.current_pos = np.copy(self.starting_loc)
        self.pre_classify()
        self.where_is_quantum()
        self.reward_table = self.take_reward_table()
        self.visit_map = np.zeros_like(self.reward_table)

    def pre_classify(self):
        self.mid_point_x = math.floor(len(self.image[:, 0]) / 2.0)
        self.mid_point_y = math.floor(len(self.image[0, :]) / 2.0)
        self.trace_x = self.image[self.mid_point_x, :]
        self.trace_y = self.image[:, self.mid_point_y]
        self.trace_range = max(self.trace_x) - min(self.trace_x)
        self.threshold_1 = self.trace_range * 0.3
        self.threshold_2 = self.trace_range * 0.02

    def get_state_and_location(self):
        id1, id2 = self.current_pos
        self.visit_map = np.zeros_like(self.reward_table)
        return np.reshape(self.data[id1][id2], (-1, 2 * 9)), np.copy(self.current_pos)

    def get_state(self, positions):
        id1, id2 = positions
        return np.reshape(self.data[id1][id2], (-1, 2 * 9))

    def current_state(self):
        id1, id2 = self.current_pos
        return np.reshape(self.data[id1][id2], (-1, 2 * 9))

    def get_reward(self, positions):
        id1, id2 = positions
        r = self.reward_table[id1, id2]
        r = r - 0.5 * self.visit_map[id1, id2]
        return r

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

    def set_session(self, session):
        self.session = session

    def step(self, action):
        # perform an action to move to the next state
        # 0: Decrease dim 1
        # 1: Increase dim 1
        # 2: Decrease dim 2
        # 3: Increase dim 2
        flagoutside = 0

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
        if flagoutside == 1:
            loc_x = np.copy(self.current_pos)
            r = -8  # terminate
            done = True
            obs = self.data[id1][id2]
        else:
            if self.visit_map[id1, id2] == 1:
                r = 0
                obs = np.zeros_like(self.data[id1][id2])
                done = False
                loc_x = np.copy(self.current_pos)
                return obs, r, done, loc_x
            r = self.get_reward(self.current_pos)
            self.visit_map[id1, id2] += 1
            done = False
            obs = self.data[id1][id2]
            if self.isquantum is None:
                self.where_is_quantum()
            if self.isquantum[id1, id2] == 1:
                r += 10
                done = True
            loc_x = np.copy(self.current_pos)
        return obs, r, done, loc_x

    def normalise(self,x):
        x_max = np.amax(x)
        x_min = np.amin(x)
        y = (x - x_min) / (x_max - x_min)
        return y

    def load_cnn(self):
        model_binary_classifier = models.load_model(
            '../../classifier/bias_triangle_binary_classifier.h5')
        return model_binary_classifier

    def check_for_bias_triangle(self, ii, jj):
        statistics = self.data[ii][jj]
        means = statistics[:9]
        for mean in means:
            if (mean > self.threshold_2) and (mean < self.threshold_1):
                self.threshold_test[ii, jj] += 1
        if self.threshold_test[ii, jj] == 0:
            return 0
        large_patch = self.image_largepatch_data[ii][jj]
        x, y = np.shape(large_patch)
        test_image = tf.image.resize(self.normalise(np.array(large_patch)).reshape(-1, x, y, 1), (32, 32))
        self.prediction[ii, jj] = self.model_binary_classifier.predict(test_image, steps=1)
        if self.prediction[ii, jj] > 0.7:
            return 1
        else:
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
                self.isquantum[ii, jj] = self.check_for_bias_triangle(ii, jj)
        return self.isquantum

    def reset_at_rand_loc(self):
        self.current_pos = [np.random.randint(0, self.dim[0]), np.random.randint(0, self.dim[1])]
        id1, id2 = self.current_pos
        self.visit_map = np.zeros_like(self.reward_table)
        return self.data[id1][id2], np.copy(self.current_pos)

    def reset_at_loc(self,loc):
        self.starting_loc = loc
        self.current_pos = loc
        id1, id2 = self.current_pos
        self.visit_map = np.zeros_like(self.reward_table)
        return self.data[id1][id2], np.copy(self.current_pos)

    def reset(self):
        self.current_pos = np.copy(self.starting_loc)
        id1, id2 = self.current_pos
        self.visit_map = np.zeros_like(self.reward_table)
        return self.data[id1][id2], np.copy(self.current_pos)
