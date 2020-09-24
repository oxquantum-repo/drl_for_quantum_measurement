import numpy as np
import pickle
import matplotlib.pyplot as plt

class mock_pygor:
    def __init__(self, file_name, block_size, offset):

        try:
            strFile = "../data/{}.p".format(file_name)

            self.image = pickle.load(open(strFile, "rb"))

        except:
            infile = open('../run_on_device/benchmark/' + file_name + '.pickle', 'rb')
            scan = pickle.load(infile)
            infile.close()

            self.image = scan['Scan data.data'].data[0]

        '''except:
            strFile = "../data/{}.npy".format(file_name)

            self.image = np.load(strFile)
        '''

        self.offset = offset

        self.image = self.image + self.offset


        self.block_size = block_size

        length = min(np.shape(self.image)[0],np.shape(self.image)[1])

        self.allowed_n_blocks = np.int(np.floor(length/block_size))

        self.env_size = np.int(self.allowed_n_blocks*self.block_size)

        self.scan = self.image[:self.env_size,:self.env_size]

        #self.scan = (self.scan - np.min(self.scan)) / (np.max(self.scan) - np.min(self.scan))

        self.start_params = np.random.randint(0,self.env_size,2)

        self.current_params = np.copy(self.start_params)

    def normalise(self,min,max):

        #self.scan = (self.scan - np.min(self.scan)) / (np.max(self.scan) - np.min(self.scan))
        self.scan =self.scan

    def get_location(self,params):

        loc_x = np.floor(params[0]/self.block_size)
        loc_y = np.floor(params[1] / self.block_size)

        return [np.int(loc_x),np.int(loc_y)]

    def setvals(self, params):

        self.current_params = np.copy(params)

        return

    def get_current_params(self):

        return np.copy(self.current_params)

    def get_params(self,location):

        if np.shape(location)[0] == 2:
            return [np.int(location[0]* self.block_size),np.int(location[1]* self.block_size)]
        else:
            return np.int(location[0]* self.block_size)

    def do1d(self,constant_loc,constant_dimension,initial,final):

        if constant_dimension == 0:
            line = self.scan[constant_loc,initial:final]
        else:
            line = self.scan[initial:final,constant_loc]

        return line

    def do2d(self,initial_x,final_x,intial_y,final_y):

        window = self.scan[initial_x:final_x,intial_y:final_y]

        return window

    def do0d(self,loc_x,loc_y):

        point = self.scan[np.int(loc_x),np.int(loc_y)]

        return point