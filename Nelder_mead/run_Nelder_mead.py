import sys

import math

sys.path.append('../')
import Pygor

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../environments')
sys.path.append('../utilities')
sys.path.append('../testing_code')
sys.path.append('../data')

from scipy.optimize import minimize

from datetime import datetime
import numpy as np
import timeit
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tqdm import tqdm
import warnings
import pickle


class Nelder_Mead_tuning:
    def __init__(self,region,params,pygor_mode,pygor_xmlip, MaxStep):
        self.pygor = Pygor.Experiment(mode=pygor_mode, xmlip=pygor_xmlip)
        self.model_binary_classifier = self.load_cnn()
        self.params_list = []
        self.objective_function_list = []
        self.starting_params = np.copy(params)
        self.bounds1 = np.array([self.starting_params[0] - (16*21) ,self.starting_params[0] + 16*21])
        self.bounds2 = np.array([self.starting_params[1] - (16*21) ,self.starting_params[1] + 16*21])
        self.initial_simplex = [params, params + np.array([0,75]), params + np.array([75,0])]
        self.bias_triangle_found = False
        self.MaxStep = MaxStep
        self.data_store = []
        self.current_simplex = np.copy(self.initial_simplex)
        self.evaluations = 0

    def main(self):
        print('Main')
        while self.bias_triangle_found == False and self.evaluations <= MaxStep:
            res = self.optimisation()
            self.current_simplex = res.final_simplex[0]
        print("end")
        return res

    def optimisation(self):
        print("Optimisation")
        res = minimize(self.objective_function,self.current_simplex[0],method='Nelder-Mead', options = {'maxiter':1, 'initial_simplex':self.current_simplex})
        return res

    def objective_function(self,params):
        self.evaluations += 1
        print("Objective function")
        param1, param2 = params[0],params[1]

        if param1 < self.bounds1[0] or param1 > self.bounds1[1] or param2 < self.bounds2[0] or param2 > self.bounds2[1]:
            print("Out of bounds (return 2), ",param1,param2)
            return 2

        data = self.pygor.do2d("c5",float(param1)-16.0,float(param1)+16.0,32,"c9",float(param2)-16.0,float(param2)+16.0,32)
        data = data.data[0]

        prob = self.predict_cnn(data)
        dif = np.array([1,0]) - np.array([prob, 1-prob])
        vec_norm = np.linalg.norm(dif)

        print("c5,c9 ",param1,param2)
        print('prob',prob)
        print("Objective function ",vec_norm)

        self.params_list.append([param1,param2])
        self.objective_function_list.append(vec_norm)

        print('iteration ', len(self.params_list))

        if prob > 0.5:
            plt.imshow(data)
            plt.title('Bias triangle')
            plt.colorbar()
            plt.show()
            print("Bias triangle found")
            self.bias_triangle_found = True
            self.data_store.append(data)

            return vec_norm
        self.data_store.append(data)
        self.call_back_bool = False
        self.bias_triangle_found = False

        return vec_norm

    def normalise(self, x):
        x_max = np.amax(x)
        x_min = np.amin(x)
        y = (x - x_min) / (x_max - x_min)
        return y

    def load_cnn(self):
        model_binary_classifier = models.load_model(
            '../../classifier/bias_triangle_binary_classifier.h5')
        return model_binary_classifier

    def predict_cnn(self, measurement):
        x, y = np.shape(measurement)
        test_image = tf.image.resize(self.normalise(np.array(measurement)).reshape(-1, x, y, 1), (32, 32))
        cnn_prediction = self.model_binary_classifier.predict(test_image, steps=1)
        return cnn_prediction[0][0]

def run_tuning(region,starting_indices, pygor_xmlip, MaxStep):

    times = []
    outcome = []
    parameters = []
    data_log = []
    steps = []

    for item in tqdm(starting_indices):
        pygor_mode = 'none'
        t0 = datetime.now()

        params = (np.array(item) - np.array([math.floor(21 / 2.0), math.floor(21 / 2.0)])) * np.array([32, 32]) + region

        tune = Nelder_Mead_tuning(region, params, pygor_mode, pygor_xmlip, MaxStep)
        res = tune.main()

        t_run = datetime.now() - t0
        times.append(t_run.total_seconds())
        outcome.append(tune.bias_triangle_found)
        parameters.append(tune.params_list)
        data_log.append(tune.data_store)
        steps.append(tune.evaluations)

        print('Run time ', t_run.total_seconds())

        print('Sucsess ', tune.bias_triangle_found)

        print('res ', res)

    print("TIMES", times)

    information = {"Times(s)": times,
                   "parameters(mV)": parameters,
                   "success(bool)": outcome,
                   "data": data_log,
                   "steps": steps,
                   "datetime": datetime.now()}

    pickle_out = open('benchmark/regime4_Nelder_Mead_tuning.pickle', "wb")
    pickle.dump(information, pickle_out)
    pickle_out.close()
    return information

pygor_xmlip = 'http://129.67.86.107:8000/RPC2'
MaxStep = 40
region = np.array([-1025.76251066,-1037.70755352])
starting_indices =[[11,10]]
''' [[12, 17],
        [18, 14],
        [ 3,  0],
        [11, 16],
        [17, 20],
        [16, 18],
        [14,  8],
        [4, 14],
        [14,  9],
        [ 2,  8]]
'''

run_tuning(region,starting_indices, pygor_xmlip, MaxStep)