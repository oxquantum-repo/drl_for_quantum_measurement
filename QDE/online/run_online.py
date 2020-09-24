import sys

import math

sys.path.append('../../')
import Pygor

sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../environments')
sys.path.append('../utilities')
sys.path.append('../testing_code')
sys.path.append('../data')

from on_device_play_episode import on_device_play_episode
from drl_models import Dueling_DQN_PER_2D
from prioritized_experience_replay import Memory

from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models
import random
import pickle
from tqdm import tqdm
from environment_creation import double_dot_2d

def initiate():
    IM_SIZE = 2  # 80
    N_CHANEL = 9  # this is the representation of a block by 9 blocks
    K = 6  # env.action_space.n
    D = IM_SIZE * N_CHANEL
    hidden_layer_sizes = [128, 64, 32]
    gamma = 0.5

    # number of random test
    batch_sz = 32
    count = 0
    tf.reset_default_graph()

    model = Dueling_DQN_PER_2D(D=D, K=K, batch_sz=batch_sz, hidden_layer_sizes=hidden_layer_sizes,
                               gamma=gamma, lr=2.3e-6, N_CHANEL=N_CHANEL, IM_SIZE=IM_SIZE, scope="DDQN")

    print("DRL model loaded")

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)

    saver = tf.train.Saver()

    MODEL_PATH = "../logs/2d/save_models/2d_mean_std"

    saver.restore(sess, MODEL_PATH)
    model.set_session(sess)

    print("Session started")

    return model

def reset_session(model):
    model.session.close()
    model = initiate()
    return model

def end_session(model):
    model.session.close()
    return

def run(model,epsilon,gates,gate_numbers,default_gates,initial_gate_cu_cv_num_indices,pygor_xmlip = None, show_log = False, MaxStep=100, starting_position=None, random_pixel = True, savefilename = None):
    
    name = "T4 - Basel 2"
    
    block_size = 32
    allowed_n_blocks = 21 # odd number needed
    
    pygor_mode = 'none'

    env = double_dot_2d(name, block_size, allowed_n_blocks, gates, gate_numbers, default_gates, initial_gate_cu_cv_num_indices, pygor_mode = pygor_mode, pygor_xmlip = pygor_xmlip, starting_position=starting_position,random_pixel = random_pixel)
    
    episode_reward, num_steps_in_episode, total_time_training,timer, env.visit_map, loc_state_list, env, sucsess = on_device_play_episode(env, model, epsilon, show_log, MaxStep)

    if show_log == True:
        plt.imshow(env.pre_classification_prediction,extent=[env.block_centre_voltages[0][0][0],env.block_centre_voltages[-1][-1][0],env.block_centre_voltages[-1][-1][1],env.block_centre_voltages[0][0][1]])
        plt.colorbar()
        plt.title("Pre-classifier Prediction")
        plt.show()

        plt.imshow(env.cnn_prediction,extent=[env.block_centre_voltages[0][0][0],env.block_centre_voltages[-1][-1][0],env.block_centre_voltages[-1][-1][1],env.block_centre_voltages[0][0][1]])
        plt.colorbar()
        plt.title("CNN Prediction")
        plt.show()

        plt.imshow(env.total_measurement,extent=[env.block_centre_voltages[0][0][0],env.block_centre_voltages[-1][-1][0],env.block_centre_voltages[-1][-1][1],env.block_centre_voltages[0][0][1]])
        plt.colorbar()
        plt.title("Total measurement")
        plt.show()
    

    date_time = now.strftime("%m_%d_%Y__%H_%M")
    
    run_information = {
        "Name": name,
        "Gates": gates,
        "Default gates": default_gates,
        "Plunger gate indices": initial_gate_cu_cv_num_indices,
        "Epsilon":epsilon,
        "Starting index": starting_position,
        "Episode reward": episode_reward,
        "Number of steps": num_steps_in_episode,
        "Total tuning time (seconds)": total_time_training,
        "Timed step list (s)":timer,
        "Location state list": loc_state_list,
        "Environment visit map": env.visit_map,
        "Bias triangle location": env.isquantum,
        "Small window measurements": env.small_window_measurements,
        "Small window statistics": env.small_window_statistics,
        "Block centre voltages": env.block_centre_voltages,
        "Number of blocks": env.allowed_n_blocks,
        "Block size": env.block_size,
        "1d-scan": [env.trace_0,env.trace_1],
        "Sucsess (bias triangle found)": sucsess
        }

    if savefilename == None:
        savefilename = "benchmark/run_information"+date_time

    if pygor_xmlip != None:
        pickle_out = open(savefilename+".pickle","wb")
        pickle.dump(run_information, pickle_out)
        pickle_out.close()

    print("Tuning completed")
    print('total_time_training',total_time_training)
    print('Time per step',total_time_training/num_steps_in_episode)
    
    return env,run_information

