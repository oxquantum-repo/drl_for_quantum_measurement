import sys

import math

sys.path.append('../')
import mock_pygor

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../environments')
sys.path.append('../utilities')
sys.path.append('../testing_code')
sys.path.append('../data')

from offline_test_play_episode import offline_test_play_episode
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
from offline_test_environment_creation import double_dot_2d

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

    return model

def reset_session(model):
    model.session.close()
    model = initiate()
    return model

def end_session(model):
    model.session.close()
    return

def run(model,epsilon,file_name, MaxStep=60, show_log = False, save = False):
    
    name = file_name
    
    block_size = 32

    env = double_dot_2d(block_size , file_name)

    '''print("Environment initialised")'''
    
    episode_reward, num_steps_in_episode, total_time_training, env.visit_map, loc_state_list, env = offline_test_play_episode(env, model, epsilon, MaxStep, show_log)

    '''plt.imshow(env.pre_classification_prediction)
    plt.colorbar()
    plt.title("Pre-classifier Prediction")
    plt.show()
    
    plt.imshow(env.cnn_prediction)
    plt.colorbar()
    plt.title("CNN Prediction")
    plt.show()

    plt.imshow(env.visit_map)
    plt.title("Visit Map")
    plt.show()
    
    plt.imshow(env.total_measurement)
    plt.colorbar()
    plt.title("Total measurement")
    plt.show()

    route = np.zeros_like(env.image)

    for index, item in enumerate(env.visit_map):
        for index2, item2, in enumerate(item):
            if item2 == 1:
                route[index * block_size:(index + 1) * block_size, index2 * block_size:(index2 + 1) * block_size] = env.image[index * block_size:(index + 1) * block_size, index2 * block_size:(index2 + 1) * block_size]

    plt.imshow(route)
    plt.title("Trajectory")
    plt.xlabel('Plunger Gate A')
    plt.ylabel('Plunger Gate B')
    #plt.savefig("trajectory5.png", transparent=True)
    plt.show()
    
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y__%H_%M")
    print("date and time:",date_time)'''
    
    run_information = {
        "Name": name,
        "Episode reward": episode_reward,
        "Number of steps": num_steps_in_episode,
        "Total training time (seconds)": total_time_training,
        "Location state list": loc_state_list,
        "Environment visit map": env.visit_map,
        "Bias triangle location": env.isquantum,
        "Small window measurements": env.small_window_measurements,
        "Small window statistics": env.small_window_statistics,
        }
    
    #print(run_information)
    if save == True:
        pickle_out = open("fine_tuning/mock_run_information"+date_time+".pickle","wb")
        pickle.dump(run_information, pickle_out)
        pickle_out.close()
    
    #np.save('fine_tuning/total_measurement'+date_time, env.total_measurement)
    '''print("Play episode completed")
    print('total_time_training',total_time_training)
    print('Time per step',total_time_training/num_steps_in_episode)'''
    
    return env,episode_reward,num_steps_in_episode



