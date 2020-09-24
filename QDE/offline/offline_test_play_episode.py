import sys
import matplotlib.pyplot as plt

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../environments')
sys.path.append('../utilities')
sys.path.append('/environments')
sys.path.append('../utilities')

from datetime import datetime
import numpy as np

def offline_test_play_episode(env, model, epsilon, MaxStep, show_log):
    t0 = datetime.now()

    loc_state_list = []

    total_training_time = 0
    num_steps_in_episode = 0
    episode_reward = 0

    location = env.get_location()

    statistics = env.get_statistics(location)

    loc_state_list.append(location)

    done = False
    while not done:
        if num_steps_in_episode > MaxStep or done is True:
            break

        action = model.sample_action(env,statistics,location,epsilon)

        statistics, reward, done, location, revisited = env.step(action)
        #print("Action taken")
        #print(loc_state_list)

        while revisited == True:
            action = model.sample_random_action(env, statistics, location)
            statistics, reward, done, location, revisited = env.step(action)

        loc_state_list.append(location)

        episode_reward += reward
        # Punish number of steps
        #episode_reward += -1
        #print("Episode reward", episode_reward)
        
        num_steps_in_episode += 1

        if show_log == True:

            plt.imshow(env.small_window_measurements[location[0]][location[1]])
            plt.title('Small window')
            plt.colorbar()
            plt.show()

            plt.imshow(env.total_measurement)
            plt.title('Whole environment')
            plt.colorbar()
            plt.show()


            print("Step ", num_steps_in_episode)
            print("Action ", action)
            print("Location ", location)

            if done == True and num_steps_in_episode > MaxStep:
                print('Maximum number of steps exceeded')
            elif done == True:
                print('Bias triangle found at ',location)

    t0_2 = datetime.now()
    dt = t0 - t0_2
    total_time_training = dt.total_seconds()

    return episode_reward, num_steps_in_episode, total_time_training, env.visit_map, loc_state_list, env

