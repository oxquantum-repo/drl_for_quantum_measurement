import sys
import matplotlib.pyplot as plt

sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../environments')
sys.path.append('../utilities')
sys.path.append('/environments')
sys.path.append('../utilities')

from datetime import datetime
import numpy as np

def on_device_play_episode(env, model, epsilon, show_log, MaxStep):
    t0 = datetime.now()
    sucsess = None
    loc_state_list = []

    total_training_time = 0
    num_steps_in_episode = 0
    episode_reward = 0
    timer = []

    location = env.get_location()
    print("Start location = ", location)

    statistics = env.get_statistics(location)

    loc_state_list.append(location)

    done = False
    while not done:
        if num_steps_in_episode > MaxStep or done is True:
            if num_steps_in_episode > MaxStep:
                sucsess = False
            else:
                sucsess = True
            break
        t_start = datetime.now()

        action = model.sample_action(env,statistics,location,epsilon)

        statistics, reward, done, location, revisited = env.step(action)

        while revisited == True:
            #print('Revisiting point on map')
            action = model.sample_random_action(env, statistics, location)
            statistics, reward, done, location, revisited = env.step(action)

        loc_state_list.append(location)

        episode_reward += reward
        
        num_steps_in_episode += 1

        threshold_upper = np.zeros_like(env.log_stats_mean)
        threshold_upper[:] = env.threshold_upper
        threshold_lower = np.zeros_like(env.log_stats_mean)
        threshold_lower[:] = env.threshold_lower

        if show_log == True:
            plt.plot(env.log_stats_mean)
            plt.plot(threshold_upper)
            plt.plot(threshold_lower)
            plt.title('Mean convergence')
            plt.show()

            plt.imshow(env.small_window_measurements[location[0]][location[1]])
            plt.title('Small window')
            plt.colorbar()
            plt.show()

            plt.imshow(env.total_measurement,extent=[env.block_centre_voltages[0][0][0],env.block_centre_voltages[-1][-1][0],env.block_centre_voltages[-1][-1][1],env.block_centre_voltages[0][0][1]])
            plt.title('Whole environment')
            plt.colorbar()
            plt.show()


            print("Step ", num_steps_in_episode)
            print("Action ", action)
            print("Location ", location)
            print("Centre Voltages ", env.block_centre_voltages[location[0]][location[1]])

        t_end = datetime.now()
        t_step = t_end - t_start
        timer.append(t_step)

    t0_2 = datetime.now()
    dt = t0_2 - t0
    total_time_training = dt.total_seconds()

    return episode_reward, num_steps_in_episode, total_time_training, timer, env.visit_map, loc_state_list, env, sucsess