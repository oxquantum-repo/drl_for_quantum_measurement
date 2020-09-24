# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:37:50 2019

@author: Vu
"""

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

def play_train_episode(  env_list,  total_t, n_episode, experience_replay_buffer,  model,  gamma,  
                     batch_size,  epsilon,  epsilon_change,  epsilon_min,summary_writer, MaxStep=100):
    
    t0 = datetime.now()
    # Reset the environment
    #state,loc_state = env.reset()
    
    # randomly select env
    env_id=np.random.randint(0,len(env_list))
    env=env_list[env_id]
    
    '''if np.random.random() < 0.1:
        state,loc_state = env.reset_at_rand_loc()
    else:
        state,loc_state = env.reset()'''

    state,loc_state = env.reset_at_rand_loc()
    
    loss = None

    
    total_time_training = 0
    num_steps_in_episode = 0
    episode_reward = 0

    done = False
    while not done:        
        if num_steps_in_episode>MaxStep:
            break           
        
        model.count_exp+=1
        # Take action
        action=model.sample_action(env,state,loc_state,epsilon)
        prev_state=state
        prev_loc_state=loc_state
        #prev_obs=obs
        state, reward, done, loc_state = env.step(action)
        while state.all()==0:
            #print('Revisiting point on map')
            action = model.sample_random_action(env,state,loc_state)
            state, reward, done, loc_state = env.step(action)

        episode_reward += reward
        episode_reward += -1
        
        if num_steps_in_episode==MaxStep and done is False:
            reward=-10
            done=True
            
        # update the model
        
        neighborMaps=env.get_neighborMap(prev_loc_state)
        neighborMaps_next=env.get_neighborMap(loc_state)
        
        model.add_experience(prev_state,prev_loc_state,action,reward,
                             state,loc_state,done,neighborMaps,neighborMaps_next,env.id)

        # Train the model, keep track of time
        t0_2 = datetime.now()
        summary,loss=model.fit_prioritized_exp_replay(env_list)

        #loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)
        dt = datetime.now() - t0_2

        total_time_training += dt.total_seconds()
        num_steps_in_episode += 1
    
    
        #state = next_state
        total_t += 1
    
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        
        #summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
    if summary is not None:
        summary_writer.add_summary(summary,n_episode)

    episode_reward=float("{0:.2f}".format(episode_reward))

    return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, \
        total_time_training/num_steps_in_episode, epsilon, loss, summary_writer
              
def generate_rand_point(env):
    point=[0]*len(env.img_dim)
    for dd in range(len(env.img_dim)):
        point[dd]=np.random.randint(0,env.img_dim[dd])
    return point


def play_test_episode_from_location(env, model, epsilon,MaxStep=60):

    # Reset the environment
    
    loc_state_list=[]
    position_list_x, position_list_y = [], []

    total_time = 0
    num_steps_in_episode = 0
    episode_reward = 0
    #state, loc_state = env.reset_at_rand_loc()
    state, loc_state = env.reset()
    #print(loc_state)
    #state,loc_state = env.get_state_and_location()
    done = False
    while not done:
        x, y = env.current_pos

        position_list_x.append(x)
        position_list_y.append(y)

        loc_state_list.append([x,y])

        if num_steps_in_episode > MaxStep:
            break

        model.count_exp += 1
        # Take action
        action = model.sample_action(env, state, loc_state, epsilon)
        prev_state = state
        prev_loc_state = loc_state
        # prev_obs=obs
        state, reward, done, loc_state = env.step(action)
        while state.all() == 0:
            '''print('Revisiting point on map')
            print("Revisited action",action)
            print("Revisited state location", loc_state)'''
            action = model.sample_random_action(env, state, loc_state)
            state, reward, done, loc_state = env.step(action)

        episode_reward += reward
        episode_reward += -1

        #loc_state_list.append(loc_state)

        if num_steps_in_episode == MaxStep and done is False:
            reward = -10
            done = True

        # update the model

        neighborMaps = env.get_neighborMap(prev_loc_state)
        neighborMaps_next = env.get_neighborMap(loc_state)

        #model.add_experience(prev_state, prev_loc_state, action, reward,state, loc_state, done, neighborMaps, neighborMaps_next, env.id)

        # Train the model, keep track of time
        t0_2 = datetime.now()
        #summary, loss = model.fit_prioritized_exp_replay(env_list)

        # loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)

        num_steps_in_episode += 1

        # state = next_state


    return episode_reward, num_steps_in_episode, env.visit_map, loc_state_list, env,position_list_x,position_list_y


def burn_in_experience(env_list, experience_replay_buffer, model, MaxStep=40):
    epsilon = 1

    # randomly select env
    env_id = np.random.randint(0, len(env_list))

    env = env_list[env_id]

    if np.random.random() < 0.1:
        state, loc_state = env.reset_at_rand_loc()
    else:
        state, loc_state = env.reset()

    num_steps_in_episode = 0
    episode_reward = 0

    state_list = []
    prev_state_list = []
    prev_loc_state_list = []
    loc_state_list = []
    action_list = []
    reward_list = []
    done_list = []
    neighborMaps_list = []
    neighborMaps_next_list = []
    env_id_list = []
    env_id_list.append(env.id)

    done = False
    terminate = False
    while not done:
        if num_steps_in_episode > MaxStep:
            break

        model.count_exp += 1
        # Take action
        action = model.sample_action(env, state, loc_state, epsilon)

        prev_state = state
        prev_loc_state = loc_state
        # prev_obs=obs
        state, reward, done, loc_state = env.step(action)  # last output is the location of the state

        episode_reward += reward
        episode_reward += -1

        if num_steps_in_episode == MaxStep and done is False:
            reward = -10
            done = True
            terminate = True

        # update the model

        neighborMaps = env.get_neighborMap(prev_loc_state)
        neighborMaps_next = env.get_neighborMap(loc_state)

        env_id_list.append(env.id)
        neighborMaps_next_list.append(neighborMaps_next)
        neighborMaps_list.append(neighborMaps)
        done_list.append(done)
        reward_list.append(reward)
        action_list.append(action)
        loc_state_list.append(loc_state)
        prev_loc_state_list.append(prev_loc_state)
        state_list.append(state)
        prev_state_list.append(prev_state)

        num_steps_in_episode += 1

        # state = next_state

    if terminate is False:  # found the Target
        model.add_experience(prev_state_list, prev_loc_state_list, action_list, reward_list,
                             state_list, loc_state_list, done_list, neighborMaps_list, neighborMaps_next_list,
                             env_id_list)
        return True
    else:
        return False

def plot_policy(env,model):
    epsilon = 0.0
    action = np.zeros((env.dim[0],env.dim[1]))
    states = np.zeros((env.dim[0], env.dim[1]))
    for i in range(env.dim[0]):
        for j in range(env.dim[1]):
            env.reset()
            env.current_pos = [i,j]
            state, loc = env.get_state_and_location()
            action[i,j] = model.sample_action(env, state, loc, epsilon)
            states[i,j] = state[0,4]

    return action,states