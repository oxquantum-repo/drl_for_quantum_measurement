# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:35:17 2019

@author: Vu
"""
import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../utilities')
sys.path.append('../environments')
sys.path.append('../data')
from tqdm import tqdm

sys.path.append('../testing_code')
from utility_plot_arrow import plot_arrow_to_file
import numpy as np
import tensorflow as tf
import logging
from datetime import datetime
logging.basicConfig(level=logging.DEBUG,format='%(process)d-%(levelname)s-%(message)s')

import matplotlib.pyplot as plt 
import random
#from prioritized_experience_replay import Memory
from environment_2d import Quantum_T4_2D

import pickle
import os


from print_trajectories_policies import print_trajectory_from_location,final_policy_on_test,get_value_state_on_test
from play_episodes import play_train_episode, play_test_episode_from_location,burn_in_experience

from drl_models import Dueling_DQN_PER_2D

IM_SIZE = 2 #80
N_CHANEL=9 # this is the representation of a block by 9 blocks
K = 6 #env.action_space.n

import warnings
warnings.filterwarnings("ignore")

#FILE_NAME="T4_scan_data_res_480_win_480"
#FILE_NAME="T4_scan_data_res_350_win_350"

'''File_Name_List = ["T4_scan_data_res_320_win_320", "T4_scan_data_res_350_win_350",
                  "T4_scan_data_res_400_win_400_sep", "T4_scan_data_res_480_win_480"]'''
File_Name_List = ["rotated_T4_scan_data_res_320_win_320", "rotated_T4_scan_data_res_350_win_350",
                  "rotated_T4_scan_data_res_400_win_400_sep", "rotated_T4_scan_data_res_480_win_480"]

n_env = len(File_Name_List)
env_list=[0]*n_env
for n in range(n_env):
    env_list[n] = Quantum_T4_2D(File_Name_List[n],isRepeat=True,offset=2.0e-10)
    env_list[n].id = n
    plt.imshow(env_list[n].image)
    plt.title(File_Name_List[n])
    plt.colorbar()
    plt.savefig(File_Name_List[n]+'.png',transparent = True)
    plt.show()
    plt.imshow(env_list[n].threshold_test)
    plt.title(File_Name_List[n] +" Pre-classify")
    plt.colorbar()
    plt.savefig(File_Name_List[n] + '_pre_classifier.png', transparent=True)
    plt.show()
    plt.imshow(env_list[n].prediction)
    plt.title(File_Name_List[n] +" CNN")
    plt.colorbar()
    plt.savefig(File_Name_List[n] + '_cnn_prediction.png', transparent=True)
    plt.show()
    plt.imshow(env_list[n].isquantum)
    plt.title(File_Name_List[n] +" Classification")
    plt.colorbar()
    plt.savefig(File_Name_List[n] + '_classification.png', transparent=True)
    plt.show()

#env1 = env_list[0]
#env2 = env_list[1]
#print(env_list[0])

# this is for printing purpose
initial_gate_c5_c9=[ -570.,-940]
window=350
myxrange=np.linspace(initial_gate_c5_c9[1]-window/2,initial_gate_c5_c9[1]+window/2,4).astype(int)
myyrange=np.linspace(initial_gate_c5_c9[0]-window/2,initial_gate_c5_c9[0]+window/2,4).astype(int)
myxrange=myxrange[::-1]
myyrange=myyrange[::-1]


np.random.seed(1)
random.seed(1)   
tf.set_random_seed(1)


tf.reset_default_graph()   
   
# create multiple environment

starting_pixel_loc_list=[[20,340],[320,15]]

#starting_pixel_loc_list=[[100,100],[100,200],[150,200],[50,450],[80,480],[350,50],[390,50],[320,180],[395,195],[350,15]]

n_env=len(starting_pixel_loc_list)


D = env_list[0].D
K = env_list[0].K

hidden_layer_sizes = [128,64,32]

gamma = 0.5
#batch_sz = 32
#num_episodes =10100
num_episodes = 9000
total_t = 0
experience_replay_buffer = []
episode_rewards = np.zeros(num_episodes)
myloss = np.zeros(num_episodes)

last_100_avg=np.zeros(num_episodes)
last_100_avg_step=np.zeros(num_episodes)

num_steps=np.zeros(num_episodes)
episode_rewards_Test=[]
num_steps_Test=[]

episode_rewards_Test_B=[]
episode_rewards_Test_SC=[]
episode_rewards_Test_SD=[]

num_steps_Test_B=[]
num_steps_Test_SC=[]
num_steps_Test_SD=[]

# epsilon
eps = 1.0
eps_min = 0.1
eps_change = (eps - eps_min) / (3*num_episodes)

# number of random test
batch_sz=32
count=0


model = Dueling_DQN_PER_2D(D=D,K=K,batch_sz=batch_sz,hidden_layer_sizes=hidden_layer_sizes,    
                       gamma=gamma, lr=2.3e-6,    N_CHANEL=N_CHANEL,IM_SIZE=IM_SIZE,scope="DDQN")
  
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()

def make_session(n):

    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=n, intra_op_parallelism_threads=n)) 

#cpu_count = os.cpu_count()
#sess = make_session(cpu_count)

sess.run(init)

model.set_session(sess)

# Create models

# Set the logs writer to the folder /tmp/tensorflow_logs
summary_writer = tf.summary.FileWriter('../logs/2d', graph=sess.graph)

print("Populating experience replay buffer...")

starting_loc_test=[[50,40],[40,200],[30,200],[40,340],[50,340],[50,340],[15,320],[35,345],[20,320],[30,340], # barrier
                   [340,5],[340,10],[320,15],[295,20],[265,15],[340,25],[340,30],[310,25],[285,40],[275,25], # short circut
                   [250,250],[200,200],[180,180],[160,165],[195,195],[230,240],[220,210],[190,210],[190,185],[215,225]]


nTest=len(starting_loc_test)

optimal_policy_list=[]
optimal_policy_list_2=[]
optimal_val_list=[]
optimal_val_list_2=[]
value_state_map_list=[]

count_found_target=0
'''for i in range(20): # burn in
    c=burn_in_experience(  env_list,  experience_replay_buffer,  model,MaxStep=50)
    count_found_target+=c'''
    
logging.debug("Found Target {:d}/20".format(count_found_target))

start = datetime.now()

# Play a number of episodes and learn!
for i in tqdm(range(num_episodes)):
    total_t, episode_rewards[i], duration, num_steps[i], time_per_step, eps,myloss[i], summary_writer = play_train_episode(env_list,
                    total_t,i,experience_replay_buffer,model,gamma,batch_sz, eps,eps_change,eps_min,summary_writer,MaxStep=300)
    
    last_100_avg[i] = episode_rewards[max(0, i - 100):i + 1].mean()  
    last_100_avg_step[i] = num_steps[max(0, i - 100):i + 1].mean()  
        
    if i%500==0:
        logging.debug("Epi:", i,"Duration:", duration,"#steps:", num_steps[i],"Reward:", episode_rewards[i],\
        "Train time/step:", "%.3f" % time_per_step,"Avg Reward (Last 100):", "%.3f" % last_100_avg[i], "Eps:", "%.3f" % eps     )
        
        # create another test screnario
        # where we will start at other location (not the middle)
        temp_reward=[0]*nTest
        temp_step=[0]*nTest
        location_state_list_multiple=[0]*nTest
        for jj in range(nTest):
            
            #id_env=ii%2
            rand  = random.random()
            if rand > 0.5:
                newenv = env_list[0]
            else:
                newenv = env_list[1]

            temp_reward[jj], temp_step[jj], visit_map,location_state_list_multiple[jj],newenv, position_list_x, position_list_y  = \
            play_test_episode_from_location(newenv,model    ,eps,MaxStep=300)
                        
            if i==100000:
                print_trajectory_from_location(newenv,location_state_list_multiple[jj], idx=jj,
                   myxlabel="Gate A",myxrange=myxrange,myylabel="Gate B",myyrange=myyrange,strFolder="../plot/t4_small/",filetype="png")
                
                #export pickle                            
                strTest="../plot/t4_small/location_state_list_multiple_2d_{}.pickle".format(jj)
                pickle_out = open(strTest,"wb")
                pickle.dump(location_state_list_multiple, pickle_out)
                pickle_out.close()
            
        
        print("Optimal Policy on Test: 0:Up \t 1:Down \t 2:Left \t 3:Right \t 4:Down Right \t 5: Up Left")
        optimal_policy,val_pol,optimal_policy_2,val_pol2=final_policy_on_test(newenv, model,starting_loc_test[0])
        optimal_policy_list.append(optimal_policy)
        optimal_val_list.append(val_pol)
        optimal_policy_list_2.append(optimal_policy_2)
        optimal_val_list_2.append(val_pol2)

        print(optimal_policy)
        #print(optimal_policy_2)

        count_found_target=0
        for uu in range(15): # burn in
            c=burn_in_experience(  env_list,  experience_replay_buffer,  model,MaxStep=50)
            count_found_target+=c
            
        print("Burnin Exp: Found Target {:d}/15".format(count_found_target))
        
        value_state_map=get_value_state_on_test(model,newenv)
        value_state_map_list.append(value_state_map)
        
        
        episode_rewards_Test_B.append(temp_reward[0:10])
        episode_rewards_Test_SC.append(temp_reward[10:20])
        episode_rewards_Test_SD.append(temp_reward[20:30])
        
        num_steps_Test_B.append(temp_step[0:10])
        num_steps_Test_SC.append(temp_step[10:20])
        num_steps_Test_SD.append(temp_step[20:30])
        
        print("Barrier reward Test:",episode_rewards_Test_B[-1]," #step Test:",num_steps_Test_B[-1])      
        print("SC reward Test:",episode_rewards_Test_SC[-1]," #step Test:",num_steps_Test_SC[-1])
        print("SD reward Test:",episode_rewards_Test_SD[-1]," #step Test:",num_steps_Test_SD[-1])
        
            
saver = tf.train.Saver()

save_path = saver.save(sess,  "../logs/2d/save_models/2d_mean_std")

end = datetime.now()

time_taken = end - start
print("TIME TAKEN", time_taken)

print("TIME TAKEN (s)", time_taken.total_seconds())

fig=plt.figure()
plt.plot(np.log(myloss))
plt.title('Training Loss')
plt.xlabel('Episode')
plt.ylabel('Log of Loss')
plt.show()
fig.savefig("fig/b2/TrainingLoss64.pdf",box_inches="tight")

logloss=np.log(myloss)
ave_logloss=[np.mean(logloss[max(0,i-100):i+1]) for i in range(len(logloss))]
fig=plt.figure()
plt.plot(ave_logloss)
plt.title('Average Training Loss')
plt.xlabel('Episode')
plt.ylabel('Log of Loss')
plt.show()
fig.savefig("fig/b2/TrainingAverageLoss64.pdf",box_inches="tight")


fig=plt.figure()
plt.plot(episode_rewards)
plt.title('Training Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
fig.savefig("fig/b2/TrainingReward64.pdf",box_inches="tight")

fig=plt.figure()
plt.plot(last_100_avg)
plt.title('Training Average Reward')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
fig.savefig("fig/b2/TrainingReward_Ave64.pdf",box_inches="tight")


fig=plt.figure()
plt.plot(last_100_avg[2000:])
plt.title('Training Average Reward from 2000...')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
fig.savefig("fig/b2/TrainingReward_Ave2000_64.pdf",box_inches="tight")


fig=plt.figure()
plt.plot(num_steps)
plt.title('Number of Training Steps')
plt.xlabel('Episode')
plt.ylabel('Step')
plt.show()
fig.savefig("fig/b2/TrainingStep.pdf",box_inches="tight")

fig=plt.figure()
plt.plot(last_100_avg_step)
plt.title('Average of Training Step')
plt.xlabel('Episode')
plt.ylabel('Average Steps')
plt.show()
fig.savefig("fig/b2/TrainingAveStep64.pdf",box_inches="tight")

fig=plt.figure()
plt.plot(episode_rewards_Test_B)
plt.title('Average Reward Test Barrier')
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()
fig.savefig("fig/b2/TestAveReward64_B.pdf",box_inches="tight")

fig=plt.figure()
plt.plot(num_steps_Test_B)
plt.title('Number of Test Steps Barrier')
plt.xlabel('Episode')
plt.ylabel('Average Step')
plt.show()
fig.savefig("fig/b2/TestAveStep64_B.pdf",box_inches="tight")


output=[myloss,episode_rewards,last_100_avg,num_steps,last_100_avg_step
    ,episode_rewards_Test_B,num_steps_Test_B,episode_rewards_Test_SC,num_steps_Test_SC,
   episode_rewards_Test_SD,num_steps_Test_SD, optimal_policy_list, optimal_val_list,
   optimal_policy_list_2,optimal_val_list_2,value_state_map_list]
pickle.dump( output, open( "results/result_2d_T4_small.p", "wb" ) )



initial_gate_c5_c9=[ -570.,  -940]
window=350
myxrange=np.linspace(initial_gate_c5_c9[1]-window/2,initial_gate_c5_c9[1]+window/2,4).astype(int)
myyrange=np.linspace(initial_gate_c5_c9[0]-window/2,initial_gate_c5_c9[0]+window/2,4).astype(int)
myxrange=myxrange[::-1]
myyrange=myyrange[::-1]

'''plot_arrow_to_file(newenv,optimal_policy_list, optimal_val_list,
   optimal_policy_list_2,optimal_val_list_2,"action_plot",myxlabel="Gate A",
   myxrange=myxrange,myyrange=myyrange,myylabel="Gate B")
   '''
   
'''for ii,value in enumerate(value_state_map_list):
    fig=plt.figure()
    plt.imshow(value)
    plt.colorbar()
    plt.show()'''