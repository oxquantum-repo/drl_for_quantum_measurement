# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:14:05 2019

@author: Vu
"""

import sys

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../environments')
sys.path.append('../utilities')
sys.path.append('../testing_code')
sys.path.append('../environments')
sys.path.append('../utilities')
from utility_plot_arrow import plot_arrow_to_file

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.utils import shuffle
#from scipy.misc import imresize
#from gym import wrappers
import random
from tqdm import tqdm
from prioritized_experience_replay import Memory
from vdrl_utilities import HiddenLayer




class Dueling_DQN_PER_2D:
    def __init__(self, D,K, hidden_layer_sizes, gamma,batch_sz=32,
          memory_size=50000,min_y=0,max_y=1,max_experiences=50000, min_experiences=5000,
          N_CHANEL=9,IM_SIZE=2,lr=1e-6,scope="DDQN"):
                 
        with tf.variable_scope(scope):

            self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions') # None unspecified as depends on batch size
            #self.ISWeights_=tf.placeholder(tf.float32, shape=(None,1), name='ISW')
            self.X = tf.placeholder(tf.float32, shape=(None, N_CHANEL*IM_SIZE), name='X') # Input state
            self.G = tf.placeholder(tf.float32, shape=(None,), name='G') # Discounted Reward
            self.neighborMap = tf.placeholder(tf.float32, shape=(None,K), name='neighborMap') # Returns number of visits in neighbouring locations in 6 locations (ignores 2 where actions cannot be taken) 
            self.memory_size=memory_size
            self.memory= Memory(memory_size)
    
            self.min_y=min_y
            self.max_y=max_y
            
            self.K=K
            self.D=D
            
            Z=self.X    
            tf.summary.histogram("X", self.X) # plots summary tensorboard 

            #print(Z.get_shape())
            # add visited information here ?
            # neighborMap is 4-dim [top-right-down-left] or 8-dim []
            #Z=tf.concat([Z, self.neighborMap], 1)
            #print(Z.get_shape())
    
            for ii,M2 in enumerate(hidden_layer_sizes):
                Z = tf.contrib.layers.fully_connected(Z, M2) # Defines fully connected hidden layers specified in run_training
                #Z=tf.concat([Z, self.neighborMap], 1)

                strName="fully_connected_Z_{:d}".format(ii)
                tf.summary.histogram(strName, Z) # Tensorboard

            #M3=hidden_layer_sizes[1]
            
            #print(Z.get_shape())
            tf.summary.histogram("neighborMap", self.neighborMap)
            
            # neighborMap is 4-dim [top-right-down-left] or 8-dim []
            #Z=tf.concat([Z, self.neighborMap], 1)
            #tf.summary.histogram("concat_Z", Z)

            #print(Z.get_shape())
    
            #Z = tf.contrib.layers.fully_connected(Z, M2)
            
            #tf.summary.histogram("fully_connected_Z2", Z)

                
            # calculate output and cost
            #Z=self.X
            #for layer in self.layers:
            #    Z=layer.forward(Z)
                
                
            ## Here we separate into two streams
            # The one that calculate V(s) - For Duelling
            self.value_fc = tf.layers.dense(inputs = Z,
                units = 64,  activation = tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="value_fc")
    
            tf.summary.histogram("value_fc",self.value_fc)

            self.value = tf.layers.dense(inputs = self.value_fc,
                units = 1, activation = None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="value")
            tf.summary.histogram("value",self.value)

            
            # add the visited map here
            
            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs = Z,
                units = 64,  activation = tf.nn.elu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="advantage_fc")
            
            tf.summary.histogram("advantage_fc",self.advantage_fc)

            self.advantage_fc=tf.concat([self.advantage_fc, self.neighborMap], 1) # Prevents revisiting old location
            
            tf.summary.histogram("advantage_fc_concat",self.advantage_fc)

    
            self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                units = K, activation = None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name="advantages")
                
            tf.summary.histogram("advantage",self.advantage)

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage, 
                               tf.reduce_mean(self.advantage, axis=1, keepdims=True)) # Nework output, K dimensions. 
    
            tf.summary.histogram("output",self.output)

            # Q is our predicted Q value.
            #self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)
    
            self.predict_op = self.output
    
    # Decides selected action (0-6) -> One dimension, represent the expected reward of current action from neural netowrk
            selected_action_values = tf.reduce_sum(
                self.output * tf.one_hot(self.actions, K),
                reduction_indices=[1])
    
            self.loss = tf.reduce_sum(tf.square(self.G - selected_action_values))
            #self.loss = tf.reduce_sum(self.ISWeights_*tf.square(self.G - selected_action_values))
            tf.summary.histogram("G", self.G)
            tf.summary.scalar("loss", self.loss)

    
            # The loss is the difference between our predicted Q_values and the Q_target
            # Sum(Qtarget - Q)^2
            #self.loss = tf.reduce_mean(tf.square(self.G - self.Q))
    
            # The loss is modified because of PER 
            self.absolute_errors = tf.abs(selected_action_values - self.G)# for updating Sumtree
    
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
            #self.train_op = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6).minimize(self.loss)
    
            # self.train_op = tf.train.AdagradOptimizer(1e-2).minimize(cost)
            # self.train_op = tf.train.MomentumOptimizer(1e-3, momentum=0.9).minimize(cost)
            # self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
    
            # create replay memory
            #self.experience={'s':[],'a':[],'r':[],'s2':[],'done':[]}
            #self.max_experiences=max_experiences
            #self.min_experiences=min_experiences
            self.batch_sz=batch_sz
            self.gamma=gamma
            self.count_exp=0
            
            self.merge = tf.summary.merge_all()
            
            # create replay memory, s state a action r reward s2 next state based on action (18 dimension) s2_loc location of next state, done - hit bias triangle 
            self.experience={'s':[],'a':[],'r':[],'s2':[],'s2_loc':[]
            ,'neighborMaps':[],'neighborMaps_next':[],'env_id':[],'done':[]}
            self.max_experiences=max_experiences
            self.min_experiences=min_experiences
            self.batch_sz=batch_sz
            self.gamma=gamma
    
            
    def set_session(self, session):
        self.session = session
        
    def save_model(self,name):
        self.session.save(name)
    
    def get_value_state(self, X, neighborMap):
        X = np.atleast_2d(X)
        
        if len(X.shape)==3:
            X=np.squeeze(X)

        
        neighborMap=np.atleast_2d(neighborMap)
        return self.session.run(self.value, feed_dict={self.X: X, self.neighborMap:neighborMap})
            
    # 
    def predict(self, X, neighborMap):
        X = np.atleast_2d(X)
        
        if len(X.shape)==3:
            X=np.squeeze(X)

        
        neighborMap=np.atleast_2d(neighborMap)
        return self.session.run(self.predict_op, feed_dict={self.X: X, self.neighborMap:neighborMap})
            
    def fit(self,states,targets,actions):
        # call optimizer
        actions=np.atleast_1d(actions)
        targets=np.atleast_1d(targets)

        #scale target
        #targets=(targets-self.min_y)/(self.max_y-self.min_y)

        states=np.atleast_2d(states)

        self.session.run(self.train_op,feed_dict={self.X: states,self.G: targets,self.actions: actions}         )
        
   
    def fit_prioritized_exp_replay(self,env_list):
        if self.count_exp < self.memory_size:
        # don't do anything if we don't have enough experience
            return None,None

        # Obtain random mini-batch from memory
        tree_idx, batch, ISWeights_mb = self.memory.sample(self.batch_sz)
        
        # experience replay
        #idx=np.random.choice(len(self.experience['s']), 
                     #size=self.batch_sz, replace=False)
            
        """
        states=[self.experience['s'][i] for i in idx]
        #states_loc=[self.experience['s_loc'][i] for i in idx]
        actions=[self.experience['a'][i] for i in idx]
        rewards=[self.experience['r'][i] for i in idx]
        next_states=[self.experience['s2'][i] for i in idx]
        next_states_loc=[self.experience['s2_loc'][i] for i in idx]
        neighborMaps=[self.experience['neighborMaps'][i] for i in idx]
        #neighborMaps_next=[self.experience['neighborMaps_next'][i]/20 for i in idx]
        env_id=[self.experience['env_id'][i] for i in idx]
        dones=[self.experience['done'][i] for i in idx]
        """

        states=[each[0][0] for each in batch]
        states_loc=[each[0][1] for each in batch]
        actions=[each[0][2] for each in batch]
        rewards=[each[0][3] for each in batch]
        next_states=[each[0][4] for each in batch]
        next_states_loc=[each[0][5] for each in batch]
        dones=[each[0][6] for each in batch]
        neighborMaps=[each[0][7] for each in batch]
        #neighborMaps_next=[each[0][8] for each in batch]
        env_id=[each[0][8] for each in batch]
        

        #neighborMaps=[env.get_neighborMap(loc) for loc in states_loc]
        neighborMaps_next=[env_list[myid].get_neighborMap(loc) for loc,myid in zip(next_states_loc,env_id)]
        
        next_actions = np.argmax(self.predict(next_states,neighborMaps_next), axis=1)
        targetQ=self.predict(next_states,neighborMaps_next)
        next_Q=[q[a] for a,q in zip(next_actions,targetQ)]

        targets = [r + self.gamma*next_q 
        if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        #scale target
        #targets=[ (val-self.min_y)/(self.max_y-self.min_y) for val in targets]

        #print(targets)
        summary,loss,_,absolute_errors=self.session.run( [self.merge,self.loss,self.train_op,self.absolute_errors], 
                     feed_dict={self.X:states, self.G:targets, self.actions: actions,
                            self.neighborMap:neighborMaps })

        # Update priority
        #print(absolute_errors)

        self.memory.batch_update(tree_idx, absolute_errors)
        
        return summary,loss
    

    def add_experience(self, s,loc,a,r,s2,loc_s2,done,count_neighbor,neighborMaps_next,env_id):
        """
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
            self.experience['s2_loc'].pop(0)
            self.experience['neighborMaps'].pop(0)
            self.experience['neighborMaps_next'].pop(0)
            self.experience['env_id'].pop(0)
            
        # Add experience to memory
        if type(r) is list: # adding a list of them
            self.experience['s']+=s
            self.experience['a']+=a
            self.experience['r']+=r
            self.experience['s2']+=s2
            self.experience['done']+=done
            self.experience['s2_loc']+=loc_s2
            self.experience['neighborMaps']+=count_neighbor
            self.experience['neighborMaps_next']+=neighborMaps_next
            self.experience['env_id']+=env_id
        else:
            self.experience['s'].append(s)
            self.experience['a'].append(a)
            self.experience['r'].append(r)
            self.experience['s2'].append(s2)
            self.experience['done'].append(done)
            self.experience['s2_loc'].append(loc_s2)
            self.experience['neighborMaps'].append(count_neighbor)
            self.experience['neighborMaps_next'].append(neighborMaps_next)
            self.experience['env_id'].append(env_id)
        """
    
        if type(r) is list:
            for ii in range(len(s)):
                experience = s[ii], loc[ii], a[ii], r[ii], s2[ii], loc_s2[ii], done[ii],\
                    count_neighbor[ii], env_id[ii]
                self.memory.store(experience)

    def sample_action(self, env, x, loc_x,eps,isNoOverlapping=False,is2Action=False):
        possible_actions=env.possible_actions_from_location(loc_x)
        if isNoOverlapping==True and possible_actions == []: # we got stuck, let jump out
            # find the empty place in the visit_map
            # then find the nearest location
            [idxRow,idxCol]=np.where(env.visit_map==0)
            #idxRow=idxRow.astype(float)
            #idxCol=idxCol.astype(float)
            coordinate=np.array(list(zip(idxRow,idxCol)))
            coordinate=np.reshape(coordinate,(-1,2))
            loc_x=np.asarray(loc_x).reshape((1,2))
            
            dist=[np.linalg.norm(loc_x[0]-coordinate[ii]) for ii in range(coordinate.shape[0])]
            
            idxNearest=np.argmin(dist)
            return coordinate[idxNearest] # we return a new coordinate of the new action
                        
        if is2Action==True:
            #print("is2Action==True")
            neighborMap=env.get_neighborMap(loc_x)
            
            val=self.predict(x,neighborMap)

            val_selected=val[0][possible_actions]
            #print("here",val,val_selected)
            #return np.argmax(self.predict([x])[0])
            
            if len(possible_actions)==0:
                return -1,-1,-1,-1

            idx=np.argmax(val_selected)            

            if len(possible_actions)==1:
                return possible_actions[idx],val_selected[idx],\
                    possible_actions[idx],val_selected[idx]
           
            else:
                idx2Best=np.argsort(val_selected)[-2]
                return possible_actions[idx],val_selected[idx],possible_actions[idx2Best],\
                        val_selected[idx2Best]

        
        # here perform epsilon greedy
        if np.random.random() < eps:
            #print('np.random.random() < eps and thus random action chosen')
            idx=np.random.choice( len(possible_actions))
            return possible_actions[idx] 
        else:
            
            #print('PREDICTED ACTION CHOSEN')
            #X = np.atleast_2d(x)
            neighborMap=env.get_neighborMap(loc_x)
            
            val=self.predict(x,neighborMap)
            val_selected=val[0][possible_actions]
            #print("val_selected",val_selected)
            #print("here",val,val_selected)
            #return np.argmax(self.predict([x])[0])
            
            idx=np.argmax(val_selected)
            #print('possible_actions[idx]',possible_actions[idx]) '''
            
            return possible_actions[idx]


    def sample_random_action(self, env, x, loc_x,isNoOverlapping=False):
        
        # check all posible actions
        #possible_actions=env.possible_actions()
        possible_actions=env.possible_actions_from_location(loc_x)
        
        if isNoOverlapping==True and possible_actions == []: # we got stuck, let jump out
            # find the empty place in the visit_map
            # then find the nearest location
            [idxRow,idxCol]=np.where(env.visit_map==0)
            #idxRow=idxRow.astype(float)
            #idxCol=idxCol.astype(float)
            coordinate=np.array(list(zip(idxRow,idxCol)))
            coordinate=np.reshape(coordinate,(-1,2))
            loc_x=np.asarray(loc_x).reshape((1,2))
            
            dist=[np.linalg.norm(loc_x[0]-coordinate[ii]) for ii in range(coordinate.shape[0])]
            
            idxNearest=np.argmin(dist)
            return coordinate[idxNearest] # we return a new coordinate of the new action
            
     
        idx=np.random.choice( len(possible_actions))
        return possible_actions[idx] 
       
        