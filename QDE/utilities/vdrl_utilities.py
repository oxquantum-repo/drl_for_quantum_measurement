# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 12:15:01 2018

@author: vu
"""
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler

def generate_rand_point(env):
    point=[0]*len(env.img_dim)
    for dd in range(len(env.img_dim)):
        point[dd]=np.random.randint(0,env.img_dim[dd])
    return point

# so you can test different architectures
class HiddenLayer2:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True, zeros=False):
        if zeros:
            W = np.zeros((M1, M2), dtype=np.float32)
        else:
            W = tf.random_normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32)
        self.W = tf.Variable(W)
    
        self.use_bias = use_bias
        if use_bias:
          self.b = tf.Variable(np.zeros(M2).astype(np.float32))
    
        self.f = f
    
    def forward(self, X):
        if self.use_bias:
          a = tf.matmul(X, self.W) + self.b
        else:
          a = tf.matmul(X, self.W)
        return self.f(a)


# so you can test different architectures
class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True,zeros=False):
        
        initializer = tf.contrib.layers.xavier_initializer()

        if zeros:
            self.W = np.zeros((M1, M2), dtype=np.float32)
        else:
            #self.W = tf.random_normal(shape=(M1, M2)) * np.sqrt(2. / M1, dtype=np.float32)
            self.W = tf.Variable(initializer(shape=[M1,M2]))
      
        #self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        
        self.params = [self.W]

        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(initializer(shape=[M2]))
            self.params.append(self.b)

            #self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)

class FeatureTransformer:
  def __init__(self, env, n_components=500):
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    scaler = StandardScaler()
    scaler.fit(observation_examples)

    # Used to converte a state to a featurizes represenation.
    # We use RBF kernels with different variances to cover different parts of the space
    featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
            ])
    example_features = featurizer.fit_transform(scaler.transform(observation_examples))

    self.dimensions = example_features.shape[1]
    self.scaler = scaler
    self.featurizer = featurizer

  def transform(self, observations):
    # print "observations:", observations
    scaled = self.scaler.transform(observations)
    # assert(len(scaled.shape) == 2)
    return self.featurizer.transform(scaled)



def copy_two_graphs(sourceNetwork, destNetwork):
    
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, sourceNetwork)
    
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, destNetwork)

    op_holder = []
    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder