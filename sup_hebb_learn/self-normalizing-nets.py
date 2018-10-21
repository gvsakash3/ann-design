#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (15.0, 8.0)
plt.rcParams['savefig.dpi'] = 100


# In[2]:



# create 1000 data points from a gaussian distribution
data = np.random.randn(1000, 500)

# 10 hidden layers
hidden_layer_sizes = [500]*10

# define nonlinearities
nonlinearities = ['selu']*len(hidden_layer_sizes)


# In[3]:



def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*np.where(x>=0.0, x, alpha*np.exp(x)-alpha)

def relu(x):
    return np.maximum(0, x)

def elu(x):
    return np.where(x>=0.0, x, np.exp(x)-1)

act = {'relu': lambda x: relu(x), 
       'elu': lambda x: elu(x), 
       'selu': lambda x: selu(x)}


# In[4]:


num_layers = len(hidden_layer_sizes)

stats = {}
for i in range(num_layers):
    # input layer
    X = data if i == 0 else stats[i-1]
    
    # initialize weights
    fan_in, fan_out = X.shape[1], hidden_layer_sizes[i]
    if nonlinearities[i] == 'selu':
        W = np.random.normal(size=(fan_in, fan_out), scale=np.sqrt(1/fan_in))
    else: # he et. al initialization
        W = np.random.randn(fan_in, fan_out) / np.sqrt(fan_in / 2)
    
    # matrix multiply with nonlinearity
    H = np.dot(X, W)
    H = act[nonlinearities[i]](H)
    
    # store result of layer
    stats[i] = H


# In[5]:


# mean and std for each layer
layer_means = [np.mean(s) for i,s in stats.items()]
layer_stds = [np.std(s) for i,s in stats.items()]

print('Input layer has mean {} and std {}'.format(np.mean(data), np.std(data)))
for i,s in stats.items():
    print('Hidden layer {} has mean {} and std {}'.format(i+1, layer_means[i], layer_stds[i]))


# In[6]:


# plot means and stds
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.plot(list(stats.keys()), layer_means, 'ob-')
ax1.set_title('Layer Means')
ax2.plot(list(stats.keys()), layer_stds, 'or-')
ax2.set_title('Layer Stds')
plt.show()


# In[7]:


# plot the raw distribution
plt.figure()
for i,s in stats.items():
    plt.subplot(1, num_layers, i+1)
    plt.hist(s.ravel(), 30, range=(-5, 5))
plt.suptitle('SELU Activation Distribution')
plt.savefig('selu.eps', format='eps', dpi=200)


# In[ ]:




