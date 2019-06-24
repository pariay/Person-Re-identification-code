# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:13:38 2019

@author: eep801

"""

''' This Python code is the implementation of the CVPR paper:
  De Cheng CVPR 2016- Person RE-identification by Multi-channel Parts-based 
  CNN with Improved Triplet Loss Function.
  
  There is no gurantee that this code will work '''
  
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv3D, Dense, MaxPooling3D, Flatten, Cropping1D, Lambda
import keras.backend as K

thresh1 = -1
thresh2 = 0.01
beta = 0.002

''' a: anchor_output, p:positive_output, n:negative_output'''
def imporved_loss_func(a, p, n, thresh1, thresh2, beta):
  dp = K.sum(K.square(a,p))
  dn = K.sum(K.square(a,n))
  s = K.sum(K.max(thresh1, dp-dn) + beta * (K.max(thresh2, dp)))
  return K.l2_normalize(s, axis = 1)
  
''' Model ---------------------------------------------------------------- '''  
def global_body(net):
  net.add(Conv3D(filters = 32, kernel_size = (7,7,3), strides = (3),\
                 kernel_initializer = 'uniform' , activation = 'relu', \
                 input_shape = (230,80,3,1)))
  return net

def full_body(net):
  net.add(MaxPooling3D(pool_size=(3, 3, 1)))
  net.add(Conv3D(filters = 32, kernel_size = (5,5,1), strides = (1),\
                 kernel_initializer = 'uniform' , activation = 'relu'))
  net.add(MaxPooling3D(pool_size=(3, 3, 1)))
  net.add(Flatten())
  net.add(Dense(units = 400, activation = 'relu')) 
  net.add(Lambda(lambda  x: K.l2_normalize(x,axis=1))) # normalise
  return net

def body_parts(divided_net):
  divided_net.add(Conv3D(filters = 32, kernel_size = (3,3,1),strides = (1), activation = 'relu'))
  divided_net.add(Conv3D(filters = 32, kernel_size = (3,3,1),strides = (1), activation = 'relu'))
  divided_net.add(Flatten())
  divided_net.add(Dense(units = 100, activation = 'relu'))
  divided_net.add(Lambda(lambda  x: K.l2_normalize(x,axis=1))) #normalise
  return divided_net

model = Sequential()
global_model = global_body(model)
new_global_model = global_model

full_body_model = full_body(global_model)

'''We may need to flip row and column - leave it for now'''
divided_net_1 = body_parts(new_global_model.add(Cropping1D(cropping = (0,18))))
divided_net_1 = body_parts(divided_net_1)

divided_net_2 = body_parts(new_global_model.add(Cropping1D(cropping = (19,37))))
divided_net_2 = body_parts(divided_net_2)

divided_net_3 = body_parts(new_global_model.add(Cropping1D(cropping = (38,56))))
divided_net_3 = body_parts(divided_net_3)

divided_net_4 = body_parts(new_global_model.add(Cropping1D(cropping = (57,75))))
divided_net_4 = body_parts(divided_net_4)

# Add CNNs on top of each others
id_model = Model(global_model.input, \
                 [divided_net_1, divided_net_2, full_body_model, divided_net_3, divided_net_4])

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))
  
''' Read triple images -----------------------------------------------------
---------------------------------------------------------------------------'''

from keras.preprocessing.image import ImageDataGenerator
import random 

datagen_train = ImageDataGenerator(rescale=1./255,zoom_range=[0.2, 0.08])
datagen_test = ImageDataGenerator(rescale=1./255)

def training_set_triplet():  
  # Make class names
  class_names = [] # anchor and positive
  for i in range(1,221):
    if i < 10:
      class_names.append(['person00'+str(i)])
    elif i >= 10 and i < 100:
      class_names.append(['person0'+str(i)])
    elif i >= 100:
      class_names.append(['person'+str(i)])  
  neg_class_names = []  # negative class   
  for c in class_names:
    rem_class = class_names
    rem_class.remove(c)
    neg_class_names.append(random.choice(rem_class))
  
  anchor_train = datagen_train.flow_from_directory('i-LIDS-VID/images/cam1',\
                         target_size = (250, 100),\
                         batch_size = 32,\
                         class_mode ='input',
                         classes = class_names)
    
  pos_train = datagen_train.flow_from_directory('i-LIDS-VID/images/cam2',\
                         target_size = (250, 100),
                         batch_size = 32,
                         class_mode ='input',
                         classes = class_names)
    
  neg_train = datagen_train.flow_from_directory('i-LIDS-VID/images/cam2',\
                          target_size = (250, 100),
                          batch_size = 32,
                          class_mode ='input',
                          classes = neg_class_names )
    
  while True:
    X1i = anchor_train.next() # Xli[0] image, Xli[1] lable
    X2i = pos_train.next()
    X3i = neg_train.next()

    yield [X1i[0], X2i[0], X3i[0]], X1i[1]

''' Triple models ---------------------------------------------------------
---------------------------------------------------------------------------'''
input_shape = (230,80,3)
input_anchor = keras.engine.input_layer.Input(shape = input_shape)
input_pos = keras.engine.input_layer.Input(shape = input_shape)
input_neg = keras.engine.input_layer.Input(shape = input_shape)

anchor_model = id_model(input_shape)
pos_model = id_model(input_shape)
neg_model = id_model(input_shape)

stacked_model = Model([input_anchor,input_pos, input_neg],\
                      [anchor_model,pos_model,neg_model])

'''Compile the model on triple images '''
stacked_model.compile(optimizer = 'adam', \
                      loss = imporved_loss_func(anchor_model,pos_model, neg_model, thresh1, thresh2, beta))
stacked_model.fit_generator(generator = training_set_triplet(),steps_per_epoch = 8000, epochs = 25)
