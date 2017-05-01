import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense, Reshape, merge, Lambda
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.regularizers import l2 as L2
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import numpy as np
import os
from PIL import Image
import glob 
import pickle
import sys
import msgpack
import msgpack_numpy as m; m.patch()
import numpy as np
import json
import re
import random

def loader(th = None):
  Xs = []
  Ys = [] 
  Rs = []
  idle_bwh = { }
  with open('dataset/bwh.txt') as f:
    for line in f:
      line      = line.strip()
      bwh       = list(map(lambda x:float(x)/100.0, re.findall("\d{1,}", line)))
      idle_name = line.split()[0]
      idle_bwh[idle_name] = bwh

  files = glob.glob('dataset/img/*')
  random.shuffle(files)
  if '--mini' in sys.argv:
    files = files[:500]

  trains = files[:35000]
  evals  = files[35001:]
  open("trains.pkl", "wb").write( pickle.dumps(trains) )
  open("evals.pkl", "wb").write( pickle.dumps(evals)  )
  for gi, name in enumerate(trains):
    if th is not None and gi > th:
      break
    idle_name = re.search(r"/.*?/(.*?)_", name).group(1)
    Rs.append( idle_name ) 
    bwh       = idle_bwh[idle_name]
    if len(bwh) != 3:
      continue
    print(bwh)
    y = bwh
    img = Image.open('{name}'.format(name=name))
    img = img.convert('RGB')
    arr   = np.array(img)
    Ys.append( y   )
    Xs.append( arr )
  Xs = np.array(Xs)
  return Ys, Xs, Rs

from keras.applications.resnet50 import ResNet50
def build_model():
  input_tensor = Input(shape=(224, 224, 3))
  resnet_model = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)

  dense  = Flatten()( \
             Dense(2048, activation='relu')( \
               BN()( \
	         resnet_model.layers[-1].output ) ) )
  result = Activation('linear')( \
	            Dense(3)(\
                 dense) )
  
  model = Model(inputs=resnet_model.input, outputs=result)
  for layer in model.layers[:80]: # default 179
    if 'BatchNormalization' in str(layer):
      ...
    else:
      layer.trainable = False
  model.compile(loss='mse', optimizer='adam')
  return model

weight_decay = 1e-4
add_tables = []
death_rate = 0.5
from keras.layers.merge import add
def _residual_drop(x, input_shape, output_shape, strides=(1, 1)):
  global add_tables
  #nb_filter = output_shape[0]
  nb_filter = 32
  print(nb_filter)
  print(x.shape)
  conv = Convolution2D(nb_filter, (3, 3), subsample=strides, padding="same", kernel_regularizer=L2(weight_decay))(x)
  conv = BN(axis=1)(conv)
  conv = Activation("relu")(conv)
  conv = Convolution2D(nb_filter, (3, 3), padding="same", kernel_regularizer=L2(weight_decay))(conv)
  conv = BN(axis=1)(conv)
  if strides[0] >= 2:
      x = AveragePooling2D(strides)(x)
  if (output_shape[0] - input_shape[0]) > 0:
      pad_shape = (1,
                   output_shape[0] - input_shape[0],
                   output_shape[1],
                   output_shape[2])
      padding = K.zeros(pad_shape)
      padding = K.repeat_elements(padding, K.shape(x)[0], axis=0)
      x = Lambda(lambda y: K.concatenate([y, padding], axis=1),
                 output_shape=output_shape)(x)
  _death_rate = K.variable(death_rate)
  scale = K.ones_like(conv) - _death_rate
  conv = Lambda(lambda c: K.in_test_phase(scale * c, c),
                output_shape=output_shape)(conv)
  print(x.shape)
  print(conv.shape)
  out = add([x, x])
  out = Activation("relu")(out)
  gate = K.variable(1, dtype="uint8")
  add_tables += [{"death_rate": _death_rate, "gate": gate}]
  return Lambda(lambda tensors: K.switch(gate, tensors[0], tensors[1]),
                output_shape=output_shape)([out, x])
def build_residual_drop():
  """ ANCHOR """
  inputs = Input(shape=(3, 224, 224))
  net = Convolution2D(16, (3, 3), padding="same", kernel_regularizer=L2(weight_decay))(inputs)
  net = BN(axis=1)(net)
  net = Activation("relu")(net)
  #for i in range(18):
  net = _residual_drop(net, input_shape=(16, 32, 32), output_shape=(16, 32, 32))
  net = _residual_drop(net, input_shape=(16, 32, 32), output_shape=(16, 32, 32))

#build_residual_drop()

def train():
  print('load lexical dataset...')
  Ys, Xs, Rs = loader()
  print('build model...')
  model = build_model()
  for i in range(300):
    model.fit(np.array(Xs), np.array(Ys), batch_size=16, nb_epoch=1 )
    if i%1 == 0:
      model.save('models/model%05d.model'%i)

def eval():
  model = build_model()

  mn_delta = {}
  for model_name in sorted(glob.glob('models.back/model*.model')):
    model = load_model(model_name) 
    target_size = (224,224)
    dir_path = "to_pred/*"
    max_size = len(glob.glob(dir_path))
    ps = []
    for i, name in enumerate(glob.glob(dir_path)):
      try:
        img = Image.open(name)
      except OSError as e:
        continue
      print(i, max_size, name.split('/')[-1])
      w, h = img.size
      if w > h :
        blank = Image.new('RGB', (w, w))
      if w <= h :
        blank = Image.new('RGB', (h, h))
      blank.paste(img, (0, 0) )
      blank = blank.resize( target_size )
      Xs = np.array([np.asanyarray(blank)])
      result = model.predict(Xs)
      ares   = map(lambda x:int(x*100.), result.tolist()[0] )
      tags   = ["B", "W", "H"]
      reals  = list(map(lambda x:int(x), re.findall(r"\d{2,}", name)))
      for t, a, r in zip(tags, ares, reals):
        print(model_name, t, a, r)
        p = abs(a - r)
        ps.append(p)
    print("delta", sum(ps))
    mn_delta[model_name] = sum(ps)
    open("mn_delta.pkl", "wb").write(pickle.dumps(mn_delta))

def pred():
  model = build_model()
  model = load_model(sorted(glob.glob('models.back/model00200.model'))[-1]) 
  target_size = (224,224)
  dir_path = "to_pred/*"
  max_size = len(glob.glob(dir_path))
  for i, name in enumerate(glob.glob(dir_path)):
    try:
      img = Image.open(name)
    except OSError as e:
      continue
    print(i, max_size, name.split('/')[-1])
    w, h = img.size
    if w > h :
      blank = Image.new('RGB', (w, w))
    if w <= h :
      blank = Image.new('RGB', (h, h))
    blank.paste(img, (0, 0) )
    blank = blank.resize( target_size )
    Xs = np.array([np.asanyarray(blank)])
    result = model.predict(Xs)
    ares   = map(lambda x:int(x*100.), result.tolist()[0] )
    tags   = ["B", "W", "H"]
    for t, a in zip(tags, ares):
      print(t, a)
     


if __name__ == '__main__':
  if '--train' in sys.argv:
    train()
  if '--eval' in sys.argv:
    eval()
  if '--pred' in sys.argv:
    pred()
