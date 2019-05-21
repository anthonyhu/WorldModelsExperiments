'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import os
import argparse

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae.vae import ConvVAE, reset_graph

ROOT = '/data/cvfs/ah2029/datasets/gym/carracing/'
NUM_DATA = 2500
NUM_FRAMES = 1000

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, required=True, help='gpu to use')
parser.add_argument('--beta', type=float, required=True, help='beta coefficient')
parser.add_argument('--record', type=str, required=True, help='record directory')
parser.add_argument('--name', type=str, required=True, help='model name prefix')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Hyperparameters for ConvVAE
z_size=32
batch_size=100
learning_rate=0.0001
kl_tolerance=0.5
beta = args.beta # beta-VAE coefficient

# Parameters for training
NUM_EPOCH = 10
DATA_DIR = os.path.join(ROOT, args.record)


model_save_path = os.path.join(ROOT, 'tf_vae')
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

# def count_length_of_filelist(filelist):
#   # although this is inefficient, much faster than doing np.concatenate([giant list of blobs])..
#   N = len(filelist)
#   total_length = 0
#   for i in range(N):
#     filename = filelist[i]
#     raw_data = np.load(os.path.join("record", filename))['obs']
#     l = len(raw_data)
#     total_length += l
#     if (i % 1000 == 0):
#       print("loading file", i)
#   return  total_length

def create_dataset(filelist, N=10000, M=1000): # N is 10000 episodes, M is number of timesteps
  data = np.zeros((M*N, 64, 64, 3), dtype=np.uint8)
  idx = 0
  for i in range(N):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))['obs'][:M]
    l = len(raw_data)
    if (idx+l) > (M*N):
      data = data[0:idx]
      print('premature break')
      break
    data[idx:idx+l] = raw_data
    idx += l
    if ((i+1) % 100 == 0):
      print('number of frames', l)
      print("loading file", i+1)
  return data

# load dataset from record/*. only use first 10K, sorted by filename.
filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[:NUM_DATA]
#print("check total number of images:", count_length_of_filelist(filelist))
dataset = create_dataset(filelist, N=NUM_DATA, M=NUM_FRAMES)

# split into batches:
total_length = len(dataset)
num_batches = int(np.floor(total_length/batch_size))
print("num_batches", num_batches)

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              beta=beta,
              is_training=True,
              reuse=False,
              gpu_mode=True)
print('Training beta-VAE with beta={:.1f}'.format(beta))

# train loop:
print("train", "step", "loss", "recon_loss", "kl_loss")
for epoch in range(NUM_EPOCH):
  np.random.shuffle(dataset)
  for idx in range(num_batches):
    batch = dataset[idx*batch_size:(idx+1)*batch_size]

    obs = batch.astype(np.float)/255.0

    feed = {vae.x: obs,}

    (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
      vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
    ], feed)
  
    if ((train_step+1) % 500 == 0):
      print("step", (train_step+1), train_loss, r_loss, kl_loss)
    if ((train_step+1) % 5000 == 0):
      vae.save_json(os.path.join(model_save_path, args.name + '_vae.json'.format(beta)))

# finished, final model:
vae.save_json(os.path.join(model_save_path, args.name + '_vae.json'.format(beta)))
