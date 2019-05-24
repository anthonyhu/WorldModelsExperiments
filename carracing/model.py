import numpy as np
import random

import os
import json
import sys

from sklearn.neighbors import NearestNeighbors

from env import make_env
import time

from vae.vae import ConvVAE
from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size

ROOT = '/data/cvfs/ah2029/datasets/gym/carracing/'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
vae_path = os.path.join(ROOT, 'tf_vae')
rnn_path = os.path.join(ROOT, 'tf_rnn')

render_mode = True

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH

def make_model(model_name='', load_model=True, load_full_model=False, full_model_path=''):
  # can be extended in the future.
  model = Model(model_name=model_name, load_model=load_model, load_full_model=load_full_model, full_model_path=full_model_path)
  return model

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Model:
  ''' simple one layer model for car racing '''
  def __init__(self, model_name='', load_model=True, load_full_model=False, full_model_path=''):
    self.model_name = model_name
    self.env_name = "carracing"
    self.vae = ConvVAE(batch_size=1, gpu_mode=False, is_training=False, reuse=True)

    self.rnn = MDNRNN(hps_sample, gpu_mode=False, reuse=True)

    if load_full_model:
      self.vae.load_json(os.path.join(full_model_path, 'vae.json'))
      self.rnn.load_json(os.path.join(full_model_path, 'rnn.json'))
    elif load_model:
      self.vae.load_json(os.path.join(vae_path, self.model_name + '_vae.json'))
      self.rnn.load_json(os.path.join(rnn_path, self.model_name + '_rnn.json'))

    self.state = rnn_init_state(self.rnn)
    self.rnn_mode = True

    self.input_size = rnn_output_size(EXP_MODE)
    self.z_size = 32

    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      self.hidden_size = 40
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, 3)
      self.bias_output = np.random.randn(3)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*3+3)
    else:
      self.weight = np.random.randn(self.input_size, 3)
      self.bias = np.random.randn(3)
      self.param_count = (self.input_size)*3+3

    self.render_mode = False

  def make_env(self, seed=-1, render_mode=False, full_episode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode, full_episode=full_episode)

  def reset(self):
    self.state = rnn_init_state(self.rnn)

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    mu, logvar = self.vae.encode_mu_logvar(result)
    mu = mu[0]
    logvar = logvar[0]
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
    return z, mu, logvar

  def get_action(self, z):
    h = rnn_output(self.state, z, EXP_MODE)

    '''
    action = np.dot(h, self.weight) + self.bias
    action[0] = np.tanh(action[0])
    action[1] = sigmoid(action[1])
    action[2] = clip(np.tanh(action[2]))
    '''
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
    else:
      action = np.tanh(np.dot(h, self.weight) + self.bias)
    
    action[1] = (action[1]+1.0) / 2.0
    action[2] = clip(action[2])

    self.state = rnn_next_state(self.rnn, z, action, self.state)

    return action

  def set_model_params(self, model_params):
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:3]
      self.weight_output = params_2[3:].reshape(self.hidden_size, 3)
    else:
      self.bias = np.array(model_params[:3])
      self.weight = np.array(model_params[3:]).reshape(self.input_size, 3)

  def load_model(self, filename):
    with open(os.path.join(ROOT, 'log', filename)) as f:
      data = json.load(f)
    print('loading file %s' % (os.path.join(ROOT, 'log', filename)))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)
    vae_params = self.vae.get_random_model_params(stdev=stdev)
    self.vae.set_model_params(vae_params)
    rnn_params = self.rnn.get_random_model_params(stdev=stdev)
    self.rnn.set_model_params(rnn_params)

def simulate(model, train_mode=False, render_mode=True, num_episode=5, novelty_search=False, novelty_mode='', seq_len=1000, seed=-1, max_len=-1):

  reward_list = []
  t_list = []
  bc_list = []

  max_episode_length = 1000
  recording_mode = False
  penalize_turning = False

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):

    model.reset()

    obs = model.env.reset()

    total_reward = 0.0

    random_generated_int = np.random.randint(2**31-1)

    #filename = "record/"+str(random_generated_int)+".npz"
    recording_mu = []
    recording_h = []
    recording_logvar = []
    recording_action = []
    recording_reward = [0]

    for t in range(max_episode_length):

      if render_mode:
        model.env.render("human")
      else:
        model.env.render('rgb_array')

      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)

      recording_mu.append(mu)
      # here we append the next state h
      recording_h.append(model.state.h[0])
      recording_logvar.append(logvar)
      recording_action.append(action)

      obs, reward, done, info = model.env.step(action)

      extra_reward = 0.0 # penalize for turning too frequently
      if train_mode and penalize_turning:
        extra_reward -= np.abs(action[0])/10.0
        reward += extra_reward

      recording_reward.append(reward)

      #if (render_mode):
      #  print("action", action, "step reward", reward)

      total_reward += reward

      if done:
        break

    # #for recording:
    # z, mu, logvar = model.encode_obs(obs)
    # action = model.get_action(z)
    # recording_mu.append(mu)
    # recording_logvar.append(logvar)
    # recording_action.append(action)
    #
    # recording_mu = np.array(recording_mu, dtype=np.float16)
    # recording_logvar = np.array(recording_logvar, dtype=np.float16)
    # recording_action = np.array(recording_action, dtype=np.float16)
    # recording_reward = np.array(recording_reward, dtype=np.float16)
    #
    # if not render_mode:
    #   if recording_mode:
    #     np.savez_compressed(filename, mu=recording_mu, logvar=recording_logvar, action=recording_action, reward=recording_reward)

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)
    if novelty_search:
      # Mean h vector
      if novelty_mode == 'h':
        bc_array = np.stack(recording_h, axis=0).mean(axis=0)  # shape 256
      elif novelty_mode == 'z':
        bc_array = np.stack(recording_mu, axis=0).mean(axis=0)  # shape 32

      if novelty_mode in ['h_concat', 'z_concat', 'a_concat']:
        if novelty_mode == 'h_concat':
          recording_bc = recording_h
        elif novelty_mode == 'z_concat':
          recording_bc = recording_mu
        elif novelty_mode == 'a_concat':
          recording_bc = recording_action

        # if the array is smaller repeat last element
        if len(recording_bc) < max_episode_length:
          recording_bc = recording_bc + [recording_bc[-1]] * (max_episode_length - len(recording_bc))

        if seq_len < max_episode_length:
          assert max_episode_length % seq_len == 0, 'Max episode length is not divisible by seq_len'
          step = max_episode_length // seq_len
          recording_bc = recording_bc[::step]
        bc_array = np.concatenate(recording_bc, axis=0)  # shape seq_len * d

      bc_list.append(bc_array)


  return reward_list, bc_list, t_list


def rank(scores):
    """
    Parameters
    ----------
        scores: 1D np.array

    Returns
    -------
        ranks: 1D np.array
            from 1 (highest value) to len(scores) (lowest value)
    """
    order = np.argsort(-scores)
    ranks = np.zeros_like(scores)
    ranks[order] = np.arange(len(scores)) + 1
    return ranks


def rank_transform(k, n):
  """ Rank transform, as defined in http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf"""
  denom = np.sum(np.maximum(0, np.log(n / 2 + 1) - np.log(np.arange(1, n + 1))))
  return max(0, np.log(n / 2 + 1) - np.log(k)) / denom - 1 / n


def score_to_rank_transform(scores):
  ranks = rank(scores)
  return np.array([rank_transform(k, len(ranks)) for k in ranks])


def compute_novelty(bc_array, archive, k=10):
  """
  Parameters
  ----------
    bc_array: np.array shape (population, bc_size)
    archive: list of array shape (bc_size,)
    k: int
      number of nearest neighbours

  Returns
  -------
    fitness: np.array shape (population)
      novelty rank transformed
  """
  population = len(bc_array)
  fitness = np.zeros(population)

  if archive:
    training_set = np.concatenate([bc_array, np.array(archive)], axis=0)
  else:
    training_set = bc_array

  if len(training_set) < k:
    print('Only {} examples for the nearest neighbours algorithm'.format(len(training_set)))
    k = len(training_set)
  neighbors = NearestNeighbors(k, metric='euclidean')

  neighbors.fit(training_set)

  for i in range(population):
    fitness[i] = neighbors.kneighbors(np.expand_dims(bc_array[i], 0))[0].mean()  # mean average distance to k-nearest neighbours

  # Rank transform
  fitness = score_to_rank_transform(fitness)
  return fitness

def main():

  assert len(sys.argv) > 1, 'python model.py render/norender path_to_mode.json [seed]'


  render_mode_string = str(sys.argv[1])
  if (render_mode_string == "render"):
    render_mode = True
  else:
    render_mode = False

  use_model = False
  if len(sys.argv) > 2:
    use_model = True
    filename = sys.argv[2]
    print("filename", filename)

  the_seed = np.random.randint(10000)
  if len(sys.argv) > 3:
    the_seed = int(sys.argv[3])
    print("seed", the_seed)

  if (use_model):
    model = make_model()
    print('model size', model.param_count)
    model.make_env(render_mode=render_mode)
    model.load_model(filename)
  else:
    model = make_model(load_model=False)
    print('model size', model.param_count)
    model.make_env(render_mode=render_mode)
    model.init_random_model_params(stdev=np.random.rand()*0.01)

  N_episode = 100
  if render_mode:
    N_episode = 1
  reward_list = []
  for i in range(N_episode):
    reward, steps_taken = simulate(model,
      train_mode=False, render_mode=render_mode, num_episode=1)
    if render_mode:
      print("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
    else:
      print(reward[0])
    reward_list.append(reward[0])
  if not render_mode:
    print("seed", the_seed, "average_reward", np.mean(reward_list), "stdev", np.std(reward_list))

if __name__ == "__main__":
  main()
