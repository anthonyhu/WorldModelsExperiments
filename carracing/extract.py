'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import argparse

import gym

from model import make_model

parser = argparse.ArgumentParser()
parser.add_argument('--record', type=str, required=True, help='record name')
parser.add_argument('--full_model_path', type=str, default='', help='full model path [containing all weights]')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = ''

MAX_FRAMES = 1000 # max length of carracing
MAX_TRIALS = 250 # just use this to extract one trial.

render_mode = False # for debugging.

ROOT = '/data/cvfs/ah2029/datasets/gym/carracing/'

DIR_NAME = os.path.join(ROOT, args.record)
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

if args.full_model_path:
  model = make_model(vae_name='', load_model=False, load_full_model=True, full_model_path=args.full_model_path)
else:
  model = make_model(vae_name='', load_model=False)

total_frames = 0
model.make_env(render_mode=render_mode, full_episode=True)
for trial in range(MAX_TRIALS): # 200 trials per worker
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_obs = []
    recording_action = []

    np.random.seed(random_generated_int)
    model.env.seed(random_generated_int)

    # random policy
    if args.full_model_path:
      model.load_model(os.path.join(args.full_model_path, 'log', 'carracing.cma.5.12.best.json'))
    else:
      model.init_random_model_params(stdev=np.random.rand()*0.01)

    model.reset()
    obs = model.env.reset() # pixels

    for frame in range(MAX_FRAMES):
      if render_mode:
        model.env.render("human")
      else:
        model.env.render("rgb_array")

      recording_obs.append(obs)

      z, mu, logvar = model.encode_obs(obs)
      action = model.get_action(z)

      recording_action.append(action)
      obs, reward, done, info = model.env.step(action)

      if done:
        break

    total_frames += (frame+1)
    print("dead at", frame+1, "total recorded frames for this worker", total_frames)
    recording_obs = np.array(recording_obs, dtype=np.uint8)
    recording_action = np.array(recording_action, dtype=np.float16)
    np.savez_compressed(filename, obs=recording_obs, action=recording_action)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    model.env.close()
    model.make_env(render_mode=render_mode)
    continue
model.env.close()
