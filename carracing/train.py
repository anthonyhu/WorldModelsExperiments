# training settings
''' roboschool envs available
robo_pendulum
robo_double_pendulum
robo_reacher
robo_flagrun

robo_ant
robo_reacher
robo_hopper
robo_walker
robo_humanoid
'''

from mpi4py import MPI
import numpy as np
import os
import json
import subprocess
import sys

from collections import deque

from model import make_model, simulate, compute_novelty
from es import CMAES, SimpleGA, OpenES, PEPG
import argparse
import time

ROOT = '/data/cvfs/ah2029/datasets/gym/carracing/'
### ES related code
num_episode = 1
eval_steps = 25 # evaluate every N_eval steps
retrain_mode = True
cap_time_mode = True

num_worker = 8
num_worker_trial = 1

population = num_worker * num_worker_trial

gamename = 'carracing'
optimizer = 'pepg'
antithetic = True
batch_mode = 'mean'

# seed for reproducibility
seed_start = 0

### name of the file (can override):
filebase = None

model = None
num_params = -1

es = None

### MPI related code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

H_SIZE = 256
Z_SIZE = 32
MAX_ARCHIVE_ELEMENTS = 500
NOVELTY_THRESHOLD = 0
PROBABILITY_ADD = 0.02
N_NEAREST_NEIGHBOURS = 10
FIXED_MAP = True
FIXED_SEED = 30
###

def initialize_settings(sigma_init=0.1, sigma_decay=0.9999):
  global population, filebase, game, model, num_params, es, PRECISION, SOLUTION_PACKET_SIZE, RESULT_PACKET_SIZE, model_name, novelty_search, unique_id, novelty_mode, BC_SIZE
  population = num_worker * num_worker_trial
  os.makedirs(os.path.join(ROOT, 'log'), exist_ok=True)
  filebase = os.path.join(ROOT, 'log', gamename+'.'+optimizer+'.'+ model_name + '.' + str(num_episode)+'.'+str(population)) + '.' + unique_id
  if novelty_search:
    filebase = filebase + '.novelty'
    if novelty_mode == 'h':
      BC_SIZE = H_SIZE
    elif novelty_mode == 'z':
      BC_SIZE = Z_SIZE
    elif novelty_mode =='h_concat':
      BC_SIZE = 1000 * H_SIZE
      #NOVELTY_THRESHOLD = 180
    elif novelty_mode == 'z_concat':
      BC_SIZE = 1000 * Z_SIZE
    else:
      raise ValueError('Not recognised novelty_mode: {}'.format(novelty_mode))

  model = make_model(model_name, load_model=True)
  num_params = model.param_count
  print("size of model", num_params)
  PRECISION = 10000
  SOLUTION_PACKET_SIZE = (5 + num_params) * num_worker_trial
  RESULT_PACKET_SIZE = (4 + BC_SIZE) * num_worker_trial

  if optimizer == 'ses':
    ses = PEPG(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_alpha=0.2,
      sigma_limit=0.02,
      elite_ratio=0.1,
      weight_decay=0.005,
      popsize=population)
    es = ses
  elif optimizer == 'ga':
    ga = SimpleGA(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_limit=0.02,
      elite_ratio=0.1,
      weight_decay=0.005,
      popsize=population)
    es = ga
  elif optimizer == 'cma':
    cma = CMAES(num_params,
      sigma_init=sigma_init,
      popsize=population)
    es = cma
  elif optimizer == 'pepg':
    pepg = PEPG(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_alpha=0.20,
      sigma_limit=0.02,
      learning_rate=0.01,
      learning_rate_decay=1.0,
      learning_rate_limit=0.01,
      weight_decay=0.005,
      popsize=population)
    es = pepg
  else:
    oes = OpenES(num_params,
      sigma_init=sigma_init,
      sigma_decay=sigma_decay,
      sigma_limit=0.02,
      learning_rate=0.01,
      learning_rate_decay=1.0,
      learning_rate_limit=0.01,
      antithetic=antithetic,
      weight_decay=0.005,
      popsize=population)
    es = oes
###

def sprint(*args):
  print(args) # if python3, can do print(*args)
  sys.stdout.flush()

class OldSeeder:
  def __init__(self, init_seed=0):
    self._seed = init_seed
  def next_seed(self):
    result = self._seed
    self._seed += 1
    return result
  def next_batch(self, batch_size):
    result = np.arange(self._seed, self._seed+batch_size).tolist()
    self._seed += batch_size
    return result

class Seeder:
  def __init__(self, init_seed=0):
    np.random.seed(init_seed)
    self.limit = np.int32(2**31-1)
  def next_seed(self):
    result = np.random.randint(self.limit)
    return result
  def next_batch(self, batch_size):
    result = np.random.randint(self.limit, size=batch_size).tolist()
    return result

def encode_solution_packets(seeds, solutions, train_mode=1, max_len=-1):
  n = len(seeds)
  result = []
  worker_num = 0
  for i in range(n):
    worker_num = int(i / num_worker_trial) + 1
    result.append([worker_num, i, seeds[i], train_mode, max_len])
    result.append(np.round(np.array(solutions[i])*PRECISION,0))
  result = np.concatenate(result).astype(np.int32)
  result = np.split(result, num_worker)
  return result

def decode_solution_packet(packet):
  packets = np.split(packet, num_worker_trial)
  result = []
  for p in packets:
    result.append([p[0], p[1], p[2], p[3], p[4], p[5:].astype(np.float)/PRECISION])
  return result

def encode_result_packet(results):
  r = np.array(results)
  print('encode_results_packets receives array of shape {}'.format(r.shape))
  r[:, 2:] *= PRECISION
  return r.flatten().astype(np.int32)

def decode_result_packet(packet):
  r = packet.reshape(num_worker_trial, 4 + BC_SIZE)
  workers = r[:, 0].tolist()
  jobs = r[:, 1].tolist()
  times = r[:, 2].astype(np.float)/PRECISION
  times = times.tolist()
  fits = r[:, 3].astype(np.float)/PRECISION
  fits = fits.tolist()
  behaviour_chars = r[:, 4:].astype(np.float)/PRECISION
  result = []
  n = len(jobs)
  for i in range(n):
    result.append([workers[i], jobs[i], times[i], fits[i], behaviour_chars[i]])
  return result

def worker(weights, seed, train_mode_int=1, max_len=-1):
  behaviour_char = np.zeros(BC_SIZE)

  train_mode = (train_mode_int == 1)
  model.set_model_params(weights)
  reward_list, bc_list, t_list = simulate(model,
    train_mode=train_mode, render_mode=False, num_episode=num_episode, novelty_search=novelty_search, novelty_mode=novelty_mode, seed=seed, max_len=max_len)

  if novelty_search:
    # bc_list shape (n_episodes, bc_size)
    behaviour_char = np.array(bc_list).mean(axis=0)
  if batch_mode == 'min':
    reward = np.min(reward_list)
  else:
    reward = np.mean(reward_list)
  t = np.mean(t_list)
  return reward, behaviour_char, t  # bc_list shape (bc_size,)

def slave():
  model.make_env()
  packet = np.empty(SOLUTION_PACKET_SIZE, dtype=np.int32)
  while 1:
    comm.Recv(packet, source=0)
    assert len(packet) == SOLUTION_PACKET_SIZE, 'packet size: {}, expected {}'.format(len(packet), SOLUTION_PACKET_SIZE)
    solutions = decode_solution_packet(packet)
    results = []
    for solution in solutions:
      worker_id, jobidx, seed, train_mode, max_len, weights = solution
      assert (train_mode == 1 or train_mode == 0), str(train_mode)
      worker_id = int(worker_id)
      possible_error = "work_id = " + str(worker_id) + " rank = " + str(rank)
      assert worker_id == rank, possible_error
      jobidx = int(jobidx)
      seed = int(seed)
      fitness, behaviour_char, timesteps = worker(weights, seed, train_mode, max_len)
      results.append(np.concatenate([[worker_id, jobidx, timesteps, fitness], behaviour_char]))
    print('results after worker of len {} and element of shape {}'.format(len(results), results[0].shape))
    result_packet = encode_result_packet(results)
    assert len(result_packet) == RESULT_PACKET_SIZE
    comm.Send(result_packet, dest=0)

def send_packets_to_slaves(packet_list):
  num_worker = comm.Get_size()
  assert len(packet_list) == num_worker-1
  for i in range(1, num_worker):
    packet = packet_list[i-1]
    assert len(packet) == SOLUTION_PACKET_SIZE, 'packet size: {}, expected {}'.format(len(packet), SOLUTION_PACKET_SIZE)
    comm.Send(packet, dest=i)

def receive_packets_from_slaves():
  """
  Returns
  -------
    reward_list_total: np.array (population, 2 + bc_size)
    columns: first index is the timetamp, then reward, and the remaining ones are the bc vector

  """
  result_packet = np.empty(RESULT_PACKET_SIZE, dtype=np.int32)

  reward_list_total = np.zeros((population, 2 + BC_SIZE))

  check_results = np.ones(population, dtype=np.int)
  for i in range(1, num_worker+1):
    comm.Recv(result_packet, source=i)
    results = decode_result_packet(result_packet)
    for result in results:
      worker_id = int(result[0])
      possible_error = "work_id = " + str(worker_id) + " source = " + str(i)
      assert worker_id == i, possible_error
      idx = int(result[1])
      reward_list_total[idx, 0] = result[2]
      reward_list_total[idx, 1] = result[3]
      reward_list_total[idx, 2:] = result[4]
      check_results[idx] = 0

  check_sum = check_results.sum()
  assert check_sum == 0, check_sum
  return reward_list_total

def evaluate_batch(model_params, max_len=-1):
  # duplicate model_params
  solutions = []
  for i in range(es.popsize):
    solutions.append(np.copy(model_params))

  seeds = np.arange(es.popsize)
  if FIXED_SEED:
    seeds = [FIXED_SEED] * es.popsize

  packet_list = encode_solution_packets(seeds, solutions, train_mode=0, max_len=max_len)

  send_packets_to_slaves(packet_list)
  reward_list_total = receive_packets_from_slaves()

  reward_list = reward_list_total[:, 1]
  return np.mean(reward_list)

def master():

  start_time = int(time.time())
  sprint("training", gamename)
  sprint("population", es.popsize)
  sprint("num_worker", num_worker)
  sprint("num_worker_trial", num_worker_trial)
  if novelty_search:
    sprint('Novelty search with bc: {}'.format(novelty_mode))
  sys.stdout.flush()

  seeder = Seeder(seed_start)

  filename = filebase+'.json'
  filename_log = filebase+'.log.json'
  filename_hist = filebase+'.hist.json'
  filename_hist_best = filebase+'.hist_best.json'
  filename_best = filebase+'.best.json'

  model.make_env()

  t = 0

  history = []
  history_best = [] # stores evaluation averages every 25 steps or so
  eval_log = []
  best_reward_eval = 0
  best_model_params_eval = None

  max_len = -1 # max time steps (-1 means ignore)

  # novelty search archive
  archive = deque(maxlen=MAX_ARCHIVE_ELEMENTS)

  while True:
    t += 1

    solutions = es.ask()

    if antithetic:
      seeds = seeder.next_batch(int(es.popsize/2))
      seeds = seeds+seeds
    else:
      seeds = seeder.next_batch(es.popsize)

    if FIXED_MAP:
      seeds = [FIXED_SEED] * es.popsize

    packet_list = encode_solution_packets(seeds, solutions, max_len=max_len)

    send_packets_to_slaves(packet_list)
    reward_list_total = receive_packets_from_slaves()

    objective_reward_list = reward_list_total[:, 1] # get rewards shape (population,)
    ########
    # novelty search
    if novelty_search:
      # array of shape (population, bc_size) with bc_size the vector length of the behaviour vector
      bc_array = reward_list_total[:, 2:]
      reward_list = compute_novelty(bc_array, archive, N_NEAREST_NEIGHBOURS)

      # update archive
      for i, r in enumerate(reward_list):
        if r > NOVELTY_THRESHOLD and np.random.random_sample() < PROBABILITY_ADD:
          print('Adding an element to the archive with novelty {:.2f}'.format(r))
          print('New length of archive: {}'.format(len(archive)))
          archive.append(bc_array[i])
    else:
      reward_list = objective_reward_list

    avg_obj_reward = int(np.mean(objective_reward_list)*100)/100.

    mean_time_step = int(np.mean(reward_list_total[:, 0])*100)/100. # get average time step
    max_time_step = int(np.max(reward_list_total[:, 0])*100)/100. # get average time step
    avg_reward = int(np.mean(reward_list)*100)/100. # get average time step
    std_reward = int(np.std(reward_list)*100)/100. # get average time step

    es.tell(reward_list)

    es_solution = es.result()
    model_params = es_solution[0] # best historical solution
    reward = es_solution[1] # best reward
    curr_reward = es_solution[2] # best of the current batch
    model.set_model_params(np.array(model_params).round(4))

    r_max = int(np.max(reward_list)*100)/100.
    r_min = int(np.min(reward_list)*100)/100.

    curr_time = int(time.time()) - start_time

    h = (t, curr_time, avg_reward, r_min, r_max, std_reward, int(es.rms_stdev()*100000)/100000., mean_time_step+1., int(max_time_step)+1, avg_obj_reward)

    if cap_time_mode:
      max_len = 2*int(mean_time_step+1.0)
    else:
      max_len = -1

    history.append(h)

    print('Saving history, generation {}'.format(t))
    with open(filename, 'wt') as out:
      res = json.dump([np.array(es.current_param()).round(4).tolist()], out, sort_keys=True, indent=2, separators=(',', ': '))

    with open(filename_hist, 'wt') as out:
      res = json.dump(history, out, sort_keys=False, indent=0, separators=(',', ':'))

    sprint('timestep, curr_time, avg_r, r_min, r_max, std_r, .., avg_obj_r')
    sprint(gamename, h)

    if (t == 1):
      best_reward_eval = avg_obj_reward
    if (t % eval_steps == 0): # evaluate on actual task at hand

      prev_best_reward_eval = best_reward_eval
      model_params_quantized = np.array(es.current_param()).round(4)
      reward_eval = evaluate_batch(model_params_quantized, max_len=-1)
      model_params_quantized = model_params_quantized.tolist()
      improvement = reward_eval - best_reward_eval
      eval_log.append([t, reward_eval, model_params_quantized])
      with open(filename_log, 'wt') as out:
        res = json.dump(eval_log, out)
      if (len(eval_log) == 1 or reward_eval > best_reward_eval):
        best_reward_eval = reward_eval
        best_model_params_eval = model_params_quantized
      else:
        if retrain_mode:
          sprint("reset to previous best params, where best_reward_eval =", best_reward_eval)
          es.set_mu(best_model_params_eval)
      with open(filename_best, 'wt') as out:
        res = json.dump([best_model_params_eval, best_reward_eval], out, sort_keys=True, indent=0, separators=(',', ': '))
      # dump history of best
      curr_time = int(time.time()) - start_time
      best_record = [t, curr_time, "improvement", improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval]
      history_best.append(best_record)
      with open(filename_hist_best, 'wt') as out:
        res = json.dump(history_best, out, sort_keys=False, indent=0, separators=(',', ':'))

      sprint("Eval", t, curr_time, "improvement", improvement, "curr", reward_eval, "prev", prev_best_reward_eval, "best", best_reward_eval)


def main(args):
  global optimizer, num_episode, eval_steps, num_worker, num_worker_trial, antithetic, seed_start, retrain_mode
  global cap_time_mode, model_name, novelty_search, unique_id, novelty_mode

  optimizer = args.optimizer
  num_episode = args.num_episode
  eval_steps = args.eval_steps
  num_worker = args.num_worker
  num_worker_trial = args.num_worker_trial
  antithetic = (args.antithetic == 1)
  retrain_mode = (args.retrain == 1)
  cap_time_mode= (args.cap_time == 1)
  seed_start = args.seed_start
  model_name = args.name
  novelty_search = args.novelty_search
  unique_id = args.unique_id
  novelty_mode = args.novelty_mode

  initialize_settings(args.sigma_init, args.sigma_decay)

  sprint("process", rank, "out of total ", comm.Get_size(), "started")
  if (rank == 0):
    master()
  else:
    slave()

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nworkers, rank
    nworkers = comm.Get_size()
    rank = comm.Get_rank()
    print('assigning the rank and nworkers', nworkers, rank)
    return "child"

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))

  parser.add_argument('-o', '--optimizer', type=str, help='ses, pepg, openes, ga, cma.', default='cma')
  parser.add_argument('--num_episode', type=int, default=16, help='num episodes per trial')  # was 16
  parser.add_argument('--eval_steps', type=int, default=5, help='evaluate every eval_steps step')  # was 25, each eval correspond to each worker doing num_episodes
  parser.add_argument('-n', '--num_worker', type=int, default=12)  # was 64
  parser.add_argument('-t', '--num_worker_trial', type=int, help='trials per worker', default=1)  # was 1
  parser.add_argument('--antithetic', type=int, default=1, help='set to 0 to disable antithetic sampling')
  parser.add_argument('--cap_time', type=int, default=0, help='set to 0 to disable capping timesteps to 2x of average.')
  parser.add_argument('--retrain', type=int, default=0, help='set to 0 to disable retraining every eval_steps if results suck.\n only works w/ ses, openes, pepg.')
  parser.add_argument('-s', '--seed_start', type=int, default=0, help='initial seed')
  parser.add_argument('--sigma_init', type=float, default=0.1, help='sigma_init')
  parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')
  parser.add_argument('--name', type=str, required=True, help='model name')
  parser.add_argument('--novelty_search', default=False, action='store_true', help='novelty fitness')
  parser.add_argument('--unique_id', type=str, required=True)
  parser.add_argument('--novelty_mode', type=str, default='', help='either h, z or h_concat')

  args = parser.parse_args()
  if "parent" == mpi_fork(args.num_worker+1): os.exit()
  main(args)
