import os
import json

import numpy as np
import matplotlib.pyplot as plt

def plot_reward_evolution(root, env_name, optimizer, model_name, num_rollouts=1, popsize=14, unique_id='',
                          novelty_search=False, novelty_mode='', ns_mode='', required_score=800):
    file_base = (env_name + '.' + optimizer + '.' + model_name + '.' + str(num_rollouts) + '.' + str(popsize))

    if unique_id:
        file_base = file_base + '.' + unique_id
    if novelty_search:
        file_base = file_base + '.novelty'
    if novelty_mode:
        file_base = file_base + '.' + novelty_mode
    if ns_mode:
        file_base = file_base + '.' + ns_mode

    # Load history
    filename = os.path.join(root, 'log', file_base + '.hist.json')
    with open(filename, 'r') as f:
        raw_data = json.load(f)
    data = np.array(raw_data)
    print('Model trained for {} generations'.format(len(data)))

    # Load best model history
    filename = os.path.join(root, 'log', file_base + '.hist_best.json')
    with open(filename, 'r') as f:
        raw_data = json.load(f)
    raw_best_data = np.array(raw_data)
    best_data = []
    for bdata in raw_best_data:
        best_data.append([float(bdata[0]), float(bdata[1]), float(bdata[5]), float(bdata[9]), required_score])
    best_data = np.array(best_data)
    print('Model evaluated {} times'.format(len(best_data)))

    fig = plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
    line_mean, = plt.plot(data[:, 0], data[:, 2])
    line_min, = plt.plot(data[:, 0], data[:, 3])
    line_max, = plt.plot(data[:, 0], data[:, 4])
    line_best, = plt.plot(best_data[:, 0], best_data[:, 2])
    line_req, = plt.plot(best_data[:, 0], best_data[:, 4])
    plt.legend([line_mean, line_min, line_max, line_req, line_best],
               ['mean', 'min', 'max', 'requirement', 'best avg score'])
    plt.xlabel('generation')
    plt.xticks(np.arange(0, len(data), 5))
    plt.ylabel('cumulative reward')
    plt.yticks(np.arange(-100, 1000, 50))
    plt.title(file_base)
    # plt.savefig(file_base+".svg")
    plt.show()


def plot_latent_variations(frame, vae, factor=0):
    batch_z = vae.encode_mu_logvar(frame)[0]  # vae.encode(frame)
    print('Latent vector z')
    print(batch_z[0])  # print out sampled z
    reconstruct = vae.decode(batch_z)

    plt.figure(figsize=((10, 5)))
    plt.subplot(121)
    plt.imshow(frame[0])
    plt.title('Original')
    # show reconstruction
    plt.subplot(122)
    plt.imshow(reconstruct[0])
    plt.title('Reconstruction')
    plt.show()

    print('Visualise latent factor {}, original value={:.2f}'.format(factor, batch_z[0][factor]))
    batch_z_copy = batch_z.copy()
    plt.figure(figsize=(20, 5))
    for i, value in enumerate(np.linspace(-1.5, 1.5, 10)):
        batch_z_copy[0, factor] = value
        reconstruct = vae.decode(batch_z_copy)
        # show reconstruction
        plt.subplot(1, 10, i + 1)
        plt.imshow(reconstruct[0])
        plt.title('{:.2f}'.format(value))
    plt.show()