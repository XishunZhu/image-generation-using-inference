from __future__ import division
from __future__ import print_function

import dateutil
import dateutil.tz
import datetime
import argparse
from pprint import pprint


from misc.datasets import TextDataset
from stageI.model import CondALI
from stageI.trainer import CondALITrainer
from StackGAN.misc.utils import mkdir_p
from misc.config import cfg, cfg_from_file

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args

def show_images(samples_batches):
    for samples in samples_batches:
        latents = np.array(samples[0])
        latent_mean = np.mean(latents, axis = 0)
        latent_std = np.std(latents, axis = 0)
        fake_latents = np.array(samples[1])
        fake_latent_mean = np.mean(fake_latents, axis = 0)
        fake_latent_std = np.std(fake_latents, axis = 0)
        real_logit = np.array(samples[2])
        fake_images_logit = np.array(samples[3])
        fake_latents_logit = np.array(samples[4])
        image_samples = np.array(samples[5:])
        (n, sample_nb, x, y, c) = image_samples.shape
        for s in range(10):#sample_nb):
            super_image = np.zeros((x*n + (n-1)*5, y, c))
            for k in range(n):
                super_image[k*(x+5):x + k*(x+5), :, :] = image_samples[k, s]
            plt.imshow((super_image + 1.)/2.)
            plt.text(x + 3, 8, 'latent distance (std): %.2f (%.2f)' % (np.linalg.norm(latents[s]-latent_mean),\
                                                                   np.linalg.norm(latent_std)))
            plt.text(x + 3, 16, 'fake latent distance (std): %.2f (%.2f)' % (np.linalg.norm(fake_latents[s]-fake_latent_mean),\
                                                                         np.linalg.norm(fake_latent_std)))
            plt.text(x + 3, 24, 'real logit: %.2f' % sp.special.expit(real_logit[s]))
            plt.text(x + 3, 32, 'fake image: %.2f' % sp.special.expit(fake_images_logit[s]))
            plt.text(x + 3, 40, 'fake latent: %.2f' % sp.special.expit(fake_latents_logit[s]))
            #if cfg.SAVE_EMBEDDINGS:
            #    plt.savefig( "img" + str(s) + ".png" )
            plt.show()

def save_images(samples_batches, startID, save_dir):
    if not os.path.isdir(save_dir):
        print('Make a new folder: ', save_dir)
        mkdir_p(save_dir)
    
    k = 0
    for samples in samples_batches:
        for sample in samples:
            full_path = os.path.join(save_dir, startID + k)
            sp.misc.imsave(str(full_path), sample)
            k += 1
    print("%i images saved in %s directory" % (k, save_dir))

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    #print('Using config:')
    #pprint(cfg)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    datadir = 'Data/%s' % cfg.DATASET_NAME
    dataset = TextDataset(datadir, cfg.EMBEDDING_TYPE, 1)
    filename_test = '%s/test' % (datadir)
    dataset.test = dataset.get_data(filename_test)
    if cfg.TRAIN.FLAG:
        filename_train = '%s/train' % (datadir)
        dataset.train = dataset.get_data(filename_train)

        ckt_logs_dir = "ckt_logs/%s/%s_%s" % \
            (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
        mkdir_p(ckt_logs_dir)
    else:
        s_tmp = cfg.TRAIN.PRETRAINED_ALI_MODEL
        ckt_logs_dir = s_tmp[:s_tmp.find('.ckpt')]

    model = CondALI(
        image_shape=dataset.image_shape
    )

    algo = CondALITrainer(
        model=model,
        dataset=dataset,
        ckt_logs_dir=ckt_logs_dir
    )
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        ''' For every input text embedding/sentence in the
        training and test datasets, generate cfg.TRAIN.NUM_COPY
        images with randomness from noise z and conditioning augmentation.'''
        samples_batches = algo.evaluate()
        #if cfg.SAVE_IMAGES:
            #save_images(samples_batches)
