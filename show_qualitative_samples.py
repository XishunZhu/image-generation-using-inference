from __future__ import division
from __future__ import print_function

import dateutil
import dateutil.tz
import datetime
import argparse
from pprint import pprint
import torchfile


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
    parser.add_argument('--nb', dest='examples_nb',
                        help='Number of examples to show',
                        default=10, type=int)
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args

def show_images(samples):
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
    for s in range(min(sample_nb, args.examples_nb)):
        super_image = np.zeros((x*n + (n-1)*5, y, c))
        for k in range(n):
            super_image[k*(x+5):x + k*(x+5), :, :] = image_samples[k, s]
        plt.imshow((super_image + 1.)/2.)
        """plt.text(x + 3, 8, 'latent distance (std): %.2f (%.2f)' % (np.linalg.norm(latents[s]-latent_mean),\
                                                                   np.linalg.norm(latent_std)))
        plt.text(x + 3, 16, 'fake latent distance (std): %.2f (%.2f)' % (np.linalg.norm(fake_latents[s]-fake_latent_mean),\
                                                                         np.linalg.norm(fake_latent_std)))
        plt.text(x + 3, 24, 'real logit: %.2f' % sp.special.expit(real_logit[s]))
        plt.text(x + 3, 32, 'fake image: %.2f' % sp.special.expit(fake_images_logit[s]))
        plt.text(x + 3, 40, 'fake latent: %.2f' % sp.special.expit(fake_latents_logit[s]))"""
        plt.show()
        k += 1

def prepare_bottlenecks(images):
    with slim.arg_scope(resnet_arg_scope()):
        #logits, end_points = inception.inception_v3(images, num_classes=228, is_training=False)
        #bottlenecks = end_points['PreLogits']
        bottlenecks, end_points = resnet(images, is_training=False)
        bottlenecks = tf.squeeze(bottlenecks)
    return bottlenecks


def sample_conditionned_latent_variable(embeddings, batch_size, model): # TO VERIFY
    '''Helper function for init_opt'''
    c_mean_logsigma = model.generate_condition(embeddings)
    mean = c_mean_logsigma[0]
    if cfg.TRAIN.COND_AUGMENTATION:
        # epsilon = tf.random_normal(tf.shape(mean))
        epsilon = tf.truncated_normal(tf.shape(mean))
        stddev = tf.exp(c_mean_logsigma[1])
        c = mean + stddev * epsilon
    else:
        c = mean
        
    z = tf.random_normal([batch_size, cfg.Z_DIM])

    return tf.concat([c, z], 1)


def build_model(sess, image_shape, embedding_dim, batch_size):
    model = CondALI(
        image_shape=image_shape
    )

    embeddings = tf.placeholder(
        tf.float32, [batch_size, embedding_dim],
        name='conditional_embeddings')

    images = tf.placeholder(
        tf.float32, [batch_size] + image_shape,
        name='real_images')

    bottlenecks = prepare_bottlenecks(images)

    with tf.variable_scope("e_net"):
        #Add Cond_AUG to fake latents
        fake_latents = model.get_encoder(bottlenecks)

    with pt.defaults_scope(phase=pt.Phase.test):
        with tf.variable_scope("g_net"):
            latents = sample_conditionned_latent_variable(embeddings, batch_size, model)
            fake_latents = sample_conditionned_latent_variable(fake_latents, batch_size, model)
            fake_images = model.get_generator(latents)
            fake_images_2 = model.get_generator(fake_latents)

    ckt_path = cfg.TEST.PRETRAINED_MODEL
    if ckt_path.find('.ckpt') != -1:
        print("Reading model parameters from %s" % ckt_path)
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess, ckt_path)
    else:
        print("Input a valid model path.")
    return embeddings, fake_images, fake_images_2

if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id

    # Load text embeddings generated from the encoder
    cap_path = cfg.TEST.CAPTION_PATH
    t_file = torchfile.load(cap_path)
    captions_list = t_file.raw_txt
    print(t_file.fea_txt)
    embeddings = np.concatenate(t_file.fea_txt, axis=0)
    num_embeddings = len(captions_list)
    print('Successfully load sentences from: ', cap_path)
    print('Total number of sentences:', num_embeddings)
    print('num_embeddings:', num_embeddings, embeddings.shape)
    datadir = 'Data/%s' % cfg.DATASET_NAME
    dataset = TextDataset(datadir, cfg.EMBEDDING_TYPE, 1)

    # path to save generated samples
    save_dir = cap_path[:cap_path.find('.t7')]
    if num_embeddings > 0:
        batch_size = np.minimum(num_embeddings, cfg.TEST.BATCH_SIZE)

        # Build StackGAN and load the model
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                embeddings_holder, fake_images_opt, fake_images_2_opt =\
                    build_model(sess, dataset.image_shape, embeddings.shape[-1], batch_size)

                count = 0
                while count < num_embeddings:
                    iend = count + batch_size
                    if iend > num_embeddings:
                        iend = num_embeddings
                        count = num_embeddings - batch_size
                    embeddings_batch = embeddings[count:iend]
                    captions_batch = captions_list[count:iend]

                    samples_batches = []
                    samples_2_batches = []
                    # Generate up to 16 images for each sentence with
                    # randomness from noise z and conditioning augmentation.
                    for i in range(np.minimum(16, cfg.TEST.NUM_COPY)):
                        samples, samples_2 =\
                            sess.run([fake_images_opt, fake_images_2_opt],
                                     {embeddings_holder: embeddings_batch})
                        samples_batches.append(samples)
                        samples_2_batches.append(samples_2)
                    save_super_images(samples_batches,
                                      samples_2_batches,
                                      captions_batch,
                                      batch_size,
                                      count, save_dir)
                    count += batch_size

        print('Finish generating samples for %d sentences:' % num_embeddings)
        print('Example sentences:')
        for i in xrange(np.minimum(10, num_embeddings)):
            print('Sentence %d: %s' % (i, captions_list[i]))
        

