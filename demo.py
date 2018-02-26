from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import argparse
import torchfile
from PIL import Image, ImageDraw, ImageFont
import re

from tensorflow.contrib import slim
from models.research.slim import nets
from models.research.slim.nets.resnet_v1 import resnet_v1_50 as resnet
from models.research.slim.nets.resnet_v1 import resnet_arg_scope

from stageI.model import CondALI
from stageI.trainer import CondALITrainer

from misc.datasets import TextDataset
from misc.config import cfg, cfg_from_file
from misc.utils import mkdir_p


def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=-1, type=int)
    parser.add_argument('--caption_path', type=str, default=None,
                        help='Path to the file with text sentences')
    # if len(sys.argv) == 1:
    #    parser.print_help()
    #    sys.exit(1)
    args = parser.parse_args()
    return args


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
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, ckt_path)
    else:
        print("Input a valid model path.")
    return embeddings, fake_images, fake_images_2


def drawCaption(img, caption):
    img_txt = Image.fromarray(img)
    # get a font
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)

    # draw text, half opacity
    d.text((10, 256), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
    d.text((10, 512), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))
    if img.shape[0] > 832:
        d.text((10, 832), 'Stage-I', font=fnt, fill=(255, 255, 255, 255))
        d.text((10, 1088), 'Stage-II', font=fnt, fill=(255, 255, 255, 255))

    idx = caption.find(' ', 60)
    if idx == -1:
        d.text((256, 10), caption, font=fnt, fill=(255, 255, 255, 255))
    else:
        cap1 = caption[:idx]
        cap2 = caption[idx+1:]
        d.text((256, 10), cap1, font=fnt, fill=(255, 255, 255, 255))
        d.text((256, 60), cap2, font=fnt, fill=(255, 255, 255, 255))

    return img_txt


def save_super_images(sample_batchs, hr_sample_batchs,
                      captions_batch, batch_size,
                      startID, save_dir):
    if not os.path.isdir(save_dir):
        print('Make a new folder: ', save_dir)
        mkdir_p(save_dir)

    # Save up to 16 samples for each text embedding/sentence
    img_shape = hr_sample_batchs[0][0].shape
    for j in range(batch_size):
        if not re.search('[a-zA-Z]+', captions_batch[j]):
            continue

        padding = np.zeros(img_shape)
        row1 = [padding]
        row2 = [padding]
        # First row with up to 8 samples
        for i in range(np.minimum(8, len(sample_batchs))):
            lr_img = sample_batchs[i][j]
            hr_img = hr_sample_batchs[i][j]
            hr_img = (hr_img + 1.0) * 127.5
            re_sample = scipy.misc.imresize(lr_img, hr_img.shape[:2])
            row1.append(re_sample)
            row2.append(hr_img)
        row1 = np.concatenate(row1, axis=1)
        row2 = np.concatenate(row2, axis=1)
        superimage = np.concatenate([row1, row2], axis=0)

        # Second 8 samples with up to 8 samples
        if len(sample_batchs) > 8:
            row1 = [padding]
            row2 = [padding]
            for i in range(8, len(sample_batchs)):
                lr_img = sample_batchs[i][j]
                hr_img = hr_sample_batchs[i][j]
                hr_img = (hr_img + 1.0) * 127.5
                re_sample = scipy.misc.imresize(lr_img, hr_img.shape[:2])
                row1.append(re_sample)
                row2.append(hr_img)
            row1 = np.concatenate(row1, axis=1)
            row2 = np.concatenate(row2, axis=1)
            super_row = np.concatenate([row1, row2], axis=0)
            superimage2 = np.zeros_like(superimage)
            superimage2[:super_row.shape[0],
                        :super_row.shape[1],
                        :super_row.shape[2]] = super_row
            mid_padding = np.zeros((64, superimage.shape[1], 3))
            superimage =\
                np.concatenate([superimage, mid_padding, superimage2], axis=0)

        top_padding = np.zeros((128, superimage.shape[1], 3))
        superimage =\
            np.concatenate([top_padding, superimage], axis=0)

        fullpath = '%s/sentence%d.jpg' % (save_dir, startID + j)
        superimage = drawCaption(np.uint8(superimage), captions_batch[j])
        scipy.misc.imsave(fullpath, superimage)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    if args.caption_path is not None:
        cfg.TEST.CAPTION_PATH = args.caption_path

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
    filename_test = '%s/test' % (datadir)
    dataset.test = dataset.get_data(filename_test)




    # path to save generated samples
    save_dir = cap_path[:cap_path.find('.t7')]
    if num_embeddings > 0:
        batch_size = np.minimum(num_embeddings, cfg.TEST.BATCH_SIZE)

        # Build StackGAN and load the model
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                #embeddings_holder, fake_images_opt, fake_images_2_opt =\
                #    build_model(sess, dataset.image_shape, embeddings.shape[-1], batch_size)
                model = CondALI(
                    image_shape=dataset.image_shape
                )
                algo = CondALITrainer(
                    model=model,
                    dataset=dataset,
                    ckt_logs_dir=cfg.TEST.PRETRAINED_MODEL
                )

                ckt_path = cfg.TEST.PRETRAINED_MODEL
                if ckt_path.find('.ckpt') != -1:
                    algo.init_opt()
                    print("Reading model parameters from %s" % ckt_path)
                    saver = tf.train.Saver(tf.global_variables())
                    saver.restore(sess, ckt_path)
                else:
                    print("Input a valid model path.")

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
                        samples =\
                            sess.run([algo.fake_images],
                                     {algo.embeddings: embeddings_batches[i]})
                        samples_batches.append(samples)
                        #samples_2_batches.append(samples_2)
                    save_super_images(samples_batches,
                                      #samples_2_batches,
                                      captions_batch,
                                      batch_size,
                                      count, save_dir)
                    count += batch_size

        print('Finish generating samples for %d sentences:' % num_embeddings)
        print('Example sentences:')
        for i in xrange(np.minimum(10, num_embeddings)):
            print('Sentence %d: %s' % (i, captions_list[i]))
