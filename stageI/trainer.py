from __future__ import division
from __future__ import print_function

import prettytensor as pt
import tensorflow as tf
import numpy as np
import scipy.misc
import os
import sys
from six.moves import range
from progressbar import ETA, Bar, Percentage, ProgressBar

from pprint import pprint

from tensorflow.contrib import slim
from models.research.slim import nets
from models.research.slim.nets.resnet_v1 import resnet_v1_50 as resnet
from models.research.slim.nets.resnet_v1 import resnet_arg_scope
from pprint import pprint
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from misc.config import cfg
from misc.utils import mkdir_p

TINY = 1e-8


# reduce_mean normalize also the dimension of the embeddings
def KL_loss(mu, log_sigma):
    with tf.name_scope("KL_divergence"):
        loss = -log_sigma + .5 * (-1 + tf.exp(2. * log_sigma) + tf.square(mu))
        loss = tf.reduce_mean(loss)
        return loss

class CondALITrainer(object):
    def __init__(self,
                 model,
                 dataset=None,
                 exp_name="model",
                 ckt_logs_dir="ckt_logs",
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.exp_name = exp_name
        self.log_dir = ckt_logs_dir
        self.checkpoint_dir = ckt_logs_dir

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        self.StackGAN_model_path = cfg.TRAIN.PRETRAINED_STACKGAN_MODEL
        self.resnet_model_path = cfg.TRAIN.PRETRAINED_RESNET_MODEL
        self.ALI_model_path = cfg.TRAIN.PRETRAINED_ALI_MODEL

        self.log_vars = []

    
    def prepare_bottlenecks(self, images):
        with slim.arg_scope(resnet_arg_scope()):
            #logits, end_points = inception.inception_v3(images, num_classes=228, is_training=False)
            #bottlenecks = end_points['PreLogits']
            bottlenecks, end_points = resnet(images, is_training=False)
            bottlenecks = tf.squeeze(bottlenecks)
        return bottlenecks
    
    def build_placeholder(self): # TO VERIFY
        '''Helper function for init_opt'''
        self.images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='real_images')
        self.wrong_images = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.image_shape,
            name='wrong_images'
        )
        self.embeddings = tf.placeholder(
            tf.float32, [self.batch_size] + self.dataset.embedding_shape,
            name='conditional_embeddings'
        )
        self.generator_lr = tf.placeholder(
            tf.float32, [],
            name='generator_learning_rate'
        )
        self.encoder_lr = tf.placeholder(
            tf.float32, [],
            name='encoder_learning_rate'
        )
        self.discriminator_lr = tf.placeholder(
            tf.float32, [],
            name='discriminator_learning_rate'
        )

    def sample_conditionned_latent_variable(self, embeddings): # TO VERIFY
        '''Helper function for init_opt'''
        c_mean_logsigma = self.model.generate_condition(embeddings)
        mean = c_mean_logsigma[0]
        if cfg.TRAIN.COND_AUGMENTATION:
            # epsilon = tf.random_normal(tf.shape(mean))
            epsilon = tf.truncated_normal(tf.shape(mean))
            stddev = tf.exp(c_mean_logsigma[1])
            c = mean + stddev * epsilon

            kl_loss = KL_loss(c_mean_logsigma[0], c_mean_logsigma[1])
        else:
            c = mean
            kl_loss = 0
            
        z = tf.random_normal([self.batch_size, cfg.Z_DIM])

        return tf.concat([c, z], 1), cfg.TRAIN.COEFF.KL * kl_loss        

    def init_opt(self):
        self.build_placeholder()
        self.bottlenecks = self.prepare_bottlenecks(self.images)
        print( "Bottleneck Shape:", self.bottlenecks.shape )
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("g_net"):
                # ####get output from G network################################
                # c, kl_loss = self.sample_encoded_context(self.embeddings)
                # z = tf.random_normal([self.batch_size, cfg.Z_DIM])
                # self.log_vars.append(("hist_c", c))
                # self.log_vars.append(("hist_z", z))
                latents, kl_loss = self.sample_conditionned_latent_variable(self.embeddings)
                self.log_vars.append(("hist_l", latents))
                print("Latent Variable Dimensions", latents.shape)
                self.fake_images = self.model.get_generator(latents)

            with tf.variable_scope("e_net"):
                #Add Cond_AUG to fake latents
                fake_latents = self.model.get_encoder(self.bottlenecks)
                fake_latents, _ = self.sample_conditionned_latent_variable(fake_latents)
                
            if cfg.TRAIN.SUPERVISED:

            # ####get discriminator_loss and generator_loss ###################
                discriminator_loss, generator_loss, encoder_loss =\
                self.compute_losses(self.images,
                                    latents,
                                    self.fake_images,
                                    self.wrong_images,
                                    fake_latents)
                self.log_vars.append(("e_loss", encoder_loss))
            else:
                discriminator_loss, generator_loss, encoder_loss =\
                self.compute_losses(self.images,
                                    fake_latents,
                                    self.fake_images,
                                    self.wrong_images,
                                    None)
                
                
            generator_loss += kl_loss # Add KL Divergence as regularizer for stable training
            self.log_vars.append(("g_loss_kl_loss", kl_loss))
            self.log_vars.append(("g_loss", generator_loss))
            self.log_vars.append(("d_loss", discriminator_loss))

            # #######Total loss for build optimizers###########################
            self.prepare_trainer(generator_loss, discriminator_loss, encoder_loss)
            # #######define self.g_sum, self.d_sum,....########################
            self.define_summaries()

        with pt.defaults_scope(phase=pt.Phase.test):
            self.visualization(cfg.TRAIN.NUM_COPY)
            with tf.variable_scope("e_net", reuse=True):
                self.fake_latents, _ = self.sample_conditionned_latent_variable(self.model.get_encoder(self.bottlenecks))
                
            with tf.variable_scope("g_net", reuse=True):
                self.latents, _ = self.sample_conditionned_latent_variable(self.embeddings)
                self.fake_images_test = self.model.get_generator(fake_latents)
                
            with tf.variable_scope("d_net"):
                self.real_logit = self.model.get_discriminator(self.images, self.latents)
                self.fake_images_logit = self.model.get_discriminator(self.fake_images, self.latents)
                self.fake_latents_logit = self.model.get_discriminator(self.images, self.fake_latents)

    def compute_losses(self, images, latents, fake_images, wrong_images, fake_latents = None):
        
        
        #Real Images, Real Latent Variables
        real_logit = self.model.get_discriminator(images, latents)
        # Intermediate latent features from the discriminator
        latent_features = self.model.get_intermediate_latent_features(latents)

        real_d_loss =\
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logit,
                                                    labels=tf.ones_like(real_logit))
        real_d_loss = tf.reduce_mean(real_d_loss)
            
        #Fake Images, Real Latent Variables
        fake_images_logit = self.model.get_discriminator(fake_images, latents)
        fake_images_d_loss =\
        tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_images_logit,
                                                labels=tf.zeros_like(fake_images_logit))
        fake_images_d_loss = tf.reduce_mean(fake_images_d_loss)
        
        #Wrong Images, Real Latent Variables
        wrong_logit = self.model.get_discriminator(wrong_images, latents)
        wrong_d_loss =\
        tf.nn.sigmoid_cross_entropy_with_logits(logits=wrong_logit,
                                                labels=tf.zeros_like(wrong_logit))
        wrong_d_loss = tf.reduce_mean(wrong_d_loss)
        
        if cfg.TRAIN.SUPERVISED:
            assert fake_latents is not None
            
            #Real Images, Fake Latent Variables
            fake_latents_logit = self.model.get_discriminator(images, fake_latents)
            # Intermediate latent features from the discriminator
            fake_latent_features = self.model.get_intermediate_latent_features(fake_latents)
            fake_latents_d_loss =\
                tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_latents_logit,
                                                        labels=tf.zeros_like(fake_latents_logit))
            fake_latents_d_loss = tf.reduce_mean(fake_latents_d_loss)

            discriminator_loss =\
            real_d_loss + (wrong_d_loss + fake_images_d_loss + fake_latents_d_loss)/3.
            
            self.log_vars.append(("d_loss_fake_latents", fake_latents_d_loss))

            if cfg.TRAIN.ENCODER_FEATURE_MATCHING:
                encoder_loss = tf.sqrt(tf.reduce_sum(tf.pow(latent_features-fake_latent_features, 2)))
            else:
                encoder_loss = \
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_latents_d_loss,
                                                        labels=tf.ones_like(fake_latents_d_loss))
                encoder_loss = tf.reduce_mean(encoder_loss)
            
        else:
            discriminator_loss =\
            real_d_loss + (wrong_d_loss + fake_images_d_loss)/2.
            
            encoder_loss = None
            

        self.log_vars.append(("d_loss_wrong", wrong_d_loss))
        self.log_vars.append(("d_loss_real", real_d_loss))
        self.log_vars.append(("d_loss_fake_images", fake_images_d_loss))

        generator_loss = \
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_images_logit,
                                                    labels=tf.ones_like(fake_images_logit))
        generator_loss = tf.reduce_mean(generator_loss)

        return discriminator_loss, generator_loss, encoder_loss

    def prepare_trainer(self, generator_loss, discriminator_loss, encoder_loss):
        '''Helper function for init_opt'''
        all_vars = tf.trainable_variables()

        if cfg.TRAIN.GENERATOR:
            g_vars = [var for var in all_vars if
                  var.name.startswith('g_')]
        
            generator_opt = tf.train.AdamOptimizer(self.generator_lr,
                                                   beta1=0.5)
            self.generator_trainer =\
            pt.apply_optimizer(generator_opt,
                               losses=[generator_loss],
                               var_list=g_vars)
            
            self.log_vars.append(("e_learning_rate", self.encoder_lr))
            

        if cfg.TRAIN.SUPERVISED and cfg.TRAIN.ENCODER:
            e_vars = [var for var in all_vars if
                  var.name.startswith('e_')]
            encoder_opt = tf.train.AdamOptimizer(self.encoder_lr,
                                               beta1=0.5)
            self.encoder_trainer =\
            pt.apply_optimizer(encoder_opt,
                               losses=[encoder_loss],
                               var_list=e_vars)
        
            self.log_vars.append(("g_learning_rate", self.generator_lr))
            
        if cfg.TRAIN.DISCRIMINATOR:
            d_vars_to_train = []
            if cfg.TRAIN.DISCRIMINATOR_IMAGES:
                d_i_vars = [var for var in all_vars if var.name.startswith('d_i_')]
                d_vars_to_train += d_i_vars
            if cfg.TRAIN.DISCRIMINATOR_LATENTS:
                d_l_vars = [var for var in all_vars if var.name.startswith('d_l_')]
                d_vars_to_train += d_l_vars
            if cfg.TRAIN.DISCRIMINATOR_FUSION:
                d_f_vars = [var for var in all_vars if var.name.startswith('d_f')]
                d_vars_to_train += d_f_vars

            discriminator_opt = tf.train.AdamOptimizer(self.discriminator_lr,
                                                   beta1=0.5)
            self.discriminator_trainer =\
            pt.apply_optimizer(discriminator_opt,
                               losses=[discriminator_loss],
                               var_list=d_vars_to_train)
            
            self.log_vars.append(("d_learning_rate", self.discriminator_lr))

    def define_summaries(self):
        '''Helper function for init_opt'''
        all_sum = {'g': [], 'e': [], 'd': [], 'hist': []}
        for k, v in self.log_vars:
            if k.startswith('g'):
                all_sum['g'].append(tf.summary.scalar(k, v))
            elif k.startswith('e'):
                all_sum['e'].append(tf.summary.scalar(k, v))
            elif k.startswith('d'):
                all_sum['d'].append(tf.summary.scalar(k, v))
            elif k.startswith('hist'):
                all_sum['hist'].append(tf.summary.histogram(k, v))

        self.g_sum = tf.summary.merge(all_sum['g'])
        self.e_sum = tf.summary.merge(all_sum['e'])
        self.d_sum = tf.summary.merge(all_sum['d'])
        self.hist_sum = tf.summary.merge(all_sum['hist'])

    def visualize_one_superimage(self, img_var, images, rows, filename):
        stacked_img = []
        for row in range(rows):
            img = images[row * rows, :, :, :]
            row_img = [img]  # real image
            for col in range(rows):
                row_img.append(img_var[row * rows + col, :, :, :])
            # each rows is 1realimage +10_fakeimage
            stacked_img.append(tf.concat(row_img, 1))
        imgs = tf.expand_dims(tf.concat(stacked_img, 0), 0)
        current_img_summary = tf.summary.image(filename, imgs)
        return current_img_summary, imgs

    def visualization(self, n):
        fake_sum_train, superimage_train = \
            self.visualize_one_superimage(self.fake_images[:n * n],
                                          self.images[:n * n],
                                          n, "train")
        fake_sum_test, superimage_test = \
            self.visualize_one_superimage(self.fake_images[n * n:2 * n * n],
                                          self.images[n * n:2 * n * n],
                                          n, "test")
        self.superimages = tf.concat([superimage_train, superimage_test], 0)
        self.image_summary = tf.summary.merge([fake_sum_train, fake_sum_test])

    def preprocess(self, x, n):
        # make sure every row with n column have the same embeddings
        for i in range(n):
            for j in range(1, n):
                x[i * n + j] = x[i * n]
        return x

    def epoch_sum_images(self, sess, n):

        #Get Images, Embeddings, and Captions during Training
        images_train, _, embeddings_train, captions_train, _ =\
            self.dataset.train.next_batch(n * n, cfg.TRAIN.NUM_EMBEDDING)
        images_train = self.preprocess(images_train, n)
        embeddings_train = self.preprocess(embeddings_train, n)
        
        #Get Images, Embeddings, and Captions during Testing
        images_test, _, embeddings_test, captions_test, _ = \
            self.dataset.test.next_batch(n * n, 1)
        images_test = self.preprocess(images_test, n)
        embeddings_test = self.preprocess(embeddings_test, n)

        #Concat images and embeddings
        images = np.concatenate([images_train, images_test], axis=0)
        embeddings =\
            np.concatenate([embeddings_train, embeddings_test], axis=0)

        #Pad images if reqd
        if self.batch_size > 2 * n * n:
            images_pad, _, embeddings_pad, _, _ =\
                self.dataset.test.next_batch(self.batch_size - 2 * n * n, 1)
            images = np.concatenate([images, images_pad], axis=0)
            embeddings = np.concatenate([embeddings, embeddings_pad], axis=0)
        feed_dict = {self.images: images,
                     self.embeddings: embeddings}
        gen_samples, img_summary =\
            sess.run([self.superimages, self.image_summary], feed_dict)

        # save images generated for train and test captions
        scipy.misc.imsave('%s/train.jpg' % (self.log_dir), gen_samples[0])
        scipy.misc.imsave('%s/test.jpg' % (self.log_dir), gen_samples[1])

        #Write captions from training and testing to a .txt file
        """pfi_train = open(self.log_dir + "/train.txt", "w")
        pfi_test = open(self.log_dir + "/test.txt", "w")
        for row in range(n):
            pfi_train.write('\n***row %d***\n' % row)
            pfi_train.write(captions_train[row * n])

            pfi_test.write('\n***row %d***\n' % row)
            pfi_test.write(captions_test[row * n])
        pfi_train.close()
        pfi_test.close()"""

        return img_summary

    def build_model(self, sess):
        self.init_opt()
        sess.run(tf.global_variables_initializer())
        if cfg.TRAIN.USE_PRETRAINED_ALI:
            if len(self.ALI_model_path) > 0:
                #Restore ALI model with vars if model path is given
                print("Reading ALI model parameters from %s" % self.ALI_model_path)
                all_vars = tf.global_variables()
                ALI_restorer = tf.train.Saver(all_vars)
                ALI_restorer.restore(sess, self.ALI_model_path)
                print("Model successfully restored")

                istart = self.ALI_model_path.rfind('_') + 1
                iend = self.ALI_model_path.rfind('.')
                counter = self.ALI_model_path[istart:iend]
                counter = int(counter)
            else:
                raise StandardError("ALI model path is not provided")
                
        else:
            if len(self.StackGAN_model_path) > 0:
                #Restore model with vars if model path is given
                print("Reading StackGAN model parameters from %s" % self.StackGAN_model_path)
                #print_tensors_in_checkpoint_file(self.StackGAN_model_path, '', False)
                all_vars = tf.global_variables()

                StackGAN_g_vars = [('g_' + var.op.name.split('g_')[-1], var) for var in all_vars if (var.op.name.startswith('g_')\
                                  and not (var.op.name.endswith('biased') or var.op.name.endswith('local_step')))]
                StackGAN_d_vars = [('d_' + var.op.name.split('d_i_')[-1], var) for var in all_vars if (var.op.name.startswith('d_i')\
                                  and not (var.op.name.endswith('biased') or var.op.name.endswith('local_step')))]
                StackGAN_vars = StackGAN_g_vars + StackGAN_d_vars
                StackGAN_vars_to_restore = {var_name:var for var_name, var in StackGAN_vars}
                #pprint([var_name for var_name, _ in StackGAN_vars])
                StackGAN_restorer = tf.train.Saver(StackGAN_vars_to_restore)
                StackGAN_restorer.restore(sess, self.StackGAN_model_path)
                print("Model successfully restored")
            else:
                print("Created StackGan model with fresh parameters.")

            #Restore pretrained encoder
            if len(self.resnet_model_path) > 0:
                print("Reading encoder model parameters from %s" % self.resnet_model_path)
                with tf.variable_scope('resnet_v1_50', reuse=True):
                    resnet_vars_to_restore = [var for var in slim.get_variables_to_restore() if var.name.startswith('resnet_v1_50')]

                    #pprint(resnet_vars_to_restore)
                    resnet_restorer = tf.train.Saver(resnet_vars_to_restore)
                    resnet_restorer.restore(sess, self.resnet_model_path)
                    print("Model successfully restored")
            else:
                print("Created ResNet model with fresh parameters.")
                    
            counter = 0
        return counter

    def train(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                counter = self.build_model(sess)
                saver = tf.train.Saver(tf.global_variables(),
                                       keep_checkpoint_every_n_hours=2)

                # summary_op = tf.merge_all_summaries()
                summary_writer = tf.summary.FileWriter(self.log_dir,
                                                        sess.graph)

                keys = ["d_loss", "g_loss", "e_loss"]
                log_vars = []
                log_keys = []
                for k, v in self.log_vars:
                    if k in keys:
                        log_vars.append(v)
                        log_keys.append(k)
                        # print(k, v)
                generator_lr = cfg.TRAIN.GENERATOR_LR
                encoder_lr = cfg.TRAIN.ENCODER_LR
                discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
                num_embedding = cfg.TRAIN.NUM_EMBEDDING
                lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
                number_example = self.dataset.train._num_examples
                updates_per_epoch = int(number_example / self.batch_size)
                epoch_start = int(counter / updates_per_epoch)
                for epoch in range(epoch_start, self.max_epoch):
                    widgets = ["epoch #%d|" % epoch,
                               Percentage(), Bar(), ETA()]
                    pbar = ProgressBar(maxval=updates_per_epoch,
                                       widgets=widgets)
                    pbar.start()

                    #Decay Learning Rate by 1/2 every 'x' steps(Change to exp?)
                    if epoch % lr_decay_step == 0 and epoch != 0:
                        generator_lr *= 0.5
                        encoder_lr *= 0.5
                        discriminator_lr *= 0.5

                    #Exponential Decay for all LRs

                    #generator_lr = tf.train.exponential_decay(generator_lr,counter, 100000,
                    #        0.96, staircase=True)
                    #discriminator_lr = tf.train.exponential_decay(discriminator_lr,counter,
                    #        100000, 0.96, staircase=True)
                    #encoder_lr = tf.train.exponential_decay(encoder_lr,counter, 100000,
                    #        0.96, staircase=True)

                    all_log_vals = []
                    for i in range(updates_per_epoch):
                        pbar.update(i)
                        
                        images, wrong_images, embeddings, _, _ =\
                            self.dataset.train.next_batch(self.batch_size,
                                                          num_embedding)
                        feed_dict = {self.images: images,
                                     self.wrong_images: wrong_images,
                                     self.embeddings: embeddings,
                                     self.generator_lr: generator_lr,
                                     self.encoder_lr: encoder_lr,
                                     self.discriminator_lr: discriminator_lr
                                     }
                        if cfg.TRAIN.DISCRIMINATOR:
                            # Train the discriminator
                            feed_out = [self.discriminator_trainer,
                                        self.d_sum,
                                        self.hist_sum,
                                        log_vars]
                            _, d_sum, hist_sum, log_vals = sess.run(feed_out,
                                                                    feed_dict)
                            summary_writer.add_summary(d_sum, counter)
                            summary_writer.add_summary(hist_sum, counter)
                            all_log_vals.append(log_vals)
                        if cfg.TRAIN.SUPERVISED and cfg.TRAIN.ENCODER and i % cfg.TRAIN.ENCODER_PERIOD == 0:
                            # Train the encoder
                            feed_out = [self.encoder_trainer,
                                        self.e_sum]
                            _, e_sum = sess.run(feed_out,
                                                feed_dict)
                            summary_writer.add_summary(e_sum, counter)
                        if cfg.TRAIN.GENERATOR:
                            # Train the generator
                            feed_out = [self.generator_trainer,
                                        self.g_sum]
                            _, g_sum = sess.run(feed_out,
                                                feed_dict)
                            summary_writer.add_summary(g_sum, counter)
                        # save checkpoint
                        counter += 1
                        if counter % self.snapshot_interval == 0:
                            snapshot_path = "%s/%s_%s.ckpt" %\
                                             (self.checkpoint_dir,
                                              self.exp_name,
                                              str(counter))
                            fn = saver.save(sess, snapshot_path)
                            print("Model saved in file: %s" % fn)

                    img_sum = self.epoch_sum_images(sess, cfg.TRAIN.NUM_COPY)
                    summary_writer.add_summary(img_sum, counter)

                    avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                    dic_logs = {}
                    #for k, v in zip(log_keys, avg_log_vals):
                        #dic_logs[k] = v
                        # print(k, v)

                    #log_line = "; ".join("%s: %s" %
                    #                     (str(k), str(dic_logs[k]))
                    #                     for k in dic_logs)
                    #print("Epoch %d | " % (epoch) + log_line)
                    sys.stdout.flush()
                    #if np.any(np.isnan(avg_log_vals)):
                        #raise ValueError("NaN detected!")

    def q_eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        count = 0
        print('num_examples:', dataset._num_examples)
        start = np.random.randint(dataset._num_examples)
        images_batches, embeddings_batches, filenames, _ =\
        dataset.next_batch_test(self.batch_size, start, 1)
        samples = sess.run([self.latents, self.fake_latents, self.real_logit, self.fake_images_logit,\
                            self.fake_latents_logit, self.images, self.fake_images, self.fake_images_test],
                           {self.embeddings: embeddings_batches[0], self.images: images_batches})

        return samples
            
    def q_evaluate(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                if self.ALI_model_path.find('.ckpt') != -1:
                    self.init_opt()
                    print("Reading model parameters from %s" % self.ALI_model_path)
                    #print_tensors_in_checkpoint_file(self.ALI_model_path, '', False)
                    saver = tf.train.Saver(tf.global_variables())
                    saver.restore(sess, self.ALI_model_path)
                    return self.q_eval_one_dataset(sess, self.dataset.test,
                                          self.log_dir, subset='test')
                else:
                    print("Input a valid model path.")

    def save_super_images(self, images, sample_batchs, filenames,
                          sentenceID, save_dir, subset):
        # batch_size samples for each embedding
        numSamples = len(sample_batchs)
        for j in range(len(filenames)):
            s_tmp_r = '%s-1real-%dsamples/%s/real-latents/%s' %\
                (save_dir, numSamples, subset, filenames[j])
            s_tmp_f = '%s-1real-%dsamples/%s/fake-latents/%s' %\
                (save_dir, numSamples, subset, filenames[j])
            folder_r = s_tmp_r[:s_tmp_r.rfind('/')]
            if not os.path.isdir(folder_r):
                print('Make a new folder: ', folder_r)
                mkdir_p(folder_r)
            folder_f = s_tmp_f[:s_tmp_f.rfind('/')]
            if not os.path.isdir(folder_f):
                print('Make a new folder: ', folder_f)
                mkdir_p(folder_f)
        #superimage_r = [images[j]]
            #superimage_f = [images[j]]
            # cfg.TRAIN.NUM_COPY samples for each text embedding/sentence
            for i in range(len(sample_batchs)):
                #superimage_r.append(sample_batchs[i][0][j])
                #superimage_f.append(sample_batchs[i][1][j])
                scipy.misc.imsave('%s_sentence%d_%d.jpg' % (s_tmp_r, sentenceID, i), sample_batchs[i][0][j])
                scipy.misc.imsave('%s_sentence%d_%d.jpg' % (s_tmp_f, sentenceID, i), sample_batchs[i][1][j])

#superimage_r = np.concatenate(superimage_r, axis=1)
#fullpath_r = '%s_sentence%d.jpg' % (s_tmp_r, sentenceID)
#scipy.misc.imsave(fullpath_r, superimage_r)
            
            #superimage_f = np.concatenate(superimage_f, axis=1)
            #fullpath_f = '%s_sentence%d.jpg' % (s_tmp_f, sentenceID)
            #scipy.misc.imsave(fullpath_f, superimage_f)

    def eval_one_dataset(self, sess, dataset, save_dir, subset='train'):
        count = 0
        print('num_examples:', dataset._num_examples)
        while count < dataset._num_examples:
            start = count % dataset._num_examples
            images_batches, embeddings_batches, filenames, _ =\
                dataset.next_batch_test(self.batch_size, start, 1)
            print('count = ', count, 'start = ', start)
            for i in range(len(embeddings_batches)):
                samples_batches = []
                # Generate up to 16 images for each sentence,
                # with randomness from noise z and conditioning augmentation.
                for j in range(np.minimum(16, cfg.TRAIN.NUM_COPY)):
                    samples = sess.run([self.fake_images, self.fake_images_test],
                                       {self.embeddings: embeddings_batches[i], self.images: images_batches})
                    samples_batches.append(samples)
                self.save_super_images(images_batches, samples_batches,
                                       filenames, i, save_dir,
                                       subset)

            count += self.batch_size

    def evaluate(self):
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            with tf.device("/gpu:%d" % cfg.GPU_ID):
                if self.ALI_model_path.find('.ckpt') != -1:
                    self.init_opt()
                    print("Reading model parameters from %s" % self.ALI_model_path)
                    saver = tf.train.Saver(tf.global_variables())
                    saver.restore(sess, self.ALI_model_path)
                    self.eval_one_dataset(sess, self.dataset.test,
                                          self.log_dir, subset='test')
                else:
                    print("Input a valid model path.")
