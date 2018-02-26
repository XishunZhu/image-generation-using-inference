import tensorflow as tf
import prettytensor as pt


from misc.config import cfg

from misc.custom_ops import *

# from mcb.compact_bilinear_pooling import compact_bilinear_pooling_layer


# TODO:
# 1. Reimplement Discriminator to take as conditionning input the latent variable used for image generation
# 2. Implement Encoder from images (image_shape) to latent space (lf_dim)
# 3. Experiment different architectures for encoder ( current encoder = VGG16
#    bottleneck

class CondALI(object):
    def __init__(self, image_shape):
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.network_type = cfg.ALI.NETWORK_TYPE
        self.batch_discrimination = cfg.ALI.BATCH_DISCRIMINATION
        self.num_kernels = cfg.ALI.BATCH_DISCRIMINATION_KERNEL_DIM
        self.image_shape = image_shape
        self.gf_dim = cfg.ALI.GF_DIM
        self.df_dim = cfg.ALI.DF_DIM
        self.ef_dim = cfg.ALI.EMBEDDING_DIM # Conditional augmenting text embedding dimension
        self.lf_dim = cfg.ALI.LF_DIM # Latent variable dimension
        self.zf_dim = cfg.Z_DIM # Added noise dimension
        
        assert self.lf_dim == self.ef_dim + self.zf_dim

        self.image_shape = image_shape
        self.s = image_shape[0]
        self.s2, self.s4, self.s8, self.s16 =\
            int(self.s / 2), int(self.s / 4), int(self.s / 8), int(self.s / 16)

        # Since D is only used during training, we build a template
        # for safe reuse the variables during computing loss for fake/real/wrong images
        # We do not do this for G,
        # because batch_norm needs different options for training and testing
        if cfg.ALI.NETWORK_TYPE == "default":
            with tf.variable_scope("d_i_net"):
                self.d_encode_img_template = self.d_encode_image()
            with tf.variable_scope("d_l_net"):
                self.d_latent_template = self.latent_variable()
            with tf.variable_scope("d_net"):
                self.discriminator_template = self.discriminator()
        elif cfg.ALI.NETWORK_TYPE == "simple":
            with tf.variable_scope("d_net"):
                self.d_encode_img_template = self.d_encode_image_simple()
                self.d_latent_template = self.latent_variable()
                self.discriminator_template = self.discriminator()
        else:
            raise NotImplementedError

    # g-net
    def generate_condition(self, c_var):
        conditions =\
            (pt.wrap(c_var).
             flatten().
             custom_fully_connected(self.ef_dim * 2).
             apply(leaky_rectify, leakiness=0.2))
        mean = conditions[:, :self.ef_dim]
        log_sigma = conditions[:, self.ef_dim:]
        return [mean, log_sigma]
    
    def generator(self, l_var):
        node1_0 =\
            (pt.wrap(l_var).
             flatten().
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             fc_batch_norm().
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]))
        node1_1 = \
            (node1_0.
             custom_conv2d(self.gf_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(tf.nn.relu))

        node2_0 = \
            (node1.
             # custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2_1 = \
            (node2_0.
             custom_conv2d(self.gf_dim * 1, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 1, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())
        node2 = \
            (node2_0.
             apply(tf.add, node2_1).
             apply(tf.nn.relu))

        output_tensor = \
            (node2.
             # custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             # custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor

    def generator_simple(self, l_var):
        output_tensor =\
            (pt.wrap(l_var).
             flatten().
             custom_fully_connected(self.s16 * self.s16 * self.gf_dim * 8).
             reshape([-1, self.s16, self.s16, self.gf_dim * 8]).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s8, self.s8, self.gf_dim * 4], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s8, self.s8]).
             # custom_conv2d(self.gf_dim * 4, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s4, self.s4, self.gf_dim * 2], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s4, self.s4]).
             # custom_conv2d(self.gf_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0, self.s2, self.s2, self.gf_dim], k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s2, self.s2]).
             # custom_conv2d(self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(tf.nn.relu).
             custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
             # apply(tf.image.resize_nearest_neighbor, [self.s, self.s]).
             # custom_conv2d(3, k_h=3, k_w=3, d_h=1, d_w=1).
             apply(tf.nn.tanh))
        return output_tensor

    def get_generator(self, l_var):
        if cfg.ALI.NETWORK_TYPE == "default":
            return self.generator(l_var)
        elif cfg.ALI.NETWORK_TYPE == "simple":
            return self.generator_simple(l_var)
        else:
            raise NotImplementedError

    def latent_variable(self):
        template = (pt.template("input").
                    custom_fully_connected(self.lf_dim).
                    apply(leaky_rectify, leakiness=0.2))
        if cfg.ALI.BATCH_DISCRIMINATION:
            template = template.minibatch_discrimination(num_kernels=self.num_kernels)
        return template

    def d_encode_image(self):
        node1_0 = \
            (pt.template("input").
             custom_conv2d(self.df_dim, k_h=4, k_w=4).
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=4, k_w=4).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 4, k_h=4, k_w=4).
             conv_batch_norm().
             custom_conv2d(self.df_dim * 8, k_h=4, k_w=4).
             conv_batch_norm())
        node1_1 = \
            (node1_0.
             custom_conv2d(self.df_dim * 2, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 2, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             custom_conv2d(self.df_dim * 8, k_h=3, k_w=3, d_h=1, d_w=1).
             conv_batch_norm())

        node1 = \
            (node1_0.
             apply(tf.add, node1_1).
             apply(leaky_rectify, leakiness=0.2))

        return node1

    def discriminator(self):
        template = \
            (pt.template("input").
             custom_conv2d(self.df_dim * 8, k_h=1, k_w=1, d_h=1, d_w=1).
             conv_batch_norm().
             apply(leaky_rectify, leakiness=0.2).
             # custom_fully_connected(1))
             custom_conv2d(1, k_h=self.s16, k_w=self.s16, d_h=self.s16, d_w=self.s16))
        return template

    def get_discriminator(self, x_var, l_var):
        x_code = self.d_encode_img_template.construct(input=x_var)
        l_code = self.d_latent_template.construct(input=l_var)
        l_code = tf.expand_dims(tf.expand_dims(l_code, 1), 1)
        l_code = tf.tile(l_code, [1, self.s16, self.s16, 1])
        #if cfg.TRAIN.MCB:
        #    x_l_code = compact_bilinear_pooling_layer(x_code, l_code, outputdim = 16000, sum_pool=False)
        x_l_code = tf.concat([x_code, l_code], 3)
        return self.discriminator_template.construct(input=x_l_code)
    
    def get_intermediate_latent_features(self, l_var):
        return self.d_latent_template.construct(input=l_var)
    
    def encoder_last_layer(self, bottlenecks):
        output_tensor =\
            (pt.wrap(bottlenecks).
            flatten().
            fully_connected(self.ef_dim, activation_fn=None))
        return output_tensor
             
    def get_encoder(self, bottlenecks=None, images=None):
        if bottlenecks is not None:
            return self.encoder_last_layer(bottlenecks)
