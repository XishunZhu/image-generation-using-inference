from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.DATASET_NAME = 'flowers'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.GPU_ID = 0
__C.Z_DIM = 100

# Demo/test options
__C.TEST = edict()
__C.TEST.LR_IMSIZE = 64
__C.TEST.HR_IMSIZE = 256
__C.TEST.NUM_COPY = 16
__C.TEST.BATCH_SIZE = 1
__C.TEST.NUM_COPY = 16
__C.TEST.CAPTION_PATH = ''
__C.TEST.PRETRAINED_MODEL = 'ckt_logs/flowers/_2017_11_27_04_09_27/model_64000.ckpt'


# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = False

__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.NUM_COPY = 4
__C.TRAIN.MAX_EPOCH = 200
__C.TRAIN.SNAPSHOT_INTERVAL = 2000
__C.TRAIN.ENCODER_FEATURE_MATCHING = True
__C.TRAIN.PRETRAINED_STACKGAN_MODEL = 'models/birds_model_164000.ckpt'
#__C.TRAIN.PRETRAINED_STACKGAN_MODEL = 'models/flowers_model_130000.ckpt'
__C.TRAIN.PRETRAINED_INCEPTION_MODEL = 'models/'
__C.TRAIN.PRETRAINED_RESNET_MODEL = 'models/resnet_v1_50.ckpt'
__C.TRAIN.USE_PRETRAINED_ALI = True
#__C.TRAIN.PRETRAINED_ALI_MODEL = 'ckt_logs/birds/_2017_11_13_04_28_46/model_138000.ckpt'
__C.TRAIN.PRETRAINED_ALI_MODEL = 'ckt_logs/flowers/_2017_11_27_04_09_27/model_64000.ckpt'
__C.TRAIN.PRETRAINED_EPOCH = 600

__C.TRAIN.SUPERVISED = True
__C.TRAIN.DISCRIMINATOR = True
__C.TRAIN.DISCRIMINATOR_IMAGES = True
__C.TRAIN.DISCRIMINATOR_LATENTS = True
__C.TRAIN.DISCRIMINATOR_FUSION = True
__C.TRAIN.ENCODER = True
__C.TRAIN.GENERATOR = True
__C.TRAIN.ENCODER_PERIOD = 1

__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.ENCODER_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.LR_DECAY_EPOCH = 50

__C.TRAIN.NUM_EMBEDDING = 4
__C.TRAIN.COND_AUGMENTATION = True
__C.TRAIN.B_WRONG = True
__C.TRAIN.MCB = False


__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0

__C.TRAIN.FINETUNE_LR = False
__C.TRAIN.FT_LR_RATIO = 0.1

# Modal options
__C.ALI = edict()
__C.ALI.EMBEDDING_DIM = 128
__C.ALI.DF_DIM = 64
__C.ALI.GF_DIM = 128
__C.ALI.LF_DIM = 228
__C.ALI.NETWORK_TYPE = 'default'
__C.ALI.BATCH_DISCRIMINATION = False
__C.ALI.BATCH_DISCRIMINATION_KERNEL_DIM = 128

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

