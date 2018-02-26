#!/bin/sh
#!/usr/bin/python
python -u feat_extract.py --network="vgg_16" --checkpoint='pretrained-models/vgg_16.ckpt' --image_path='data/images' --layer_names="vgg_16/fc7" --batch_size=5
