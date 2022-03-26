import argparse
import re
import json
import skimage.io
import skimage.exposure
import skimage.io
import numpy as np
import os
import SimpleITK as sitk
import skimage.filters

import tensorflow as tf
import tflearn
import network

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-f', '--fixed', type=str, default=None,
                    help='Specifies the fixed image')
parser.add_argument('-m', '--moving', type=str, default=None,
                    help='Specifies the moving image')
parser.add_argument('-o', '--output', type=str, default='./output',
                    help='Specifies the output directory')
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('--fast_reconstruction', action='store_true')
parser.add_argument('--net_args', type=str, default=None)
args = parser.parse_args(args=[])

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def RenderFlow(flow, coef=5, channel=(0, 1, 2), thresh=1):
    #flow = flow[:, :, 64]
    im_flow = np.stack([flow[:, :, c] for c in channel], axis=-1)
    # im_flow = 0.5 + im_flow / coef
    im_flow = np.abs(im_flow)
    im_flow = np.exp(-im_flow / coef)
    im_flow = im_flow * thresh
    # im_flow = 1 - im_flow / 20
    return im_flow


def test(fixed_image, moving_image, framework):

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES))
        checkpoint = args.checkpoint
        saver.restore(sess, checkpoint)
        print('Successfully load weights!')
        tflearn.is_training(False, session=sess)
        print('Start testing!')

        output_path = args.output
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(output_path)

        fid = "fixed_pictrure"
        mid = "moving_picture"


        keys = ['image_fixed', 'image_moving', 'warped_moving', 'real_flow', 'f1_1', 'f2_1', 'f1_2', 'f2_2', 'f1_3', 'f2_3', 'f1_4', 'f2_4',
                'flow_1', 'flow_2', 'flow_3', 'flow_4']
        gen = [{'id1': fid, 'id2': mid,
                'voxel1': np.tile(np.reshape(fixed_image, [1, 128, 128, 128, 1]), (2, 1, 1, 1, 1)),
                'voxel2': np.tile(np.reshape(moving_image, [1, 128, 128, 128, 1]), (2, 1, 1, 1, 1))}]
        results = framework.validate2(sess, gen, keys=keys, summary=False)

        return results



def find_checkpoint_step(checkpoint_path, target_steps=None):
    pattern = re.compile(r'model-(\d+).index')
    checkpoints = []
    for f in os.listdir(checkpoint_path):
        m = pattern.match(f)
        if m:
            steps = int(m.group(1))
            checkpoints.append((-steps if target_steps is None else abs(
                target_steps - steps), os.path.join(checkpoint_path, f.replace('.index', ''))))
    return min(checkpoints, key=lambda x: x[0])[1]


def RenderFlow(flow, coef=5, channel=(0, 1, 2), thresh=1):
    #flow = flow[:, :, 64]
    im_flow = np.stack([flow[:, :, c] for c in channel], axis=-1)
    # im_flow = 0.5 + im_flow / coef
    im_flow = np.abs(im_flow)
    im_flow = np.exp(-im_flow / coef)
    im_flow = im_flow * thresh
    # im_flow = 1 - im_flow / 20
    return im_flow


def RenderFlow2(flow, coef=10, channel=(0, 1, 2), thresh=1):
    # flow = flow[:, :, 64]
    im_flow = np.stack([flow[:, :, c] for c in channel], axis=-1)
    # im_flow = 0.5 + im_flow / coef
    im_flow = np.abs(im_flow)
    im_flow = np.exp(-im_flow / coef)
    im_flow = im_flow * thresh
    # im_flow = 1 - im_flow / 20
    return im_flow



if __name__ == '__main__':
    test()
