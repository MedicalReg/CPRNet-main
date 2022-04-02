import tensorflow as tf
import tflearn
import numpy as np
from keras.layers import Conv3D
import scipy
from scipy import signal
from scipy import ndimage as nd

from .utils import Network
from .base_networks import CPRNet
from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer, coordxyz
from .trilinear_sampler import TrilinearSampler
import keras.backend as K
import tensorflow.keras.backend as K

# import neuron.layers as nrn_layers


def mask_metrics(seg1, seg2):
    ''' Given two segmentation seg1, seg2, 0 for background 255 for foreground.
    Calculate the Dice score 
    $ 2 * | seg1 \cap seg2 | / (|seg1| + |seg2|) $
    and the Jacc score
    $ | seg1 \cap seg2 | / (|seg1 \cup seg2|) $
    '''
    sizes = np.prod(seg1.shape.as_list()[1:])
    seg1 = tf.reshape(seg1, [-1, sizes])
    seg2 = tf.reshape(seg2, [-1, sizes])
    seg1 = tf.cast(seg1 > 128, tf.float32)
    seg2 = tf.cast(seg2 > 128, tf.float32)
    dice_score = 2.0 * tf.reduce_sum(seg1 * seg2, axis=-1) / (
            tf.reduce_sum(seg1, axis=-1) + tf.reduce_sum(seg2, axis=-1))
    union = tf.reduce_sum(tf.maximum(seg1, seg2), axis=-1)
    return (dice_score, tf.reduce_sum(tf.minimum(seg1, seg2), axis=-1) / tf.maximum(0.01, union))


class DeformableRegistrationNetworks(Network):
    default_params = {
        'weight': 1,
        'raw_weight': 1,
        'reg_weight': 1,
        'anatomy_weight': 0,
        'folding_weight': 1e-4
    }

    def __init__(self, name, framework,
                 base_network, n_cascades, rep=1,
                 reg_factor=1.0, regularization=False, warp_gradient=True,
                 fast_reconstruction=False, warp_padding=False, finetune=None,
                 **kwargs):
        super().__init__(name)
        self.reg_factor = reg_factor
        self.base_network = eval(base_network)
        self.stems = sum([[(self.base_network(), {'raw_weight': 0})] * rep
                          for i in range(n_cascades)], [])

        self.stems[-1][1]['raw_weight'] = 1

        for _, param in self.stems:
            for k, v in self.default_params.items():
                if k not in param:
                    param[k] = v
        print(self.stems)

        self.framework = framework
        self.warp_gradient = warp_gradient
        self.fast_reconstruction = fast_reconstruction
        self.grid = coordxyz()
        self.regularization = regularization

        self.reconstruction = Fast3DTransformer(
            warp_padding) if fast_reconstruction else Dense3DSpatialTransformer(warp_padding)
        self.trilinear_sampler = TrilinearSampler()
        self.gaussian_kernel = tf.reshape(
            tf.constant(self.kernel3d(5, 1), tf.float32), [5, 5, 5, 1, 1])
        self.ones_kernel = tf.ones([5, 5, 5, 1, 1])

    @property
    def trainable_variables(self):
        return list(set(sum([stem.trainable_variables for stem, params in self.stems], [])))

    @property
    def data_args(self):
        return dict()

    def build(self, img1, img2):
        stem_results = []
        for stem, params in self.stems[0:]:
            if len(stem_results) == 0:
                stem_result = stem(img1, img2)
                # stem_result['agg_flow'] = stem_result['flow']
                if self.regularization:
                    stem_result['agg_flow'] = self.flow_regularizer(stem_result['flow_5'])
                else:
                    stem_result['agg_flow'] = stem_result['flow_5']
                stem_result['warped'] = self.reconstruction(
                    [img2, stem_result['agg_flow']])
                stem_results.append(stem_result)

        # unsupervised learning with simlarity loss and regularization loss
        for stem_result, (stem, params) in zip(stem_results, self.stems):
            if params['raw_weight'] > 0:
                stem_result['raw_loss'] = self.similarity_loss(
                    img1, stem_result['warped'])
            if params['reg_weight'] > 0:
                stem_result['reg_loss'] = self.regularize_loss(
                    stem_result['flow_5']) * self.reg_factor
            if params['folding_weight'] > 0:
                stem_result['folding_loss'] = self.folding_loss(stem_result['agg_flow'])
            stem_result['loss'] = sum(
                [stem_result[k] * params[k.replace('loss', 'weight')] for k in stem_result if k.endswith('loss')])

        ret = {}

        flow = stem_results[-1]['agg_flow']

        warped = stem_results[-1]['warped']

        simi_loss = sum([r['loss'] * params['weight'] for r, (stem, params) in zip(stem_results, self.stems)])

        loss = simi_loss

        ret.update({'loss': tf.reshape(loss, (1,)),
                    'simi_loss': tf.reshape(simi_loss, (1,)),
                    'raw_loss': tf.reshape(stem_results[0]['raw_loss'], (1,)),
                    'reg_loss': tf.reshape(stem_results[0]['reg_loss'], (1,)),
                    'image_fixed': img1,
                    'image_moving': img2,
                    'warped_moving': warped,
                    'f1_1': stem_results[0]['f1_1'],
                    'f2_1': stem_results[0]['f2_1'],
                    'f1_2': stem_results[0]['f1_2'],
                    'f2_2': stem_results[0]['f2_2'],
                    'f1_3': stem_results[0]['f1_3'],
                    'f2_3': stem_results[0]['f2_3'],
                    'f1_4': stem_results[0]['f1_4'],
                    'f2_4': stem_results[0]['f2_4'],
                    'flow_1': stem_results[0]['flow_1'],
                    'flow_2': stem_results[0]['flow_2'],
                    'flow_3': stem_results[0]['flow_3'],
                    'flow_4': stem_results[0]['flow_4'],
                    'real_flow': flow})
        return ret

    def similarity_loss(self, img1, warped_img2):
        sizes = np.prod(img1.shape.as_list()[1:])
        flatten1 = tf.reshape(img1, [-1, sizes])
        flatten2 = tf.reshape(warped_img2, [-1, sizes])

        if self.fast_reconstruction:
            _, pearson_r, _ = tf.user_ops.linear_similarity(flatten1, flatten2)
        else:
            mean1 = tf.reshape(tf.reduce_mean(flatten1, axis=-1), [-1, 1])
            mean2 = tf.reshape(tf.reduce_mean(flatten2, axis=-1), [-1, 1])
            var1 = tf.reduce_mean(tf.square(flatten1 - mean1), axis=-1)
            var2 = tf.reduce_mean(tf.square(flatten2 - mean2), axis=-1)
            cov12 = tf.reduce_mean(
                (flatten1 - mean1) * (flatten2 - mean2), axis=-1)
            pearson_r = cov12 / tf.sqrt((var1 + 1e-6) * (var2 + 1e-6))

        raw_loss = 1 - pearson_r
        raw_loss = tf.reduce_sum(raw_loss)
        return raw_loss

    def MSE(self, img1, warped_img2):
        sizes = np.prod(img1.shape.as_list()[1:])
        flatten1 = tf.reshape(img1, [-1, sizes])
        flatten2 = tf.reshape(warped_img2, [-1, sizes])
        raw_loss = tf.reduce_mean(tf.square(flatten1 - flatten2), axis=-1)
        return raw_loss

    """
    Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
    """

    def NCC_loss(self, I, J, win=9, eps=1e-5):  # local_NCC
        ndims = 3
        batch = I.shape.as_list()[0]
        win_size = win
        win_ = [win] * ndims
        weight_win_size = win
        # weight = tf.ones([weight_win_size, weight_win_size, weight_win_size, 1])

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J
        # compute filters
        # compute local sums via convolution

        I_sum = Conv3D(filters=1,
                       kernel_size=win,
                       strides=1,
                       padding='valid',
                       kernel_initializer=tf.ones_initializer(),
                       use_bias=False,
                       )(I)
        I_sum = tf.stop_gradient(I_sum)
        J_sum = Conv3D(filters=1,
                       kernel_size=win,
                       strides=1,
                       padding='valid',
                       kernel_initializer=tf.ones_initializer(),
                       )(J)
        J_sum = tf.stop_gradient(J_sum)
        I2_sum = Conv3D(filters=1,
                        kernel_size=win,
                        strides=1,
                        padding='valid',
                        kernel_initializer=tf.ones_initializer(),
                        )(I2)
        I2_sum = tf.stop_gradient(I2_sum)
        J2_sum = Conv3D(filters=1,
                        kernel_size=win,
                        strides=1,
                        padding='valid',
                        kernel_initializer=tf.ones_initializer(),
                        use_bias=False,
                        )(J2)
        J2_sum = tf.stop_gradient(J2_sum)
        IJ_sum = Conv3D(filters=1,
                        kernel_size=win,
                        strides=1,
                        padding='valid',
                        kernel_initializer=tf.ones_initializer(),
                        use_bias=False,
                        )(IJ)
        IJ_sum = tf.stop_gradient(IJ_sum)
        # compute cross correlation
        win_size = np.prod(win_)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + eps)

        return - tf.reduce_mean(cc)

    def MSE_loss(self, img1, warped_img2):
        sizes = np.prod(img1.shape.as_list()[1:])
        flatten1 = tf.reshape(img1, [-1, sizes])
        flatten2 = tf.reshape(warped_img2, [-1, sizes])
        raw_loss = tf.reduce_mean(tf.square(flatten1 - flatten2), axis=-1)
        return raw_loss

    def Get_Ja(self, flow):
        grid = self.grid(flow)
        flow = flow + grid
        D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
        D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
        D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])

        D1 = (D_x[..., 0]) * ((D_y[..., 1]) * (D_z[..., 2]) - D_z[..., 1] * D_y[..., 2])
        D2 = (D_x[..., 1]) * ((D_y[..., 0]) * (D_z[..., 2]) - D_y[..., 2] * D_x[..., 0])
        D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1]) * D_z[..., 0])
        return D1 - D2 + D3

    def regularize_loss(self, flow):
        ret = ((tf.nn.l2_loss(flow[:, 1:, :, :] - flow[:, :-1, :, :]) +
                tf.nn.l2_loss(flow[:, :, 1:, :] - flow[:, :, :-1, :]) +
                tf.nn.l2_loss(flow[:, :, :, 1:] - flow[:, :, :, :-1])) / np.prod(flow.shape.as_list()[1:5]))
        return ret

    def flow_regularizer(self, flow):
        J = self.Get_Ja(flow)
        J = tf.pad(J,((0,0),(0,1),(0,1),(0,1)),'constant',constant_values = 1)
        J = tf.expand_dims(J, -1)
        N = tf.cast(J <= 0.0, tf.float32)
        N = tf.nn.convolution(N, self.ones_kernel, padding='SAME')
        N = tf.cast((N > 0), tf.float32)
        P = tf.cast((N <= 0), tf.float32)
        flow_x = tf.nn.convolution(tf.expand_dims(flow[..., 0],-1), self.gaussian_kernel, padding='SAME')
        flow_y = tf.nn.convolution(tf.expand_dims(flow[..., 1],-1), self.gaussian_kernel, padding='SAME')
        flow_z = tf.nn.convolution(tf.expand_dims(flow[..., 2],-1), self.gaussian_kernel, padding='SAME')

        flow_x = flow_x * N + tf.expand_dims(flow[..., 0],-1) * P
        flow_y = flow_y * N + tf.expand_dims(flow[..., 1],-1) * P
        flow_z = flow_z * N + tf.expand_dims(flow[..., 2],-1) * P

        flow = tf.concat([flow_x, flow_y, flow_z], axis=4)
        return flow

    def folding_loss(self, flow):
        jacobian_det = self.Get_Ja(flow)
        return tf.reduce_sum(0.5 * (tf.abs(jacobian_det) - jacobian_det))

    def jacobian_det(self, flow):
        _, var = tf.nn.moments(tf.linalg.det(tf.stack([
            flow[:, 1:, :-1, :-1] - flow[:, :-1, :-1, :-1] +
            tf.constant([1, 0, 0], dtype=tf.float32),
            flow[:, :-1, 1:, :-1] - flow[:, :-1, :-1, :-1] +
            tf.constant([0, 1, 0], dtype=tf.float32),
            flow[:, :-1, :-1, 1:] - flow[:, :-1, :-1, :-1] +
            tf.constant([0, 0, 1], dtype=tf.float32)
        ], axis=-1)), axes=[1, 2, 3])
        return tf.sqrt(var)