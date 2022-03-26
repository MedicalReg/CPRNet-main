import tensorflow as tf
import tflearn
from keras.layers.convolutional import UpSampling3D
from tflearn.layers.normalization import batch_normalization
from tflearn.initializations import normal
from .utils import Network, ReLU, LeakyReLU, Sigmoid, Avg_pool_3d
# from .ext.neuron import layers as nrn_layers
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Conv3DTranspose, Concatenate, AveragePooling3D, multiply, ZeroPadding3D, Cropping3D, PReLU, concatenate, GaussianNoise, Lambda, add
from keras.initializers import RandomNormal
from keras.layers import Layer
from .utils import Network

from .spatial_transformer import Dense3DSpatialTransformer, Fast3DTransformer, coordxyz


def convolve(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return tflearn.layers.conv_3d(inputLayer, outputChannel, kernelSize, strides=stride,
                                  padding='same', activation='linear', bias=True, scope=opName, reuse=reuse, weights_init=weights_init)

def convolveBN(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return batch_normalization(tflearn.layers.conv_3d(inputLayer, outputChannel, kernelSize, strides=stride,
                                  padding='same', activation='linear', bias=True, scope=opName, reuse=reuse, weights_init=weights_init))


def convolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, stddev=1e-2, reuse=False):
    return ReLU(convolve(opName, inputLayer,
                         outputChannel,
                         kernelSize, stride, stddev=stddev, reuse=reuse),
                opName+'_rectified')


def convolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(convolve(opName, inputLayer,
                              outputChannel,
                              kernelSize, stride, stddev, reuse),
                     alpha, opName+'_leakilyrectified')

def convolveBNLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(batch_normalization(convolve(opName, inputLayer,
                              outputChannel,
                              kernelSize, stride, stddev, reuse),
                     alpha, opName+'_leakilyrectified'))


def upconvolve(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return tflearn.layers.conv.conv_3d_transpose(inputLayer, outputChannel, kernelSize, targetShape, strides=stride,
                                                 padding='same', activation='linear', bias=False, scope=opName, reuse=reuse, weights_init=weights_init)

def upconvolveBN(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False, weights_init='uniform_scaling'):
    return batch_normalization(tflearn.layers.conv.conv_3d_transpose(inputLayer, outputChannel, kernelSize, targetShape, strides=stride,
                                                 padding='same', activation='linear', bias=False, scope=opName, reuse=reuse, weights_init=weights_init))



def upconvolveReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, stddev=1e-2, reuse=False):
    return ReLU(upconvolve(opName, inputLayer,
                           outputChannel,
                           kernelSize, stride,
                           targetShape, stddev, reuse),
                opName+'_rectified')


def upconvolveLeakyReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(upconvolve(opName, inputLayer,
                                outputChannel,
                                kernelSize, stride,
                                targetShape, stddev, reuse),
                     alpha, opName+'_rectified')

def upconvolveLeakyBNReLU(opName, inputLayer, outputChannel, kernelSize, stride, targetShape, alpha=0.1, stddev=1e-2, reuse=False):
    return LeakyReLU(batch_normalization(upconvolve(opName, inputLayer,
                                outputChannel,
                                kernelSize, stride,
                                targetShape, stddev, reuse),
                     alpha, opName+'_rectified'))

class MRFCM(Network):
    def __init__(self, name='multi_receptive_field correlation_module', filters = 16, **kwargs):
        super().__init__(name, **kwargs)
        self.filters = filters

    def build(self, x):
        branch_1 = convolveLeakyReLU(
            'branch_a',  x,  self.filters, 1, 1)
        branch_2 = convolveLeakyReLU(
            'branch_b_1',  x,  self.filters,  1, 1)
        branch_2 = convolveLeakyReLU(
            'branch_b_2', branch_2, self.filters, 3, 1)
        branch_3 = Avg_pool_3d(x)
        branch_3 = convolveLeakyReLU(
            'branch_c', branch_3, self.filters, 3, 1)
        branch_4 = convolveLeakyReLU('branch_d_1', x, self.filters, 1, 1)
        branch_4 = convolveLeakyReLU('branch_d_2', branch_4, self.filters, 3, 1)
        branch_4 = convolveLeakyReLU('branch_d_3', branch_4, self.filters, 3, 1)
        correlation_maps = tf.concat([branch_1, branch_2, branch_3, branch_4], axis=4)
        return correlation_maps

class DFEM(Network):
    def __init__(self,
                name='DFEM',
                num_levels=4,
                is_warp = True,
                flow_residual = True,
                flow_sum = True,
                refined_flow = True,
                **kwargs):
        super().__init__(name, **kwargs)
        self.num_levels = num_levels
        self.transformer = Dense3DSpatialTransformer()
        self.MRF_CMs = []
        self.flow_layers = []
        self.is_warp = is_warp
        self.flow_residual = flow_residual
        self.flow_sum = flow_sum
        self.refined_flow = refined_flow
        shape = [128, 64, 32, 16, 8][:num_levels]
        filters = [16, 32, 64, 128, 256][:num_levels]
        for i in range(num_levels):
            self.MRF_CMs.append(MRFCM('multi_receptive_field_correlation_module_'+str(i), filters=filters[i]))
            self.flow_layers.append(build_flow_layers('flow_layers_' + str(i), filters = filters[i], shape = [shape[i], shape[i], shape[i]]))

    def warp(self, feature, flow):
        flow = flow*20.0
        _, _, _, _, channel = feature.shape.as_list()
        warped_lists = []
        for i in range(channel):
            warped = self.transformer([tf.expand_dims(feature[..., i], 4), flow])
            warped_lists.append(warped)
        return tf.concat(warped_lists, axis=4)

    def build(self, feature_pyramid1, feature_pyramid2):
        """Run the model"""
        flow = None
        context = None
        flow_up = None
        flow_stage1 = None
        flow_stage2 = None
        flow_stage3 = None
        flow_stage4 = None
        flow_stage5 = None

        feature1_1 = None
        feature1_2 = None
        feature1_3 = None
        feature1_4 = None

        feature2_1 = None
        feature2_2 = None
        feature2_3 = None
        feature2_4 = None

        for i, (features1, features2) in reversed(list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]):
            # features1 means fixed image, features2 means moving image
            level = i - 1
            if level == 3:
                feature1_1 = features1
                feature2_1 = features2
            if level == 2:
                feature1_2 = features1
                feature2_2 = features2
            if level == 1:
                feature1_3 = features1
                feature2_3 = features2
            if level == 0:
                feature1_4 = features1
                feature2_4 = features2

            if self.is_warp:
                if flow_up is not None:
                    warped2 = self.warp(features2, flow_up)
                else:
                    warped2 = features2
            else:
                warped2 = features2

            concat_feature1 = tf.concat([features1, warped2], axis=4)
            correlation_module = self.MRF_CMs[level] # di
            correlation_maps = correlation_module(concat_feature1) # di

            if context == None:
                concat_feature2 = correlation_maps
            else:
                concat_feature2 = tf.concat([correlation_maps, context, flow_up], axis=4)
                # next will go to RDFEM

            flow_layer_module = self.flow_layers[level]
            flow, context = flow_layer_module(concat_feature2)
            # ci and transformation field
            if level == 3:
                flow_stage1 = flow
          
            if flow_up is not None:
                if self.flow_residual:
                    if self.flow_sum:
                        flow = flow + flow_up
                    else:
                        flow = (self.transformer([flow_up*20.0, flow*20.0]) + flow*20.0)/20.0

            shape = feature_pyramid2[level].shape.as_list()
            flow_up = upconvolve('upsample'+str(level), flow, 3, 4, 2, shape[1:4])
            # sum and multiply 2
            if level == 3:
                flow_stage2 = flow_up
            if level == 2:
                flow_stage3 = flow_up
            if level == 1:
                flow_stage4 = flow_up

        if self.refined_flow:
            concat_feature3 = tf.concat([context, flow_up], axis=4)
            refined_flow = convolve('refine_pred', concat_feature3, 3, 3, 1)
        else:
            refined_flow = flow_up
        flow_stage5 = refined_flow

        results = {'f1_1': feature1_1, 'f2_1': feature2_1, 'f1_2': feature1_2, 'f2_2': feature2_2,
                   'f1_3': feature1_3, 'f2_3': feature2_3, 'f1_4': feature1_4, 'f2_4': feature2_4,
                   'flow_1': flow_stage1*20.0, 'flow_2': flow_stage2*20.0, 'flow_3': flow_stage3*20.0,
                   'flow_4': flow_stage4*20.0, 'flow_5': flow_stage5*20.0, }
        return results

class build_flow_layers(Network):
    def __init__(self,
        name,
        filters,
        shape,
        **kwargs
    ):
       super().__init__(name,**kwargs) 
       self.shape = shape
       self.filters = filters
    def build(self, x):
        flow = convolve('pred', x, 3, 3, 1)
        context = upconvolveLeakyReLU(
            'deconv4', x, self.filters, 4, 2, self.shape)
        return flow, context

class FeaturePyramid(Network):
    def __init__(self,name='PFP', num_levels=3, **kwargs):
        super().__init__(name, **kwargs)
        self.num_levels = num_levels 

    def build(self, x):
        features = []
        features.append(x)
        # filters = ((1, 16), (1, 32), (1, 32), (1, 32), (1, 32),
        #         (1, 32))[:self.num_levels]
        filters = ((1, 4), (1, 8), (1, 16), (1, 16), (1, 16),
                   (1, 16))[:self.num_levels]
        feature = x
        for level, (num_layers, num_filters) in enumerate(filters):
            for i in range(num_layers):
                stride = 2
                stride = 1 if num_layers != 1 and i == 0 else 2
                feature = convolveLeakyReLU(
                    'conv_'+str(level),
                    feature,
                    int(num_filters),
                    3,
                    stride)
                features.append(feature)
        return features

class CPRNet(Network):
    def __init__(self, name = 'CPRNet', num_levels=4, **kwargs):
        super().__init__(name, **kwargs)
        self.feature_pyramid = FeaturePyramid(num_levels=num_levels)
        self.flow_predict = DFEM(num_levels=num_levels)
    
    def build(self, fixed, moving):
        feature_pyramid1 = self.feature_pyramid(fixed)
        feature_pyramid2 = self.feature_pyramid(moving)
        results = self.flow_predict(feature_pyramid1, feature_pyramid2)
        return results


