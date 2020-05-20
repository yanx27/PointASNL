import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import tf_util
from pointasnl_util import PointASNLSetAbstraction, PointASNLDecodingLayer, get_repulsion_loss

def placeholder_inputs(batch_size, num_point, feature_channel=0):
    inchannel = 3 + feature_channel
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, inchannel))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None, weight_decay=None, feature_channel=0):
    """ Semantic segmentation PointNet, input is B x N x3 , output B x num_class """
    end_points = {}
    num_point = point_cloud.get_shape()[1].value
    if feature_channel > 0:
        l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
        l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, feature_channel])
    else:
        l0_xyz = point_cloud
        l0_points = point_cloud

    end_points['l0_xyz'] = l0_xyz
    num_points = [num_point//8, num_point//32, num_point//128, num_point//256]
    # Feature encoding layers
    l1_xyz, l1_points = PointASNLSetAbstraction(l0_xyz, l0_points, npoint=num_points[0], nsample=32, mlp=[32,32,64], is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay, scope='layer1', as_neighbor=8)
    l2_xyz, l2_points = PointASNLSetAbstraction(l1_xyz, l1_points, npoint=num_points[1], nsample=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay,scope='layer2', as_neighbor=4)
    l3_xyz, l3_points = PointASNLSetAbstraction(l2_xyz, l2_points, npoint=num_points[2], nsample=32, mlp=[128,128,256], is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay, scope='layer3', as_neighbor=0)
    l4_xyz, l4_points = PointASNLSetAbstraction(l3_xyz, l3_points, npoint=num_points[3], nsample=32, mlp=[256,256,512], is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay, scope='layer4', as_neighbor=0)
    end_points['l1_xyz'] = l1_xyz

    # Feature decoding layers
    l3_points = PointASNLDecodingLayer(l3_xyz, l4_xyz, l3_points, l4_points, 16, [512,512], is_training, bn_decay, weight_decay, scope='fa_layer1')
    l2_points = PointASNLDecodingLayer(l2_xyz, l3_xyz, l2_points, l3_points, 16, [256,256], is_training, bn_decay, weight_decay, scope='fa_layer2')
    l1_points = PointASNLDecodingLayer(l1_xyz, l2_xyz, l1_points, l2_points, 16, [256,128], is_training, bn_decay, weight_decay, scope='fa_layer3')
    l0_points = PointASNLDecodingLayer(l0_xyz, l1_xyz, l0_points, l1_points, 16, [128,128,128], is_training, bn_decay, weight_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay, weight_decay=weight_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, weight_decay=weight_decay, scope='fc2')

    return net, end_points


def get_loss(pred, label, end_points, smpw=1.0, uniform_weight=0.01, weights_decay=1e-4, radius=0.07):
    """
    pred: BxNxC,
    label: BxN,
    smpw: BxN
    """
    regularization_losses = [tf.nn.l2_loss(v) for v in tf.global_variables() if 'weights' in v.name]
    regularization_loss = weights_decay * tf.add_n(regularization_losses)
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    uniform_loss = get_repulsion_loss(end_points['l1_xyz'], nsample=20, radius=radius)
    weight_reg = tf.add_n(tf.get_collection('losses'))
    classify_loss_mean = tf.reduce_mean(classify_loss, name='classify_loss_mean')
    total_loss = classify_loss_mean + weight_reg + uniform_weight * uniform_loss + regularization_loss
    tf.summary.scalar('classify loss', classify_loss)
    tf.summary.scalar('total loss', total_loss)
    return total_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10, 1.0)
        print(net)

