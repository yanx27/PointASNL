import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import tf_util as tf_util
from pointnet_util import pointnet_sa_module
from pointasnl_util import PointASNLSetAbstraction, get_repulsion_loss

def placeholder_inputs(batch_size, num_point, use_normal=False):
    inchannel = 6 if use_normal else 3
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, inchannel))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, use_normal=False, bn_decay=None, weight_decay = None, num_class=40, adaptive_sample=False):
    """ Classification PointNet, input is BxNx3, output Bx40 """

    batch_size = point_cloud.get_shape()[0].value
    end_points = {}
    if use_normal:
        l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
        l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, 3])
    else:
        l0_xyz = point_cloud
        l0_points = point_cloud

    end_points['l0_xyz'] = l0_xyz
    as_neighbor = [12, 12] if adaptive_sample else [0, 0]

    # Set abstraction layers
    l1_xyz, l1_points = PointASNLSetAbstraction(l0_xyz, l0_points, npoint=512, nsample=32, mlp=[64,64,128], is_training=is_training, bn_decay=bn_decay,
                                                weight_decay=weight_decay, scope='layer1',as_neighbor=as_neighbor[0])
    end_points['l1_xyz'] = l1_xyz
    l2_xyz, l2_points = PointASNLSetAbstraction(l1_xyz, l1_points, npoint=128,  nsample=64, mlp=[128,128,256],  is_training=is_training, bn_decay=bn_decay,
                                                weight_decay=weight_decay, scope='layer2', as_neighbor=as_neighbor[1])
    end_points['l2_xyz'] = l1_xyz
    _, l3_points_res, _ = pointnet_sa_module(l1_xyz, l1_points, npoint=None, radius=None, nsample=None, mlp=[128,256,512], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3_1')
    _, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3_2')

    # Fully connected layers
    l3_points = tf.reshape(l3_points, [batch_size, -1])
    l3_points_res = tf.reshape(l3_points_res, [batch_size, -1])
    net = tf.concat([l3_points,l3_points_res], axis=-1)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, num_class, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, uniform_weight=0, weights_decay=1e-4):
    """ pred: B*NUM_CLASSES,
        label: B, """
    regularization_losses = [tf.nn.l2_loss(v) for v in tf.global_variables() if 'weights' in v.name]
    regularization_loss = weights_decay * tf.add_n(regularization_losses)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    if uniform_weight > 0:
        uniform_loss = get_repulsion_loss(end_points['l1_xyz'], nsample=20, radius=0.07)
    else:
        uniform_loss = classify_loss
    tf.summary.scalar('classify loss', classify_loss)
    tf.summary.scalar('uniform loss', uniform_loss)
    total_loss = classify_loss + uniform_weight * uniform_loss + regularization_loss
    tf.add_to_collection('losses', total_loss)
    return total_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        output, _ = get_model(inputs, tf.constant(True))
        print(output)
