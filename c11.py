import tensorflow as tf
import numpy as np
import pdb
import tensorflow.contrib.keras
import tensorflow.contrib.keras.api.keras.layers
from tensorflow.contrib.keras.api.keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Flatten,Activation,BatchNormalization,PReLU,AveragePooling2D
from tensorflow.contrib.keras import backend as K
import argparse
import time
import os

#sign-grad
@tf.RegisterGradient("QuantizeGrad")
def sign_grad(op, grad):
    input = op.inputs[0]
    cond = (input>=-1)&(input<=1)
    zeros = tf.zeros_like(grad)
    return tf.where(cond, grad, zeros)

#sign
def binary(input):
    x = input
    with tf.get_default_graph().gradient_override_map({"Sign":'QuantizeGrad'}):
        x = tf.sign(x)
    return x

#w_ = alpha*wb
def binary2(w):
    shape = w.shape.as_list()
    dim = len(shape)
    if dim == 4:
        alpha = tf.reduce_mean(tf.abs(w), axis=[0,1,2])
    elif dim == 2:
        alpha = tf.reduce_mean(tf.abs(w), axis=0)
    else:
        raise Exception("binary2 error:{}".format(dim))
    wb = binary(w)
    w_ = alpha * wb
    return w_

def binary_conv(input, ksize, o_ch, padding, strides, name, dropout=0):
    with tf.variable_scope(name):
        x = input
        x = BatchNormalization(axis=3, epsilon=1e-4, momentum=0.9, name='bn', gamma_initializer=tf.random_uniform_initializer(0,1))(x)
        xb = binary(x)

        if dropout > 0:
            xb = Dropout(dropout)(xb)

        i_ch = xb.shape.as_list()[-1]
        w = tf.get_variable('conv/kernel', shape=[ksize,ksize,i_ch,o_ch], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05))
        b = tf.get_variable('conv/bias', shape=[o_ch], initializer=tf.constant_initializer(0))
        wb = binary2(w)
        if padding > 0:
            xb = tf.pad(xb, [[0,0],[padding,padding],[padding,padding],[0,0]])
        s = tf.nn.conv2d(xb, wb, strides=[1,strides,strides,1], padding='VALID')
        x = s + b

        x = myrelu(x, 'prelu')

    norm_op = tf.assign(w, w-tf.reduce_mean(w, axis=2, keep_dims=True))
    tf.add_to_collection('norm_op', norm_op)
    clip_op = tf.assign(w, tf.clip_by_value(w, -1, 1))
    tf.add_to_collection('clip_op', clip_op)
    return x

#relu or prelu
def myrelu(x, name):
    return Activation('relu')(x)


class XNOR(object):
    def __init__(self):
        self.init_input()
        self.init_model()
        self.init_loss()

    def init_input(self):
        self.input_img = tf.placeholder(tf.float32, shape=[None,args.img_size,args.img_size,args.img_channel])
        self.input_label = tf.placeholder(tf.int32, shape=[None])
        self.label_onehot = tf.one_hot(self.input_label, args.classes)
        self.input_lr = tf.placeholder(tf.float32)

    def init_model(self):
        with tf.variable_scope('xnor'):
            x = self.input_img

            x = tf.pad(x, [[0,0],[2,2],[2,2],[0,0]])
            x = Conv2D(192, 5, padding='valid', name='conv1', kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05))(x)
            x = BatchNormalization(axis=3, epsilon=1e-4, momentum=0.9, center=False, scale=False, name='bn1')(x)
            x = Activation('relu')(x)

            x = binary_conv(x, 1, 160, 0, 1, 'conv2')
            x = binary_conv(x, 1, 96, 0, 1, 'conv3')
            x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]])
            x = MaxPooling2D((3,3), strides=2, padding='valid')(x)          

            x = binary_conv(x, 5, 192, 2, 1, 'conv4', dropout=0.5)
            x = binary_conv(x, 1, 192, 0, 1, 'conv5')
            x = binary_conv(x, 1, 192, 0, 1, 'conv6')
            x = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]])
            x = AveragePooling2D((3,3), strides=2, padding='valid')(x)

            x = binary_conv(x, 3, 192, 1, 1, 'conv7', dropout=0.5)
            x = binary_conv(x, 1, 192, 0, 1, 'conv8')
            x = BatchNormalization(axis=3, epsilon=1e-4, momentum=0.9, center=False, scale=False, name='bn8')(x)
            x = Conv2D(10, 1, padding='valid', name='conv9', kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05))(x)
            x = Activation('relu')(x)
            x = AveragePooling2D((8,8), strides=1, padding='valid')(x)

            x = Flatten()(x)
            x = Activation('softmax')(x)

            self.output = x

    def init_loss(self):
        cross_entropy = -tf.reduce_sum(self.label_onehot*tf.log(self.output+args.eps), axis=1)
        softmax_loss = tf.reduce_mean(cross_entropy)

        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #!regular_vars = [var for var in train_vars if var.name.find('kernel')!=-1]
        regular_vars = train_vars
        regularizers = tf.add_n([tf.nn.l2_loss(var) for var in regular_vars])

        self.loss = softmax_loss + args.weight_decay * regularizers

        self.opt = tf.train.AdamOptimizer(self.input_lr)
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.grad = tf.gradients(self.loss, variables)

        #mul(1-1/s[1]).mul(n)
        for i,g in enumerate(self.grad):
            if variables[i].name.find('/conv/kernel') != -1:
                shape = variables[i].shape.as_list()
                assert len(shape)==4, 'init_loss error:{}'.format(len(shape))
                n = shape[0] * shape[1] * shape[2]
                self.grad[i] = self.grad[i] * (1-1/shape[2]) * n

        bn_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = [self.opt.apply_gradients(zip(self.grad,variables))] + bn_op

        corr_pred = tf.equal(tf.cast(self.input_label,tf.int64), tf.argmax(self.output,1))
        self.acc_num = tf.reduce_sum(tf.cast(corr_pred, tf.int32))

        self.clip_op = tf.get_collection('clip_op')
        self.norm_op = tf.get_collection('norm_op') #mean center

    def get_feed(self, batch, is_train, learn_rate=1e-4):
        feed = {}
        feed[self.input_img] = batch[0]
        feed[self.input_label] = batch[1]
        feed[K.learning_phase()] = int(is_train)
        feed[self.input_lr] = learn_rate
        return feed


class DataSet(object):
    def __init__(self, filenames):
        self.data = [None,None]
        imgs = np.load(filenames[0])
        imgs = np.transpose(imgs, [0,2,3,1])
        labels = np.load(filenames[1])
        labels = labels.astype(np.int32)
        self.data[0] = imgs
        self.data[1] = labels
        
        self.data_num = self.data[0].shape[0]
        self.batch_num = int(np.ceil(self.data_num/args.batch_size))

        self.data_ixs = None
        self.data_cnt = self.data_num
        self.batch_cnt = self.batch_num

    def reset(self):
        self.data_ixs = np.arange(self.data_num)
        np.random.shuffle(self.data_ixs)

        self.data_cnt = 0
        self.batch_cnt = 0

    def get_batch(self):
        assert not self.is_end()

        beg = self.data_cnt
        end = min(beg+args.batch_size, self.data_num)
        ixs = self.data_ixs[beg:end]

        images = self.data[0][ixs]
        labels = self.data[1][ixs]

        self.data_cnt = end
        self.batch_cnt += 1

        return images,labels

    def is_end(self):
        assert (self.data_cnt==self.data_num) == (self.batch_cnt==self.batch_num)
        return self.data_cnt==self.data_num


def get_learn_rate(epoch):
    ul = np.array([120,200,240,280])
    k = np.sum(epoch>=ul)
    return args.learn_rate * 0.1**k

def run_epoch(epoch, model, sess, data, is_train=True):
    if is_train:
        train_op = model.train_op
        clip_op = model.clip_op
        norm_op = model.norm_op
    else:
        train_op = tf.no_op()
        clip_op = tf.no_op()
        norm_op = tf.no_op()

    data.reset()

    sum_acc = 0
    sum_loss = 0.0

    while not data.is_end():
        batch = data.get_batch()
        feed = model.get_feed(batch, is_train, learn_rate=get_learn_rate(epoch))

        sess.run(norm_op)
        sess.run(clip_op)

        calc_obj = [train_op, model.acc_num, model.loss]
        calc_ans = sess.run(calc_obj, feed_dict=feed)

        sum_acc += calc_ans[1]
        sum_loss += calc_ans[2]

    sess.run(norm_op)
    sess.run(clip_op)

    avg_acc = sum_acc / data.data_num
    avg_loss = sum_loss / data.batch_num

    return avg_acc,avg_loss

def train():
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement=True
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    model = XNOR()
    sess.run(tf.global_variables_initializer())

    dtrain = DataSet(['data/train_data','data/train_labels'])
    dtest = DataSet(['data/test_data','data/test_labels'])

    for epoch in range(args.epochs):
        train_acc,train_loss = run_epoch(epoch, model, sess, dtrain, is_train=True)
        test_acc,test_loss = run_epoch(epoch, model, sess, dtest, is_train=False)

        s = '[epoch {}]train-acc:{:.4} train-loss:{:.3}  test-acc:{:.4} test-loss:{:.3}'
        print(s.format(epoch,train_acc,train_loss,test_acc,test_loss))


def main():
    train()

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learn_rate', default=0.01, type=float)
    parser.add_argument('--epochs', default=320, type=int)
    parser.add_argument('--classes', default=10, type=int)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--img_channel', default=3, type=int)
    parser.add_argument('--gpu', default='0')

    global args
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if __name__ == '__main__':
    make_args()
    main()
