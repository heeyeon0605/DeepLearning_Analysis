import tensorflow as tf
import numpy as np


# channels_of_layers = [3, 5, 10, 15, 20, 25] 처음에는 RGB니까 3으로 고정되어 있음.
def initialize_filter_parameter(filter_size, num_of_input_channel, num_of_output_channel, channels_of_layers):
    num_of_layers = len(channels_of_layers)
    filter_parameters = {}
    for i in range(1, num_of_layers):
        filter_parameters["w" + str(i)] = tf.Variable(name="w" + str(i),
                                                      shape=(filter_size, filter_size, channels_of_layers[i - 1],
                                                             channels_of_layers[i]))
        filter_parameters["b" + str(i)] = tf.Variable(name="b" + str(i),
                                                      shape=(filter_size, filter_size, channels_of_layers[i - 1],
                                                             channels_of_layers[i])) # shape 이거 맞나?
    return filter_parameters


def single_convolution_forward(prev_a_slice, single_filter_w, single_filter_b):
    z_element = prev_a_slice * single_filter_w
    z_element = tf.math.reduce_sum(z_element) + single_filter_b  # 다 더한다는 의미

    return z_element


def zero_pad(x, pad_size):
    x_pad = tf.pad(x, tf.constant[[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    # tf.pad 에 디펄트가 constant_values=0 이라서 쓸 필요가 없음. 이런건 외울 필요 없이 그때마다 찾아봄.

    return x_pad


def convolution_forward(prev_a, filter_w, filter_b, hparameters):
    (m, prev_a_height, prev_a_width, prev_a_channel) = prev_a.shape
    (filter_size, filter_size, prev_a_channel, z_channel) = filter_w.shape

    pad_size = hparameters["pad_size"]
    prev_a_pad = zero_pad(prev_a, pad_size)

    filter_stride = hparameters["filter_stride"]
    z_height = int((prev_a_height + 2 * pad_size - filter_size) / filter_stride) + 1
    z_width = int((prev_a_width + 2 * pad_size - filter_size) / filter_stride) + 1

    z = tf.zeros((m, z_height, z_width, z_channel))

    for i in range(m):
        for h in range(z_height):
            for w in range(z_width):
                for c in range(z_channel):
                    height_start = h * filter_stride
                    height_end = height_start + filter_size
                    width_start = w * filter_stride
                    width_end = width_start + filter_size

                    prev_a_slice = prev_a_pad[i, height_start:height_end, width_start:width_end, :]
                    z[i, h, w, c] = single_convolution_forward(prev_a_slice, filter_w[:, :, :, c], filter_b[:, :, :, c])

    cache = prev_a, filter_w, filter_b, hparameters

    return z, cache


def relu(z):
    a = tf.math.maximum(0, z)
    activation_cache = z

    return a, activation_cache


def softmax(z):
    a = tf.exp(z) / tf.reduce_sum(np.exp(z), axis=0, keepdims=True)

    return a


def pool_forward(a, hparameters, mode="max"):
    (m, a_height, a_width, a_channel) = a.shape

    pool_size = hparameters["pool_size"]
    pool_stride = hparameters["pool_stride"]

    pool_a_height = int((a_height - pool_size) / pool_stride) + 1
    pool_a_width = int((a_width - pool_size) / pool_stride) + 1
    pool_a_channel = a_channel

    pool_a = tf.zeros((m, pool_a_height, pool_a_width, pool_a_channel))

    for i in range(m):
        for h in range(pool_a_height):
            for w in range(pool_a_width):
                for c in range(pool_a_channel):
                    height_start = h * pool_stride
                    height_end = height_start + pool_size
                    width_start = w * pool_stride
                    width_end = width_start + pool_size

                    a_slice = a[i, height_start:height_end, width_start:width_end, :]

                    if mode == "max":
                        pool_a[i, h, w, c] = tf.reduce_max(a_slice)
                    elif mode == "average":
                        pool_a[i, h, w, c] = tf.reduce_mean(a_slice)

    cache = a, hparameters

    return pool_a, cache


def flatten(a):
    return tf.reshape(a, shape=[1, 1, -1])

def forward(x, filter_parameters, num_of_layers, hparameters):
    a = x
    for i in range(1, num_of_layers):
        z = convolution_forward(a, filter_parameters["w" + str(i)], filter_parameters["b" + str(i)], hparameters # 더써야됨)
        a = relu(z)
        a = pool_forward(a, hparameters, mode="max")
    a = flatten(a)
    # add dnn code here ...

    return a

def cross_entropy(a, y):
    #베르누이 확률분포 (cost 함수 미분한 것)

    m = y.shape[1]

    cost = tf.reduce_sum(-(y * np.log(a))) / m

    return cost

def compute_cost(a, y, cost_function, params, num_of_layers):
    cost = 0
    if cost_function == 'cross_entropy':
        cost = cross_entropy(a, y)
    elif cost_function == 'mean_square_error':
        cost = mean_square_error(a, y)

    return cost

def mean_square_error(a, y):
    #가우시안 확률분포 (cost 함수 미분한 것)

    m = y.shape[1]

    cost = tf.reduce_sum(np.square(a - y)) / (2 * m)

    return cost