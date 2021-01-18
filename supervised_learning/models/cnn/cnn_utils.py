import numpy as np


def initialize_filter(filter_size, num_of_input_channel, num_of_output_channel):
    filter = np.random.randn(filter_size, filter_size, num_of_input_channel, num_of_output_channel) * 0.01

    return filter


def zero_pad(x, pad_size):
    # x : (m * height * weight * channel)
    # x를 padding 해줘서 x_pad를 만드는
    x_pad = np.pad(x, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), 'constant', constant_values=0)

    return x_pad


def single_convolution_forward(prev_a_slice, single_filter_w, single_filter_b):
    z_element = prev_a_slice * single_filter_w
    z_element = np.sum(z_element) + single_filter_b

    return z_element


def convolution_forward(prev_a, filter_w, filter_b, hparameters):
    # 사진은 prev_a, 이동하는 칸수 filter_stride
    # pad_size 어느정도 만큼씩 추상화할 것인지
    # prev_a : (m * prev_a_height * prev_a_width * prev_a_channel)
    # filter_w : (filter_size, filter_size, prev_a_channel * z_channel)
    # filter_b : (1 * 1 * 1 * z_channel)
    # hparameters :사전 : keys = "pad_size", "filter_stride"

    (m, prev_a_height, prev_a_width, prev_a_channel) = prev_a.shape
    (filter_size, filter_size, prev_a_channel, z_channel) = filter_w.shape

    pad_size = hparameters["pad_size"]
    prev_a_pad = zero_pad(prev_a, pad_size)

    filter_stride = hparameters["filter_stride"]
    z_height = int((prev_a_height + 2 * pad_size - filter_size) / filter_stride) + 1
    z_width = int((prev_a_width + 2 * pad_size - filter_size) / filter_stride) + 1

    z = np.zeros((m, z_height, z_width, z_channel))

    for i in range(m):
        for h in range(z_height):
            for w in range(z_width):
                for c in range(z_channel):
                    height_start = h * filter_stride
                    height_end = height_start + filter_size
                    width_start = w * filter_stride
                    width_end = width_start + filter_size

                    prev_a_slice = prev_a_pad[i, height_start:height_end, width_start:width_end, :]
                    z[i, h, w, c] = single_convolution_forward(prev_a_slice, filter_w[:,:,:,c], filter_b[:,:,:,c])

    cache = prev_a, filter_w, filter_b, hparameters

    return z, cache


def relu(z):
    a = np.maximum(0, z)
    activation_cache = z

    return a, activation_cache


def leaky_relu(z):
    # 0.01z는 leacked_relu에서 0대신에 특정 위치에서 0.01z값이 들어가게 해줌
    a = np.maximum(0.01 * z, z)
    activation_cache = z

    return a, activation_cache


def pool_forward(a, hparameters, mode="max"):
    (m, a_height, a_width, a_channel) = a.shape

    pool_size = hparameters["pool_size"]
    pool_stride = hparameters["pool_stride"]

    pool_a_height = int((a_height - pool_size) / pool_stride) + 1
    pool_a_width = int((a_width - pool_size) / pool_stride) + 1
    pool_a_channel = a_channel

    pool_a = np.zeros((m, pool_a_height, pool_a_width, pool_a_channel))

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
                        pool_a[i, h, w, c] = np.max(a_slice)
                    elif mode == "average":
                        pool_a[i, h, w, c] = np.mean(a_slice)

    cache = a, hparameters

    return pool_a, cache


def max_pooling_slice_backward(da, a):
    max = np.max(a)
    mask = (a == max)
    da = mask * da

    return da


def pool_backward(d_pool_a, cache, mode="max"):
    (a, hparameters) = cache

    (m, pool_a_height, pool_a_width, pool_a_channel) = d_pool_a.shape
    (m, a_height, a_width, a_channel) = a.shape

    pool_size = hparameters["pool_size"]
    pool_stride = hparameters["pool_stride"]

    da = np.zeros(a.shape)

    for i in range(m):
        for h in range(pool_a_height):
            for w in range(pool_a_width):
                for c in range(pool_a_channel):
                    height_start = h * pool_stride
                    height_end = height_start + pool_size
                    width_start = w * pool_stride
                    width_end = width_start + pool_size

                    if mode == "max":
                        a_slice = a[i, height_start:height_end, width_start:width_end, c]
                        d_a_slice = max_pooling_slice_backward(d_pool_a[i, h, w, c], a_slice)
                        da[i, height_start:height_end, width_start:width_end, c] += d_a_slice
                    elif mode == "average":
                        d_a_slice = average_pooling_slice_backward(d_pool_a[i, h, w, c], pool_size)
                        da[i, height_start:height_end, width_start:width_end, c] += d_a_slice