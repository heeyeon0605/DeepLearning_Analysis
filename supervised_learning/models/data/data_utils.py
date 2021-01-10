import h5py
import numpy as np
import os
import definitions

def load_sign_dataset() :
    train_dataset = h5py.File(
        os.path.join(definitions.ROOT_DIR, "supervised_learning/models/data/data_set/signs_train.h5"), "r")
    # print(train_dataset.keys())
    test_dataset = h5py.File(
        os.path.join(definitions.ROOT_DIR, "supervised_learning/models/data/data_set/signs_test.h5"), "r")

    # 배열을 넘피 객체로 만들어 변수에 저장
    x_train = np.array(train_dataset["train_set_x"])
    y_train = np.array(train_dataset["train_set_y"])
    # print(x_train.shape)  # (m = 1080, 64, 64, 3)
    # print(y_train.shape)  # (1080,)
    m = y_train.shape[0]

    y_train = y_train.reshape(m, -1) #(1080, 1)로 reshape

    x_test = np.array(test_dataset["test_set_x"])
    y_test = np.array(test_dataset["test_set_y"])
    m1 = y_test.shape[0]

    y_test = y_test.reshape(m1, -1)

    return x_train, x_test, y_train, y_test

def flatten(x, y) :
    m = x.shape[0]
    x = x.reshape(m, -1).T # dim x m 으로 바꿔줌, 64643 을 바꿔주려고 하는 것
    y = y.reshape(m, -1).T # dim x m 으로 바꿔줌

    return x, y

def centralized(x) :
    x = x / 255 # 모두 0과 1사이로 바뀌도록 해줌

    return x

def one_hot_encoding(y) :
    m = y.shape[1]
    one_hot = np.zeros((m, 1))

    for i in range(m) :
        if y[0][i] == 3 :
            one_hot[i][0] = 1

    return one_hot.T

