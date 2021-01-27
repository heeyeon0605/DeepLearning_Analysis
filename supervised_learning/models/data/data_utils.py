import h5py
import numpy as np
import os
import definitions


def load_sign_dataset():
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

    y_train = y_train.reshape(m, -1)  # (1080, 1)로 reshape

    x_test = np.array(test_dataset["test_set_x"])
    y_test = np.array(test_dataset["test_set_y"])
    m1 = y_test.shape[0]

    y_test = y_test.reshape(m1, -1)
    output_dim = np.array(train_dataset["list_classes"]).shape[0]  # 복습

    return x_train, x_test, y_train, y_test, output_dim


def flatten(x, y):
    m = x.shape[0]
    x = x.reshape(m, -1).T  # dim x m 으로 바꿔줌, 64,64,3 을 바꿔주려고 하는 것
    y = y.reshape(m, -1).T  # dim x m 으로 바꿔줌
    input_dim = x.shape[0]
    return x, y, input_dim


def centralized(x):
    x = x / 255  # 모두 0과 1사이로 바뀌도록 해줌

    return x


def one_hot_encoding(y, output_dim):  # 복습
    m = y.shape[1]
    one_hot = np.zeros((m, output_dim))

    for i in range(m):
        one_hot[i][y[0][i]] = 1
    return one_hot.T


def generate_random_mini_batches(x, y, size_of_mini_batch):
    # mini batch 한 번 움직일 때의 한 세트
    # 에포크 데이터 전체
    # 순서에 의한걸 방지하기 위해 셔플

    m = x.shape[1]  # 트레이닝 개수, 현재 가로는 1080
    mini_batches = []
    permutation = np.random.permutation(m)  # 0부터 1079의 랜덤 배열
    shuffled_x = x[:, permutation]  # [:]는 모든 행, 그 뒤에껀 열
    shuffled_y = y[:, permutation]

    num_of_complete_mini_batches = m // size_of_mini_batch
    for i in range(0, num_of_complete_mini_batches):
        mini_batch_x = shuffled_x[:, i * size_of_mini_batch: i * size_of_mini_batch + size_of_mini_batch]
        mini_batch_y = shuffled_y[:, i * size_of_mini_batch: i * size_of_mini_batch + size_of_mini_batch]
        mini_batch = mini_batch_x, mini_batch_y
        mini_batches.append(mini_batch)

    if m % size_of_mini_batch == 0:
        return mini_batches

    mini_batch_x = shuffled_x[:, num_of_complete_mini_batches * size_of_mini_batch:m]
    mini_batch_y = shuffled_y[:, num_of_complete_mini_batches * size_of_mini_batch:m]
    mini_batch = mini_batch_x, mini_batch_y
    mini_batches.append(mini_batch)

    return mini_batches
