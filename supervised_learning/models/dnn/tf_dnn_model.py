import tensorflow as tf
import matplotlib.pyplot as plt
import supervised_learning.models.dnn.tf_dnn_utils as dnn_utils
import supervised_learning.models.data.data_utils as data_utils


def dnn_model(x_train, y_train, x_test, y_test, activation, last_activation, learning_rate, cost_function, batch_size,
              num_of_iterations, dims_of_layers):
    params = dnn_utils.init_params(dims_of_layers)
    mini_batches = data_utils.generate_random_mini_batches(x_train, y_train, batch_size)
    num_of_layers = len(dims_of_layers)
    costs = []

    for i, mini_batch in enumerate(mini_batches):
        mini_batch_x, mini_batch_y = mini_batch
        for j in range(num_of_iterations):
            cost, grads = dnn_utils.forward_and_backward(params, mini_batch_x, mini_batch_y, activation,
                                                         last_activation, cost_function, num_of_layers)
            params = dnn_utils.update_parameters(params, grads, learning_rate, num_of_layers)
            if j % 100 == 0:
                print(cost)
                costs.append(cost)

    accuracy_train = dnn_utils.predict(params, x_train, y_train, activation, last_activation, num_of_layers)
    print(accuracy_train)
    accuracy_test = dnn_utils.predict(params, x_test, y_test, activation, last_activation, num_of_layers)
    print(accuracy_test)

    plt.figure()
    plt.plot(costs)
    plt.xlabel("반복횟수")
    plt.ylabel("cost")
    plt.title("cost graph")
    plt.show()
    return params


x_train, x_test, y_train, y_test, output_dim = data_utils.load_sign_dataset()
x_train, y_train, input_dim = data_utils.flatten(x_train, y_train)
x_train = data_utils.centralized(x_train)

x_test, y_test, input_dim = data_utils.flatten(x_test, y_test)
x_test = data_utils.centralized(x_test)

y_train = data_utils.one_hot_encoding(y_train, output_dim)
y_test = data_utils.one_hot_encoding(y_test, output_dim)

learning_rate = 0.001
num_of_iteration = 10000

dnn_model(x_train, y_train, x_test, y_test, "relu", "softmax", learning_rate=0.01, cost_function="cross_entropy",
          batch_size=108, num_of_iterations=1000, dims_of_layers=[input_dim, 16, 16, output_dim])
