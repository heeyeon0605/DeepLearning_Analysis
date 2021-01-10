import supervised_learning.models.logistic_regression.logistic_regression_utils as utils
import matplotlib.pyplot as plt
import supervised_learning.models.data.data_utils as data_utils


def logistic_regression_model(x_train, x_test, y_train, y_test, learning_rate, num_of_iteration):
    # 훈련할 때 쓸 것 train, 테스트할 때 쓸 것 test, 뮤를 learning_rate, 몇 번 반복할 것인지 num_of_iteration
    dim = x_train.shape[0]
    params = utils.init_params(dim)

    costs = []

    for i in range(num_of_iteration):
        cost, grads = utils.forward_and_backward(params["w"], params["b"], x_train, y_train)
        params["w"] = params["w"] - grads["dw"] * learning_rate
        params["b"] = params["b"] - grads["db"] * learning_rate

        if i % 100 == 0:
            costs.append(cost)
            print(cost)

    result_train = utils.check_prediction(params, x_train, y_train)
    print("accuracy of train : ", result_train)

    result_test = utils.check_prediction(params, x_test, y_test)
    print("accuracy of test : ", result_test)

    plt.figure()
    plt.plot(costs)
    plt.xlabel("반복횟수")
    plt.ylabel("cost")
    plt.title("cost graph")
    plt.show()

    return params


x_train, x_test, y_train, y_test = data_utils.load_sign_dataset()
x_train, y_train = data_utils.flatten(x_train, y_train)
x_train = data_utils.centralized(x_train)

x_test, y_test = data_utils.flatten(x_test, y_test)
x_test = data_utils.centralized(x_test)

y_train = data_utils.one_hot_encoding(y_train)
y_test = data_utils.one_hot_encoding(y_test)

learning_rate = 0.001
num_of_iteration = 10000

logistic_regression_model(x_train, x_test, y_train, y_test, learning_rate, num_of_iteration)


