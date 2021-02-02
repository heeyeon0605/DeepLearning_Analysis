import tensorflow as tf


def init_params(dim_of_layers):
    params = {}
    num_of_layers = len(dim_of_layers)

    for i in range(1, num_of_layers):
        params["w" + str(i)] = tf.Variable(tf.random.normal(shape=(dim_of_layers[i-1], dim_of_layers[i]), mean=0, stddev=1.), name="w" + str(i))
        params["b" + str(i)] = tf.Variable(tf.zeros(shape=(dim_of_layers[i], 1)), "w" + str(i))
        # 여기 문제가 있는 듯
    print(params)

    return params


def linear(w, b, a):
    z = tf.linalg.matmul(tf.linalg.matrix_transpose(w), a) + b

    return z


def relu(z):
    a = tf.math.maximum(0, z)

    return a


def leaky_relu(z):
    a = tf.math.maximum(0.01 * z, z)

    return a


def sigmoid(z):
    a = 1 / (1 + tf.math.exp(- z))

    return a


def softmax(z):
    a = tf.math.exp(z) / tf.math.reduce_sum(tf.math.exp(z), axis=0, keepdims=True)

    return a


def single_forward(w, b, a, activation):
    z = linear(w, b, a)

    if activation == "relu":
        a = relu(z)
    elif activation == "leaky_relu":
        a = leaky_relu(z)
    elif activation == "sigmoid":
        a = sigmoid(z)
    elif activation == "softmax":
        a = softmax(z)

    return a


def forward(params, x, activation, last_activation, num_of_layers):
    a = x

    for i in range(1, num_of_layers):
        if i != (num_of_layers - 1):  # i가 마지막이 아닌 경우
            a = single_forward(params["w" + str(i)], params["b" + str(i)], a, activation)
        else:
            a = single_forward(params["w" + str(i)], params["b" + str(i)], a, last_activation)

    return a


def cross_entropy(a, y):
    m = y.shape[1]

    cost = tf.math.reduce_sum(-(y * tf.math.log(a))) / m

    return cost # 여기나(여기도 아닐 것 같음)


def mean_square_error(a, y):
    m = y.shape[1]

    cost = tf.math.reduce_sum(tf.square(a - y)) / (2 * m)

    return cost


def compute_cost(a, y, cost_function, params, num_of_layers):
    cost = 0
    if cost_function == 'cross_entropy':
        cost = cross_entropy(a, y)
    elif cost_function == 'mean_square_error':
        cost = mean_square_error(a, y)

    return cost # 여기나 (여기는 아닐 것 같음)


def backward(cost, params, num_of_layers):
    grads = {}

    with tf.GradientTape(persistent=True) as tape:
        for i in reversed(range(1, num_of_layers)):
            grads["dw" + str(i)] = tape.gradient(cost, sources=params["w" + str(i)])
            grads["db" + str(i)] = tape.gradient(cost, sources=params["w" + str(i)])
    # print(grads) # 이거 미분계수가 왜 None 인지 알기 !!!!!
    return grads


def forward_and_backward(params, x, y, activation, last_activation, cost_function, num_of_layers):
    a = forward(params, x, activation, last_activation, num_of_layers)
    cost = compute_cost(a, y, cost_function, params, num_of_layers)
    grads = backward(cost, params, num_of_layers)
    return cost, grads


def update_parameters(params, grads, learning_rate, num_of_layers):
    for i in range(1, num_of_layers):
        params["w" + str(i)] = params["w" + str(i)].assign_sub(learning_rate * grads["dw" + str(i)])
        params["b" + str(i)] = params["b" + str(i)].assign_sub(learning_rate * grads["dw" + str(i)])

    # forward and backward, backward에서 논 타입이 나왔으니까 에러가 뜨는 듯.
    return params


def predict(params, x_test, y_test, activation, last_activation, num_of_layers):
    a, _ = forward(params, x_test, activation, last_activation, num_of_layers)
    zeros = tf.zeros(a.shape)
    zeros[a >= 0.75] = 1

    accuracy = tf.math.reduce_mean(tf.math.reduce_all(zeros == y_test, axis=0, keepdims=True)) * 100

    return accuracy
