import numpy as np

def init_params(dim_of_layers):
    # w 초기화 : 가우시안, Xavier initialization, He initialization ...
        # 초기화를 잘해야 나중에 최저점을 찾는데 고생을 안함.
    params = {}
    num_of_layers = len(dim_of_layers)

    for i in range(1, num_of_layers) :
        params["w" + str(i)] = np.random.rand(dim_of_layers[i], dim_of_layers[i-1]) * 0.01
        params["b" + str(i)] = np.zeros((dim_of_layers[i], 1))

    return params

def linear(w, b, a):
    z = np.matmul(w, a) + b
    linear_cache = w, b, a

    return z, linear_cache

# activation function(relu, leaked relu, sigmoid, softmax)

def relu(z):
    a = np.maximum(0, z)
    activation_cache = z

    return a, activation_cache

def leacky_relu(z):
    # 0.01z는 leacked_relu에서 0대신에 특정 위치에서 0.01z값이 들어가게 해줌
    a = np.maximum(0.01 * z, z)
    activation_cache = z

    return a, activation_cache

def sigmoid(z):
    a = 1 / (1 + np.exp(- z))
    activation_cache = z

    return a, activation_cache

def softmax(z):
    # z = z - np.max(z, axis=0, keepdim=True)
    a = np.exp(z) / np.sum(np.exp(z), axis=0, keepdim=True)
    activation_cache = z

    return a, activation_cache

def single_forward(w, b, a, activation):
    z, linear_cache = linear(w, b, a)

    if activation == "relu" :
        a, activation_cache = relu(z)
    elif activation == "leacked_relu" :
        a, activation_cache = leacked_relu(z)
    elif activation == "sigmoid" :
        a, activation_cache = sigmoid(z)
    elif activation == "softmax" :
        a, activation_cache = softmax(z)

    linear_activation_cache = linear_cache, activation_cache

    return a, linear_activation_cache

def forward(params, x, activation, last_activation, num_of_layers):
    a = x
    forward_cache = []

    for i in range(1, num_of_layers):
        if i != (num_of_layers - 1): # i가 마지막이 아닌 경우
            a, linear_activation_cache = single_forward(params["w" + str(i)], params["b" + str(i)], a, activation)
        else:
            a, linear_activation_cache = single_forward(params["w" + str(i)], params["b" + str(i)], a, last_activation)
        forward_cache.append(linear_activation_cache)

    return a, forward_cache

def cross_entropy(a, y):
    #베르누이 확률분포

    m = y.shape[1]

    cost = np.sum(-(y * np.log(a))) / m

    return cost

def mean_square_error(a, y):
    #가우시안 확률분포

    m = y.shape[1]

    cost = np.sum(np.square(a - y)) / (2 * m)

    return cost

def cross_entropy_gradient(a, y):
    m = y.shape[1]
    da = -(y / a) / m

    return da

def mean_square_error_gradient(a, y):
    m = y.shape[1]
    da = (a - y) / m

    return da

def relu_gradient(da, activation_cache):
    z = activation_cache

    dz=np.ones(z.shape)
    dz[z < 0]= 0
    dz = da * dz

    return dz

def leaky_relu_gradient(da, activation_cache):
    z = activation_cache

    dz=np.ones(z.shape)
    dz[z < 0]= 0.01
    dz = da * dz

    return dz

def sigmoid_gradient(da, activation_cache):
    z = activation_cache
    a = 1 / (1 +np.exp(-z))

    dz = da * a * (1 - a)

    return dz

def softmax(da, activation_cache):
    z = activation_cache
    z = z -np.max(z, axis=0, keepdims=True)
    a = np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)

    (r, m) = da.shape
    dz = np.zeros(da.shape)
    for k in range(m):
        middle_matrix = np.zeros((r, r))
        for i in range(r):
            for j in range(r):
                if i == j:
                    middle_matrix[i, j] = a[i, k] * (1 - a[i, k])
                else:
                    middle_matrix[i, j] = -(a[i, k] * a[j, k])
        dz[:, k] = np.matmul(middle_matrix, da[:, k])

    return dz

def linear_gradient(dz, linear_cache):
    w, b, a = linear_cache

    dw = np.matmul(a, dz.T).T
    db = np.mean(dz, axis=1, keepdims=True)
    da = np.matmul(w.T, dz)

    return dw, db, da

def single_backward(da, linear_activation_cache, activation):

def backward(a, y, forward_cache, cost_function, activation, last_activation, num_of_layers):

def forward_and_backward(pams, x, y, activation, last_activation, cost_function, num_of_layers):

def update_parameters(params, grads, learning_rate, num_of_layers):

def predict(params, x_test, y_test, activation, last_activation, num_of_layers):
