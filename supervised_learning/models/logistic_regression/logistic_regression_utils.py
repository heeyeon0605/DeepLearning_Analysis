import numpy as np

def init_params(dim) :
    params = {}
    # 먼저 빈 dictionary 생성.
    # dim, 1 만들어진다는 얘기. 행렬에 난수를 넣는 것.
    # random 함수에서 randn은 가우시안 초기화(벨 형태)를 해서 초기화가 잘됨.

    params["w"] = np.random.randn(1, dim) * 0.01
    params["b"] = 0

    return params

def linear(w, b, x) :
    # 선형변환
    # matmul은 곱셈임.
    z = np.matmul(w, x) + b

    return z

def sigmoid(z) :
    # 비선형변환(sigmoid 함수)
    a = 1 / (1 + np.exp(- z))

    return a

def single_forward(w, b, x) :
    z = linear(w, b, x)
    a = sigmoid(z)

    return a

def forward_and_backward(w, b, x, y) :
    # y가 정답임, 1은 데이터의 개수
    m = x.shape[1]
    a = single_forward(w, b, x)

    cost = np.sum(-(y * np.log(a) + (1 - y) * np.log(1 - a))) / m
    # w 모양은 1 * dim
    # dw 모양도 1 * dim
    # x 모양은 dim * m
    # a - y 모양은 1 * m
    dw = np.matmul((a - y), x.T) / m # 헷갈림
    db = np.sum(a - y) / m

    grads = {}
    grads["dw"] = dw
    grads["db"] = db

    return cost, grads

def check_prediction(params, x, y) :
    w = params["w"]
    b = params["b"]

    a = single_forward(w, b, x)

    zeros = np.zeros(a.shape)
    zeros[a >= 0.9] = 1

    accuracy = (1 - np.mean(np.abs(zeros - y))) * 100

    return accuracy


# x = np.array([[1,2,3],[4,5,6]])
# print(np.sum(x))
# print(x.shape)


