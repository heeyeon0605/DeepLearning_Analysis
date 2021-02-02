import numpy as np

def single_lstm_forward(aPrev, mPrev, xt, parameters):
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wu = parameters["Wu"]
    bu = parameters["bu"]
    Wcm = parameters["Wcm"]
    bcm = parameters["Wcm"]
    Wca = parameters["Wca"]
    bca = parameters["Wca"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    (n_x, m) = xt.shape
    (n_y, n_a) = Wy.shape

    # a_prev / xt 쌓기
    concat = np.zeros(((n_a + n_x), m))
    concat[:n_a, :] = aPrev
    concat[n_a:, :] = xt

    # forget gate
    fg = sigmoid(np.matmul(Wf, concat) + bf)

    # input gate
    ug = sigmoid(np.matmul(Wu, concat) + bu)
    cm = np.tanh(np.matmul(Wcm, concat) + bcm)

    # memory
    mt = fg * mPrev + ug * cm

    # candidate activation
    ca = rnnAndLstmUtils.sigmoid(np.matmul(Wca, concat) + bca)

    # at
    at = ca * np.tanh(mt)

    return at, mt


# mt가 현재 메모리, ug는 업데이트해서 기억, at가 최종 결론(기억면에서)
# mt랑 at가 최종 값들임.
# a_prev, m_prev 가 처음에는 지정되어 있지 않으니까 0으로 시작하는 것.

# lstm single forward만 적고 forward는 안해도 됨