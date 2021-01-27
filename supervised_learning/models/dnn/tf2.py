import tensorflow as tf

"""
# 텐서플로우 version 2
# 1. Graph 모드 : Graph 를 미리 다 build 해놓으면 나중에 계산이 쉬워짐. 속도가 빠름.(컴파일 모드)
# 2. Eagerly 모드 : 계산을 eagerly 하게(즉시) 수행할 수 있도록 도움. (즉시 실행 모드)
# - tf1은 Graph 만 모드 가지고 있고, tf2는 Graph 와 Eagerly 다 같이 있음.
# - 컴파일된 그래프의 이점을 그대로 보존함.
# - Keras 를 딥러닝의 고수준 API 를 채택함.
# - 텐서플로우는 기본적으로 수학 계산 및 미분계수를 구해주는 도구임 (텐서플로우는 미분을 도와주는 도구)

print(tf.__version__)  # 버전 체크
# 구글 서버 (google Colab)름 에 들어가서 코딩하면 개인 컴퓨터 보다 훨씬 빨

# 상수 (constant)
c = tf.constant([[5, 2],
                 [1, 3]])  # 과 np나 그냥 배열과 달리 shape 과 dtype 를 함께 보여줌.

# 텐서플로우 공식 홈페이지에 한글로 설명되어 있음. 함수도 함께 모두 설명되어 있음.
print(c)
print(c.shape)  # (2, 2)
print(c.dtype)  # <dtype : int32>
print(tf.ones(shape=(2, 2)))
print(tf.zeros(shape=(2, 2)))
print(tf.random.normal(shape=(2, 2), mean=0, stddev=1.))
print(tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype='int32'))

# 미지수 (수학에서 미지수의 의와 달리 텐서플로우에서는 무엇이든 될 수 있다는 가정 하에 시작. 일단 아무 값이나 주어져야함.)
x = tf.Variable(tf.random.normal(shape=(2, 2)))
print(x)

x.assign(tf.zeros(shape=(2, 2)))  # 모두 0으로 assign
x.assign_add(tf.ones(shape=(2, 2)))  # 모두에게 1 더하기
x.assign_sub(tf.ones(shape=(2, 2)))  # 모두에게 1 빼기

# 미지수 없는 식 세우기 (연산)
c1 = tf.random.normal(shape=(2, 2), minval=1, maxval=4, dtype='float32')
c2 = tf.random.normal(shape=(2, 2), minval=1, maxval=4, dtype='float32')
c3 = c1 + c2  # c3 = tf.square(c1) + tf.square(c2) 랑 동일한 의미
print(c3)

# 미지수 없는 식 미분계수 구하기
with tf.GradientTape() as tape:  # 미분계수 구하는 법    tape.watch(c1)  # c1을 지켜보겠다는 의미
    c2 = tf.random.normal(shape=(2, 2), minval=1, maxval=4, dtype='float32')
    c3 = c1 + c2
    dc3_dc1 = tape.gradient(c3, c1)  # c1으로 c3를 미분해달라는 의미
    print(dc3_dc1)

# 미지수 있는 식 미분계수 구하기 (미지수는 자동으로 워치됨)
with tf.GradientTape() as tape:
    y = tf.square(x) + tf.square(c1)
    # 미지수는 자동으로 워치돼서 따로 위치 코드 안적어도 됨
    dy_dx = tape.gradient(y, x)
    print(dy_dx)

# 미지수 있는 식 이계도함수 구하기(미지수는 자동으로 워치됨)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        y = tf.square(x) + tf.square(c1)
        dy_dx = inner_tape.gradient(y, x)  # 1차 미분
    d2y_dx2 = outer_tape.gradient(dy_dx, x)  # 2차 미분
    print(d2y_dx2)  # 두 번 미분 한 것을 결과로 알려줌.

# 실습 : f(x,y) = x^3*y^3 + 2x^2*y^2 + 3xy + 4, 내가 구하고 싶은 것은 f'(1, 1).
"""

x = tf.Variable(1.)
y = tf.Variable(1.)

with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        z = tf.math.pow(x, 3) * tf.math.pow(y, 3) + 2 * tf.math.pow(x, 2) * tf.math.pow(y, 2) + 3 * x * y + 4
        dz_dy = inner_tape.gradient(z, y)
    dz_dx = outer_tape.gradient(dz_dy, x)
    print(dz_dx)
