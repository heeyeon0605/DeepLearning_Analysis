import tensorflow as tf
import supervised_learning.models.data.data_utils as data_utils

# 1. 데이터 전처리
# x_train, x_test, y_train, y_test, output_dim = data_utils.load_sign_dataset()
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# mnist
# tensorflow 에 있는 데이터 셋 가져오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:].astype('float32') / 255
x_test = x_test[:].astype('float32') / 255


# 2. 모델 생성

# Resnet (현재 자율주행 차의 사물 인식 알고리즘으로 쓰이는 중)
input = tf.keras.Input(shape=(28, 28))
input_flatten = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
hidden1 = tf.keras.layers.Dense(64, activation='relu')(input_flatten)
hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense(64, activation='relu')(hidden1 + hidden2)
output = tf.keras.layers.Dense(10, activation='softmax')(hidden3)
model = tf.keras.Model(input, output)

# 3. 모델 트레이닝
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', matrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=10)

# 4. 모델 평가
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)
# 정확도가 97쯤 나와야 정상.