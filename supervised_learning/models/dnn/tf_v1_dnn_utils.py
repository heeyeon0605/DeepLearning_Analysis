import tensorflow as tf
import supervised_learning.models.data.data_utils as data_utils

# 1. 데이터 전처리
# x_train, x_test, y_train, y_test, output_dim = data_utils.load_sign_dataset()
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# mnist
# tensorflow 에 있는 데이터 셋 가져오기
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_train) 하면 (60000, 28, 28) 이고 맨 뒤에 RGB 3 이 없는 이유는 흑백이기 때문에

x_train = x_train[:].astype('float32') / 255
x_test = x_test[:].astype('float32') / 255
# y_train 이 아직 one_hot 벡터가 아님을 프린트 하면 알 수 있음. 그런데 텐서플로우가 자동으로 바꿔줘서 그대로 가도 됨.


# 2. 모델 생성
# sequential 안에는 배열을 넣을 수 있음.
# layer 을 만들고 싶으면
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=32, activation='relu'),  # layer 하나씩 만드는거. single forward
    tf.keras.layers.Dropout(rate=0.1), # overfitting, test 의 퍼센트가 너무 train 과 동떨어지지 않게 공부를 덜하는 것. 실전에도 강하도록.
    tf.keras.layers.Dense(units=32, activation='relu'),  # 대부분 보통 숫자는 32임.
    tf.keras.layers.Dropout(rate=0.1),
    tf.keras.layers.Dense(units=10, activation='softmax'),

])

# 3. 모델 트레이닝
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', matrics=['accuracy'])
# accuracy 라고 했기에 정확도로 추후에 모델 평가를 할 수 있어짐. 다른 것도 있는데 대부분 accuracy 를 사용.
# categorical 은 마지막에 softmax 써서 여러개 였을 때, binary 는 마지막에 sigmoid 를 써서 두 개 일때.
# sparse_ 를 쓸 수도 안 쓸 수도 있는데 y_train 을 one_hot 벡터로 바꾸고 싶으면 sparse_ 앞에 쓰면 알아서 바꿔줌.

# 한 에포크 안에 여러개의 미니배치가 있어서 돌아가는 건데 그걸 안에 써주면 됨.
model.fit(x_train, y_train, batch_size=100, epochs=10)  # gradient decent 에 variation 더하고 adam(gradient 구하는 알고리즘)
# 이용하면 됨

# 4. 모델 평가
model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)
# 정확도가 97쯤 나와야 정상.

# (60000, 28, 28)를 (60000, 784)로 reshape --> .reshape(60000, 784)
