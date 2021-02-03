import tensorflow as tf
import supervised_learning.models.data.data_utils as data_utils

# 1. 데이터 전처리

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:].astype('float32') / 255
x_test = x_test[:].astype('float32') / 255
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_data = train_data.shuffle(buffer_size=1000).batch(100)
test_data = test_data.shuffle(buffer_size=1000).batch(100)


# 2. 레이어 생성

# 레이어 생성 1
class Dense1(tf.keras.layers):
    def __init__(self, input_dim, units, activation):
        super(Dense1, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.random_normal_initializer(mean=0, stddev=1),
            shape=(input_dim, units),
            dtype='float32',
            trainable=True  # 미분 가능하도록
        )
        self.b = tf.Variable(
            initial_value=tf.zeros_initializer(),
            shape=(units,),
            dtype='float32',
            trainable=True  # 미분 가능하도록
        )
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.linalg.matmul(inputs, self.w) + self.b)


# 레이어 생성 2 (build 메소드)
class Dense2(tf.keras.layers):
    def __init__(self, input_dim, units, activation):
        super(Dense2, self).__init__()
        self.input_dim = input_dim
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    # tf.variable 말고 원래 부모클래스 되어있는 것 처럼 add.weight 사용하는 경우.
    def build(self, input_shape):
        super().build(input_shape)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True  # 미분 가능하도록
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True  # 미분 가능하도록
        )

    def call(self, inputs):
        return self.activation(tf.linalg.matmul(inputs, self.w) + self.b)


# 레이어 생성 3
class Dropout1(tf.keras.layers):
    def __init__(self, rate):
        super(Dropout1, self).__init__()
        self.rate = rate

    def call(self, input, training=None):
        if training:
            return tf.nn.dropout(input, rate=self.rate)
        return input

'''
# 레이어 생성 4 (조합)
class Combination(tf.keras.layers):
    def __init__(self):
        super(Combination, self).__init__()
        self.dense = Dense1(input_dim, units, activation)
        self.dropout = Dropout1(0.5)

    def call(self, input, training=None):
        a = self.dense(input)
        return self.dropout(a, training=training)
'''

# 3. 모델 생성
input = tf.keras.Input(shape=(28, 28))
input_flatten = tf.keras.layers.Flatten(input_shape=(28, 28))(input)
hidden1 = tf.keras.layers.Dense2(64, activation='relu')(input_flatten)
hidden2 = tf.keras.layers.Dense2(64, activation='relu')(hidden1)
hidden3 = tf.keras.layers.Dense2(64, activation='relu')(hidden1 + hidden2)
output = tf.keras.layers.Dense2(10, activation='softmax')(hidden3)
model = tf.keras.Model(input, output)

# 과제는 v4에 3. 모델 생성 만들어보기, v1-3까지 실제 데이터로도 돌아가는지 인해보고 정확도 낮으면 높여보기
