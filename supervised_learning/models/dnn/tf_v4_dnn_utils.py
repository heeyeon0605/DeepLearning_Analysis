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


dense1 = Dense1(input_dim=2, units=1, activation='sigmoid')
assert dense1.weights == [dense1.w, dense1.b]  # weights 는 w 와 b 를 활용하려고 배열로 넣어놓은 것
assert dense1.non_trainable_weights == []
assert dense1.trainable_weights == [dense1.w, dense1.b]


#  dense2 도 똑같이 사용해볼 수 있음.

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

    def call(self, inputs, training=None):  # training 을 할지 말지 결정할 수 있음.
        if training:
            return tf.nn.dropout(input, rate=self.rate)
        return inputs


# 레이어 생성 4 (조합)
class Combination(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation):
        super(Combination, self).__init__()
        self.dense1 = Dense1(input_dim, units, activation)
        self.dropout = Dropout1(0.1)

    def call(self, inputs, training=None):
        a = self.dense1(inputs)  # 왜 안불러지는지 알아기
        return self.dropout(a, training=training)


# 3. 모델 생성
class Model1(tf.keras.models.Model):
    def __init__(self):
        super(Model1, self).__init__()
        self.hidden_layer1 = Combination(input_dim=784, units=64, activation='relu')
        self.hidden_layer2 = Combination(input_dim=64, units=64, activation='relu')
        self.hidden_layer3 = Combination(input_dim=64, units=32, activation='relu')
        self.output_layer = Combination(input_dim=32, units=10, activation='softmax')

    def call(self, inputs, traning=None):
        hidden1 = self.hidden_layer1(inputs, training=training)
        hidden2 = self.hidden_layer1(inputs, training=training)
        hidden3 = self.hidden_layer1(inputs, training=training)
        return self.output_layer(hidden2, traning=training)


# 4. 모델 로스/옵티마이저/메트릭
loss = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


# 5. 모델 트레이닝/평가
# @ tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = Model1(x, training=True)
        loss_value = loss(y, logits)
    gradients = tape.gradient(loss_value, Model1.trainable_weights)
    optimizer.apply_gradients(zip(gradients, Model1.trainable_weights))
    accuracy.update_state(y, logits)

    if step % 100 == 0:
        print("%d 단계: " % (step))
        print('loss_value: %f' % (float(loss_value)))
        print('train accuracy: %f' % (float(accuracy.result())))

    return loss_value


def test_step(x, y):
    accuracy.reset_states()
    logits = Model1(x, training=False)
    accuracy.update_state(y, logits)


for epoch in range(10):
    print('%d번째 epoch' % (epoch + 1))
    for index, (x, y) in enumerate(train_data):
        loss_value = train_step(x, y)
        print('%d 단계 / loss_value: %f / accuracy: %f' % (index, float(loss_value), float(accuracy.result())))

    for step, (x, y) in enumerate(test_data):
        test_step(x, y)
        print('test accuracy: %f' % (float(accuracy.result())))

# v1-4까지 실제 데이터로도 돌아가는지 인해보고 정확도 낮으면 높여보기
