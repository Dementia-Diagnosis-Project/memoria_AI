# model.py
import tensorflow as tf

class InceptionResNetV2Loader(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.backbone = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
        self.backbone.trainable = True

    def call(self, inputs):
        return self.backbone(inputs)

class DiagnoseModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.concat = tf.keras.layers.Concatenate()
        self.dense = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')

    def call(self, inputs):
        gender, age_group, image_number = inputs
        gender = tf.expand_dims(gender, -1)
        age_group = tf.expand_dims(age_group, -1)
        image_number = tf.expand_dims(image_number, -1)
        x = self.concat([gender, age_group, image_number])
        return self.dense(x)

class FinalModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.cnn = InceptionResNetV2Loader()
        self.tabular = DiagnoseModel()
        self.concat = tf.keras.layers.Concatenate()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.fc2 = tf.keras.layers.Dense(128, activation='relu')
        self.output_layer = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, inputs):
        img, tab = inputs
        x1 = self.cnn(img)
        x1 = tf.keras.layers.GlobalAveragePooling2D()(x1)
        x2 = self.tabular(tab)
        x = self.concat([x1, x2])
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output_layer(x)
