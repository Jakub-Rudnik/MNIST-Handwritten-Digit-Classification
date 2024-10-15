import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers

# Wczytanie danych
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalizacja danych z pixeli do wartości z zakresu 0-1
train_images = train_images/255
test_images = test_images/255

# print(train_images.shape)
# print(train_labels.shape)
#
# plt.imshow(train_images[0], cmap='Greys')
# plt.show()

model = tf.keras.models.Sequential([
        layers.Input(train_images.shape[1:]),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

# one-hot representation
# model.compile(optimizer='adam', loss='categorical_crossentropy')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.summary()

# One-hot encoding and training
# labels_onehot_train = tf.one_hot(train_labels, 10)
# model.fit(train_images, labels_onehot_train)

model_history = model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels), verbose=False)

# wykresy historii uczenia
plt.plot(model_history.history['loss'], label='train')
plt.plot(model_history.history['val_loss'], label='val')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(model_history.history['accuracy'], label='train')
plt.plot(model_history.history['val_accuracy'], label='val')
plt.ylabel('accuracy')
plt.legend()
plt.show()


# testowanie modelu
model.evaluate(test_images, test_labels)

probs = model.predict(test_images[:5])
preds = np.argmax(probs, axis=1)

for i in range(3):
    print(probs[i], "=>", preds[i])
    plt.imshow(test_images[i], cmap='Greys')
    plt.show()

