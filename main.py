import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers

# Wczytanie danych
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalizacja danych z pixeli do wartości z zakresu 0-1
train_images = train_images/255
test_images = test_images/255

model = tf.keras.models.Sequential([
    # layers.Input(train_images.shape[1:]),
    # layers.Flatten(),
    #~104k
    # layers.Dense(128, activation='relu'),
    # layers.Dropout(0.3),
    # layers.Dense(10, activation='softmax')
    #acc: ~0.97

    layers.Input(train_images.shape[1:]),
    # Dodawanie kanału koloru do obrazu (1 - skala szarości)
    layers.Reshape((28, 28, 1)),
    # 32 filtrów, każdy o wymiarach 3x3 z funkcją aktywacji relu
    layers.Conv2D(32, (3, 3), activation='relu'),
    # wybranie najważniejszych wartości w każdym obszarze 2x2
    layers.MaxPooling2D((2, 2)),
    # Druga warstwa konwolucyjna
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # Spłaszczenie danych do wektora (Z 2 WYMIARÓW NA JEDEN)
    layers.Flatten(),
    #225,034
    layers.Dense(128, activation='relu'),
    #wyłaczanie 30% neuronów w warstwie aby uniknąć overfittingu (nadmiernego dopasowania)
    layers.Dropout(0.3),
    #10 neuronów wyjściowych (odpowiadających cyfrom 0-9) z funkcją aktywacji softmax
    layers.Dense(10, activation='softmax')
    #acc: ~0.99
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model_history = model.fit(train_images, train_labels, epochs=30, batch_size=64, validation_data=(test_images, test_labels), verbose=False)

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

probs = model.predict(test_images[64:67])
predictions = np.argmax(probs, axis=1)

for i in range(3):
    print(probs[i], "=>", predictions[i])

for i in range(64,67):
    plt.imshow(test_images[i], cmap='Greys')
    plt.show()

