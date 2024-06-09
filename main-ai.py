import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

# 1. Загрузка данных MNIST
trainingNumbers = 60000
validatingNumbers = 10000

height, width, depth = 28, 28, 1
amountOfClasses = 10

(trainImages, trainLabels), (validationImages, validationLabels) = mnist.load_data()

# 2. Предварительная обработка данных
trainImages = trainImages.reshape((trainingNumbers, height * width)).astype('float32') / 255
validationImages = validationImages.reshape((validatingNumbers, height * width)).astype('float32') / 255

trainLabels = to_categorical(trainLabels, amountOfClasses)
validationLabels = to_categorical(validationLabels, amountOfClasses)

# 3. Создание модели Sequential
batchSize = 128  # in each iteration, we consider 128 training examples at once
numEpochs = 20  # we iterate twenty times over the entire training set
hiddenSize = 512  # there will be 512 neurons in both hidden layers

model = keras.Sequential([
    layers.Dense(hiddenSize, activation='relu', input_shape=(784,)),
    layers.Dense(hiddenSize, activation='relu'),
    layers.Dense(amountOfClasses, activation='softmax')
])

# 4. Компиляция модели
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Обучение модели
model.fit(trainImages, trainLabels,
          epochs=numEpochs, batch_size=batchSize,
          verbose=1, validation_split=0.1)

model.evaluate(validationImages, validationLabels, verbose=1)

# 6. Сохранение модели в новом формате Keras
model.save('mnist_model.keras')

# Функция для предсказания на новых изображениях с визуализацией
def predict_image(image_path, model):
    try:
        image = Image.open(image_path).convert('L')  # Конвертируем изображение в оттенки серого
        image = image.resize((28, 28))  # Изменяем размер изображения до 28x28
        image_array = np.array(image).reshape(1, 784).astype('float32') / 255  # Подготавливаем данные

        # Визуализация изображения
        plt.imshow(image_array.reshape(28, 28), cmap='gray')
        plt.title(f"Input Image: {os.path.basename(image_path)}")
        plt.show()

        # Предсказание
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)

        return predicted_digit
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Загрузка сохраненной модели
loaded_model = keras.models.load_model('mnist_model.keras')

# Пример использования предсказания на новом изображении
currentFolder = os.getcwd()
imagesFolder = os.path.join(currentFolder, 'images')
for eachImage in os.listdir(imagesFolder):
    imageSample = os.path.join(imagesFolder, eachImage)
    predicted_digit = predict_image(imageSample, loaded_model)
    print(f'В файле с названием {eachImage} предсказана цифра: {predicted_digit}\n')
