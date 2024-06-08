
import os
import importlib
import sys
import subprocess
requiredLibraries = ['tensorflow', 'PIL']
for lib in requiredLibraries:
    try:
        importlib.import_module(lib)
    except ImportError:
        print(f"{lib} не установлена. Устанавливаем...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib])
            print(f"{lib} успешно установлена.")
        except Exception as e:
            print(f"Ошибка при установке {lib}: {e}\n Скрипт будет остановлен. Попробуйте установить {lib} вручную")
            sys.exit()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# 1. Загрузка данных MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. Предварительная обработка данных
train_images = train_images.reshape((60000, 784)).astype('float32') / 255
test_images = test_images.reshape((10000, 784)).astype('float32') / 255

# 3. Создание модели Sequential
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 4. Компиляция модели
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Обучение модели
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))

# 6. Сохранение модели
model.save('mnist_model.h5')

# Функция для предсказания на новых изображениях
def predict_image(image_path, model):
    from PIL import Image
    image = Image.open(image_path).convert('L')  # Конвертируем изображение в оттенки серого
    image = image.resize((28, 28))  # Изменяем размер изображения до 28x28
    image_array = np.array(image).reshape(1, 784).astype('float32') / 255  # Подготавливаем данные
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    return predicted_digit

# Пример использования предсказания на новом изображении
image_path = 'path_to_your_image.png'  # Укажите путь к изображению
predicted_digit = predict_image(image_path, model)
print(f'Predicted digit: {predicted_digit}')