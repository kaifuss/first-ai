# create_new_model.py

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

def createModel():
    # 1. Загрузка данных MNIST
    trainingNumbers = 60000
    validatingNumbers = 10000
    height, width = 28, 28
    amountOfClasses = 10
    (trainImages, trainLabels), (validationImages, validationLabels) = mnist.load_data()

    # 2. Предварительная обработка данных
    trainImages = trainImages.reshape((trainingNumbers, height * width)).astype('float32') / 255
    validationImages = validationImages.reshape((validatingNumbers, height * width)).astype('float32') / 255
    trainLabels = to_categorical(trainLabels, amountOfClasses)
    validationLabels = to_categorical(validationLabels, amountOfClasses)

    # 3. Создание модели Sequential
    batchSize = 64  # 1000 семплов в каждой эпохе
    numEpochs = 5  # 5 эпох
    hiddenSize = 784  # 784 нейронов в слое
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(hiddenSize, activation='relu'),
        layers.Dense(196, activation='relu'),
        layers.Dense(49, activation='relu'),
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
    score = model.evaluate(validationImages, validationLabels, verbose=1)
    print('Потери на тестовом наборе: ', score[0])
    print('Точность на тестовом наборе: ', score[1])

    return model