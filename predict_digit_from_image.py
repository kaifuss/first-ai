# predict_digit_from_image.py

import os
import win32gui
import tkinter as tk
import numpy as np
from tkinter import *
from keras import models
from PIL import Image, ImageGrab
from matplotlib import pyplot as plt
from create_new_model import createModel

# проверка существования модели в директории
if not os.path.exists('model_for_predict.keras'):
    print('Не найдена модель для предсказания. Будет создана новая модель.\n')
    currentModel = createModel()
    currentModel.save('model_for_predict.keras')
    print('\nМодель создана и сохранена в файл model_for_predict.keras.\n')
else:
    currentModel = models.load_model('model_for_predict.keras')

def predictImage(image):
    image = image.resize((28, 28))  # Изменяем размер изображения до 28x28 под данные модели
    image = image.convert('L')      # Конвертируем изображение в оттенки серого
    image = np.array(image)         # Преобрабатываем изображение в массив пикселей
    image = image.reshape(1, 784).astype('float32') / 255.0   # Подготовка и норамализация данных
    image = 1 - image               # Перевод из БЧ в ЧБ
    prediction = currentModel.predict([image])[0] 
    return np.argmax(prediction), max(prediction)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Создание элементов
        self.canvas = tk.Canvas(self, width=300, height=300, bg='white', cursor='cross')
        self.label = tk.Label(self, text = 'Происходит размышление...', font=('Helvetica', 40))
        self.classify_btn = tk.Button(self, text = 'Распознать', command =         self.classify_handwriting)
        self.button_clear = tk.Button(self, text = 'Очистить', command=self.clear_all)

        # Сетка окна
        self.canvas.grid(row=0, column=0, pady=2, sticky=W,)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2, padx=2)

        # Обработка событий (рисование)
        self.canvas.bind("<B1-Motion>", self.draw_lines)


    def clear_all(self):
        self.canvas.delete('all')

    def classify_handwriting(self):
        HWD = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(HWD) 
        img = ImageGrab.grab(rect)

        digit, acc = predictImage(img)
        self.label.configure(text= str(digit) + ', ' + str(int(acc*100)) + '%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')


app = App()
app.mainloop()