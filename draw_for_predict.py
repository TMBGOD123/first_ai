import tkinter as tk
from tkinter import Canvas
import numpy as np
from PIL import Image, ImageDraw, ImageGrab
import tensorflow as tf
import threading

# Загружаем нашу сохраненную модель
model = tf.keras.models.load_model('mnist_model.h5')


class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Рисовалка цифр для нейросети")
        master.geometry("400x500")
        master.resizable(False, False)

        # Создаем холст для рисования размером 28x28 (10x увеличение от 28x28)
        self.canvas = Canvas(master, width=280, height=280, bg='white', cursor='cross')
        self.canvas.pack(pady=20)

        # Привязываем события мыши для рисования
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # Переменные для рисования
        self.last_x = None
        self.last_y = None
        self.drawing = False

        # Кнопки управления
        button_frame = tk.Frame(master)
        button_frame.pack()

        self.predict_button = tk.Button(button_frame, text="Распознать цифру", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(button_frame, text="Очистить", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        # Метка для вывода результата
        self.result_label = tk.Label(master, text="Нарисуйте цифру и нажмите 'Распознать'", font=('Arial', 14))
        self.result_label.pack(pady=10)

        # Метка для вероятностей
        self.prob_label = tk.Label(master, text="", font=('Arial', 10))
        self.prob_label.pack()

    def paint(self, event):
        # Рисуем жирные линии (ширина 15 пикселей), как настоящий маркер
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                    width=15, fill='black', capstyle=tk.ROUND, smooth=True)
        self.last_x = x
        self.last_y = y

    def reset(self, event):
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="Нарисуйте цифру и нажмите 'Распознать'")
        self.prob_label.config(text="")

    def preprocess_image(self):
        # Получаем координаты холста
        x = self.master.winfo_rootx() + self.canvas.winfo_x()
        y = self.master.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        # Захватываем изображение с холста
        img = ImageGrab.grab(bbox=(x, y, x1, y1))

        # Конвертируем в градации серого
        img = img.convert('L')

        # Инвертируем цвета (у нас черное на белом, а сеть училась на белом на черном)
        img = Image.eval(img, lambda x: 255 - x)

        # Уменьшаем до 28x28 пикселей (как в MNIST)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Превращаем в numpy массив и нормализуем
        img_array = np.array(img) / 255.0

        return img_array

    def predict_digit(self):
        # Запускаем в отдельном потоке, чтобы интерфейс не зависал
        threading.Thread(target=self._predict, daemon=True).start()

    def _predict(self):
        """Внутренняя функция для предсказания"""
        try:
            # Подготавливаем изображение
            img_array = self.preprocess_image()

            # Вытягиваем в одномерный массив (как при обучении)
            img_flat = img_array.reshape(1, 784)

            # Получаем предсказание
            predictions = model.predict(img_flat, verbose=0)[0]
            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit] * 100

            # Формируем строку с вероятностями для всех цифр
            probs_text = "Вероятности: "
            for i, prob in enumerate(predictions):
                probs_text += f"{i}: {prob * 100:.1f}% "

            # Обновляем интерфейс (в главном потоке)
            self.master.after(0, self.update_result, predicted_digit, confidence, probs_text)

        except Exception as e:
            self.master.after(0, self.show_error, str(e))

    def update_result(self, digit, confidence, probs_text):
        """Обновляем результат в интерфейсе"""
        self.result_label.config(
            text=f"Это цифра: {digit} (уверенность: {confidence:.1f}%)",
            fg='green' if confidence > 80 else 'orange'
        )
        self.prob_label.config(text=probs_text)

    def show_error(self, error):
        """Показываем ошибку"""
        self.result_label.config(text=f"Ошибка: {error}", fg='red')


# Запускаем приложение
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()