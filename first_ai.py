import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)
print(x_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

print(x_train.shape)

model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adagrad',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=5,
                    validation_data=(x_test, y_test))
model.save('mnist_model.h5')

# Визуализация точности на графике
plt.figure(figsize=(12, 4))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучении', marker='o')
plt.plot(history.history['val_accuracy'], label='Точность на проверке', marker='o')
plt.title('Точность модели по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на обучении', marker='o')
plt.plot(history.history['val_loss'], label='Потери на проверке', marker='o')
plt.title('Потери модели по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Получаем предсказания для всего тестового набора
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy (точность)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy (общая точность): {accuracy:.4f} ({accuracy*100:.2f}%)")

# F1-score (средний)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
print(f"F1-score (macro average): {f1_macro:.4f}")
print(f"F1-score (weighted average): {f1_weighted:.4f}")

# Детальный отчет по каждой цифре
print(classification_report(y_test, y_pred,
                          target_names=[f'Цифра {i}' for i in range(10)],
                          digits=4))

# 10. Матрица ошибок (confusion matrix)
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('Матрица ошибок')
plt.xlabel('Предсказанная цифра')
plt.ylabel('Истинная цифра')
plt.show()


