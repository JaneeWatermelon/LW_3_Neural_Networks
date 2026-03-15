import os
import random
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def plot_training_history(history: Dict[str, list[float]]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["loss"], label="Тренировочные потери")
    ax1.plot(history["val_loss"], label="Валидационные потери")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Потери")
    ax1.set_title("Функция потерь во время обучения")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["mae"], label="Тренировочная MAE")
    ax2.plot(history["val_mae"], label="Валидационная MAE")
    ax2.set_xlabel("Эпоха")
    ax2.set_ylabel("MAE")
    ax2.set_title("График MAE")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    print("График обучения сохранен в 'training_history.png'")
    plt.show()

if __name__ == "__main__":
    os.environ["TCL_LIBRARY"] = r"C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
    os.environ["TK_LIBRARY"] = r"C:\Users\Warer\AppData\Local\Programs\Python\Python313\tcl\tk8.6"

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    print("Загрузка данных...")

    # 1. Загрузка данных
    df = pd.read_csv("StudentsPerformance.csv")

    # количество примеров
    num_samples = len(df)
    print("Количество примеров в датасете:", num_samples)

    # 2. Создание целевой переменной
    print("Создаём целевую переменную — средний балл по экзаменам")

    df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

    # 3. Признаки
    X = df[["math score", "reading score", "writing score"]].values
    y = df["average_score"].values

    num_features = X.shape[1]
    print("Количество признаков:", num_features)

    # 4. Нормализация
    print("Нормализация данных")

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X - X_mean) / X_std

    # 5. Разделение данных
    print("Разделение данных на обучающую и тестовую выборки")

    indices = np.random.permutation(len(X))

    split = int(len(X) * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Размер обучающей выборки:", len(X_train))
    print("Размер тестовой выборки:", len(X_test))

    # 6. Создание модели
    print("Создаём нейронную сеть")

    model = Sequential([
        Dense(16, activation="relu", input_shape=(3,)),
        Dense(8, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    print("Модель скомпилирована")

    # 7. Обучение
    print("Начинаем обучение модели")

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
    )

    # 8. Оценка
    print("\nОценка модели на тестовых данных")

    loss, mae = model.evaluate(X_test, y_test)

    print("loss:", loss)
    print("MAE:", mae)

    # 9. Предсказания
    print("\nПримеры предсказаний:")

    sample_X = X_test[:5]
    sample_y = y_test[:5]

    predictions = model.predict(sample_X)

    for i in range(len(sample_X)):
        real_y = np.round(sample_y[i], 2)
        pred_y = np.round(predictions[i][0], 2)

        print(f"№ {i+1}")
        print("Предсказанный средний балл:", pred_y)
        print("Реальный средний балл:", real_y)
        print("Разница:", np.round(abs(real_y - pred_y), 2))
        print()

    # 10. Графики обучения
    print("Строим графики обучения")

    plot_training_history(history.history)