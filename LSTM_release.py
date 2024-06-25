# Загружаем все необходимые библиотеки
import time
import numpy as np
import psutil
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import ctypes
import os
import csv
import json
import matplotlib.pyplot as plt

# Функция сбора данных о процессах и системных ресурсах, сбор данных производится в течении заданного времени с временными интервалами
def collect_system_data(duration=3600, interval=1): # Время сбора данных в секундах с интервалом
   
    process_data = []
    start_time = time.time()
    
    while len(process_data) < 1000: # Ограничение сбора данных, ограничивает количество строк датасета
        total_net_io = psutil.net_io_counters()
        
        for proc in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
           
            try:
                data_point = [
                    proc.info['cpu_percent'],
                    proc.info['memory_percent'],
                    psutil.cpu_percent(interval=None), # Загруженность процессора
                    psutil.virtual_memory().percent,  # Загруженность оперативной памяти
                    total_net_io.bytes_sent,# Количество байтов, отправленных процессом через сетевые интерфейсы с момента его запуска
                    total_net_io.bytes_recv,      # Количество байтов, полученных процессом через сетевые интерфейсы с момента его запуска
                    total_net_io.packets_sent,# Количество пакетов, отправленных процессом через сетевые интерфейсы
                    total_net_io.packets_recv,# Количество пакетов, полученных процессом через сетевые интерфейсы
                    total_net_io.errin,  # Количество пакетов с ошибками, полученных процессом
                    total_net_io.errout,  # Количество пакетов с ошибками, отправленных процессом
                    total_net_io.dropin,  # Количество входящих пакетов, отброшенных процессом
                    total_net_io.dropout  # Количество исходящих пакетов, отброшенных процессом
                ]
                process_data.append(data_point)
                
                if len(process_data) >= 1000: # Ограничение сбора данных, ограничивает количество строк датасета
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        time.sleep(interval)
    return np.array(process_data)

# Функция загрузки датасета из csv файла
def load_data_from_csv(filename):
    
    try:
        data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        return data
    except Exception as e:
        print(f"Ошибка при загрузке данных из файла {filename}: {e}")
        return None

# Функция с различными вариантами работы программного средства
def prompt_user():
    
    print("Выберите, каким образом начать обучение нейронной сети:")
    print("1: Собрать данные о процессах системы")
    print("2: Загрузить данные из файла")
    print("3: Проверить точность модели")
    choice = input("Введите 1, 2 или 3: ").strip()
    return choice

# Функция уведомления пользователя об аномалиях
def notify_user(message):
    
    if os.name == 'nt':  # Windows
        ctypes.windll.user32.MessageBoxW(
            0, message, "Уведомление об аномалиях", 1)
    else:
        os.system(f'notify-send "Уведомление" "{message}"')

# Функция загрузки перечня рекомендательных действий из txt файла
def read_recommendations(filename='rec.txt'):
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return "Не удалось загрузить рекомендации."

# Функция обучения нейронно сети
def train_lstm_autoencoder(data):
    
    # Инициализация StandardScaler для нормализации данных
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    timesteps = 10
    input_dim = data.shape[1]
    sequences = []
    
    # Создание последовательностей данных
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    sequences = np.array(sequences)
    
    # Обучение нейронной сети, добавление слоёв LSTM, Dropout, функций активаций и выходного слоя
    model = Sequential()
    model.add(LSTM(512, activation='relu', input_shape=(timesteps, input_dim), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256, activation='relu', return_sequences=False))
    model.add(Dropout(0.3))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(256, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(input_dim)))
    
    # Инициализация оптимизатора обучений нейронной сети
    optimizer = Adam(learning_rate=0.0005)
    # Компиляция модели с функцией потерь среднеквадратичной ошибки
    model.compile(optimizer=optimizer, loss='mse')
    
    # Инициализация EarlyStopping для ранней остановки обучения нейронной сети
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Загрузка параметров для обучения нейронной сети
    history = model.fit(sequences,
                        sequences, 
                        epochs=100,
                        batch_size=64,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[early_stopping])
    
    # Сохранение обученной нейронной сети в 2х расширениях
    model.save("LSTM_model_1k_net.h5")
    model.save("LSTM_model_1k_net.keras")
    
    # Предсказание обученной нейронной сети на обучающем датасете
    reconstructions = model.predict(sequences)
    reconstruction_errors = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    
    # Определение порогового значения для аномалий, равное mse + 2 отклонения
    threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
    
    return model, scaler, threshold

# Функция обнаружения аномалий с использованием обученной нейронной сети
def detect_anomalies_lstm(model, scaler, threshold, data):
    
    data = scaler.transform(data)
    timesteps = 10
    sequences = []
    
    # Создание последовательностей данных
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    sequences = np.array(sequences)
    
    # Предсказание модели на новых данных
    reconstructions = model.predict(sequences)
    reconstruction_errors = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    anomalies = np.where(reconstruction_errors > threshold)[0]
    return len(anomalies) > 0

# Функция сохранения датасета в csv файл
def save_data_to_csv(data, filename='data_1k_net.csv'):
    
    header = ['CPU Percent', 'Memory Percent', 'Total CPU Percent', 'Total Memory Percent',
              'Bytes Sent', 'Bytes Recv', 'Packets Sent', 'Packets Recv', 'Errin', 'Errout', 'Dropin', 'Dropout']
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

# Функция сохранения датасета в json файл
def save_data_to_json(data, filename='data_1k_net.json'):
    
    header = ['CPU Percent', 'Memory Percent', 'Total CPU Percent', 'Total Memory Percent',
              'Bytes Sent', 'Bytes Recv', 'Packets Sent', 'Packets Recv', 'Errin', 'Errout', 'Dropin', 'Dropout']
    json_data = []
    for row in data:
        json_data.append(dict(zip(header, row.tolist())))

    with open(filename, 'w') as file:
        json.dump(json_data, file, indent=4)

# Функция загрузки обученной модели
def load_trained_model(model_path):
    
    model = load_model(model_path)
    return model

# Функция оценки точности обученной нейронной сети
def evaluate_model_accuracy(model, scaler, data):
    
    data = scaler.transform(data)
    timesteps = 10
    sequences = []
    
    # Создание последовательностей данных
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    sequences = np.array(sequences)

    # Предсказание модели на новых данных
    reconstructions = model.predict(sequences)
    reconstruction_errors = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    mse = np.mean(reconstruction_errors)

    # Прогнозируемые значения
    predicted = model.predict(sequences)
    true_values = sequences

    # Построение графиков
    plt.figure(figsize=(12, 8))

    # График MSE
    plt.subplot(2, 1, 1)  
    plt.plot(reconstruction_errors, label='Reconstruction Error')
    plt.axhline(y=mse, color='r', linestyle='--', label='Mean MSE')
    plt.xlabel('Samples')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Error Over Time')
    plt.legend()

    # График точности прогнозирования
    plt.subplot(2, 1, 2)  
    plt.plot(true_values[:, -1, 0], label='True Values')
    plt.plot(predicted[:, -1, 0], label='Predicted Values')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.title('True vs Predicted Values Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return mse


if __name__ == "__main__":
    
    user_choice = prompt_user()

    # Сбор данных для датасета и обучение нейронной сети с "нуля"
    if user_choice == '1':
        
        data = collect_system_data(duration=3600, interval=1)
        save_data_to_csv(data)
        save_data_to_json(data)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        model, scaler, threshold = train_lstm_autoencoder(train_data)

    # Загрузка готового датасета
    elif user_choice == '2':
        
        # Путь к датасету
        data = load_data_from_csv('data_1k_net.csv')
        if data is not None:
            print("Данные успешно загружены из файла.")
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            model, scaler, threshold = train_lstm_autoencoder(train_data)
        else:
            print("Не удалось загрузить данные. Программа завершена.")
            exit()

    # Загрузка обученной нейронной сети
    elif user_choice == '3':
        
        # Путь к обученной нейронной стеи
        model_path = "LSTM_model_1k_net.keras"  
        model = load_trained_model(model_path)
        
        # Тестовый датасет
        data = load_data_from_csv('data_1k_net.csv')

        # Проверка корректности загрузки датасета
        if data is not None:
            
            print("Тестовые данные успешно загружены из файла.")
            scaler = StandardScaler()
            scaler.fit(data)
            evaluate_model_accuracy(model, scaler, data)
        else:
            print("Не удалось загрузить тестовые данные. Программа завершена.")
            exit()

    else:
        print("Неверный выбор. Программа завершена.")
        exit()

    if user_choice in ['1', '2'] and detect_anomalies_lstm(model, scaler, threshold, test_data):
        
        # Уведомление пользователя об аномалии
        notify_user("Обнаружены аномалии в системных ресурсах вашего компьютера!")
        
        # Уведомление с рекомендациями
        recommendations = read_recommendations()
        notify_user(recommendations)
    else:
        notify_user("Аномалий не обнаружено.")
