

import time
import numpy as np
import psutil
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import ctypes
import os
import json

# Функция для сбора данных о текущих процессах
def collect_system_data(duration=3600, interval=1):
    
    process_data = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        total_net_io = psutil.net_io_counters()
        for proc in psutil.process_iter(attrs=['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                data_point = [
                    proc.info['cpu_percent'],
                    proc.info['memory_percent'],
                    psutil.cpu_percent(interval=None),
                    psutil.virtual_memory().percent,
                    total_net_io.bytes_sent,
                    total_net_io.bytes_recv,
                    total_net_io.packets_sent,
                    total_net_io.packets_recv,
                    total_net_io.errin,
                    total_net_io.errout,
                    total_net_io.dropin,
                    total_net_io.dropout
                ]
                process_data.append(data_point)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        time.sleep(interval)
    return np.array(process_data)

# Функция для уведомления пользователя
def notify_user(message):
    
    if os.name == 'nt':  # Windows
        ctypes.windll.user32.MessageBoxW(0, message, "Уведомление об аномалиях", 1)
    else:
        os.system(f'notify-send "Уведомление" "{message}"')

# Функция для загрузки модели
def load_trained_model(model_path):
    
    model = load_model(model_path)
    return model

# Функция для обнаружения аномалий
def detect_anomalies_lstm(model, scaler, threshold, data):
    
    data = scaler.transform(data)
    timesteps = 10
    sequences = []
    
    for i in range(len(data) - timesteps):
        sequences.append(data[i:i + timesteps])
    sequences = np.array(sequences)
    reconstructions = model.predict(sequences)
    reconstruction_errors = np.mean(np.power(sequences - reconstructions, 2), axis=(1, 2))
    anomalies = np.where(reconstruction_errors > threshold)[0]
    return len(anomalies) > 0

# Загрузка обученной нейронной сети
if __name__ == "__main__":
    model_path = "LSTM_model_100k_net.keras"
    model = load_trained_model(model_path)

    print("Модель успешно загружена.")

    # Сбор данных о текущих процессах
    data = collect_system_data(duration=300, interval=1)

    # Инициализация StandardScaler для нормализации данных  
    scaler = StandardScaler()
    scaler.fit(data)

    # Установка порога отклонения
    threshold = 0.5

    if detect_anomalies_lstm(model, scaler, threshold, data):
        notify_user("Обнаружены аномалии в системных ресурсах вашего компьютера!")
    else:
        notify_user("Системные ресурсы работают в нормальном режиме.")