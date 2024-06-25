import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Загрузка датасета из CSV-файла
csv_file_path = 'D:\\Code\\datasets_net\\data_250k_net.csv'

# Чтение датасета 
df = pd.read_csv(csv_file_path)
print(df.head())

# Построение графиков

# 1. Линейный график (Line plot)
plt.figure(figsize=(10, 5))
plt.plot(df['Total CPU Percent'], df['CPU Percent'], label='Value 1')
plt.plot(df['Total CPU Percent'], df['CPU Percent'], label='Value 2')
plt.xlabel('Total CPU Percent')
plt.ylabel('Values')
plt.title('Line plot of Values over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Гистограмма (Histogram)
plt.figure(figsize=(10, 5))
plt.hist(df['Total CPU Percent'], bins=20, alpha=0.5, label='Value 1')
plt.hist(df['CPU Percent'], bins=20, alpha=0.5, label='Value 2')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Values')
plt.legend()
plt.tight_layout()
plt.show()

# 3. Точечный график (Scatter plot)
plt.figure(figsize=(10, 5))
plt.scatter(df['Total CPU Percent'], df['CPU Percent'], alpha=0.5)
plt.xlabel('Value 1')
plt.ylabel('Загруженность процессора')
plt.title('Scatter plot of Value1 vs Value2')
plt.tight_layout()
plt.show()

# 4. Диаграмма размаха (Box plot)
plt.figure(figsize=(10, 5))
plt.boxplot([df['Total CPU Percent'], df['CPU Percent']], labels=['Value 1', 'Value 2'])
plt.ylabel('Values')
plt.title('Box plot of Values')
plt.tight_layout()
plt.show()

#Вывод данных о слоях нейронной сети
new_model = tf.keras.models.load_model('LSTM_model_100k_net.keras')
new_model.summary()