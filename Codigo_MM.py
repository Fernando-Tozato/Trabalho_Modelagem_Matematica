# 1️⃣ Importar bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 2️⃣ Fazer upload do arquivo CSV
df = pd.read_csv('data/Power_Consumption_and_Generation_Dataset.csv')
df.columns = df.columns.str.strip()  # remove espaços nas colunas

# 3️⃣ Filtrar apenas os dados de 23/01/2025
df['Timestamp'] = pd.to_datetime(df['Timestamp'])  # converte datas
df = df[df['Timestamp'].dt.date == pd.to_datetime('2025-01-23').date()]  # filtra dia
df = df.sort_values('Timestamp')  # ordena por tempo
df = df.reset_index(drop=True)

# 4️⃣ Selecionar a coluna de consumo como Y e o índice como X
x = np.arange(len(df))  # X = 0, 1, 2, ..., N-1 (assume dados por hora)
y = df['Real-time Power Usage (kW)'].to_numpy()  # Y = consumo

# 5️⃣ Interpolação com CubicSpline
spline = CubicSpline(x, y)
x_interp = np.linspace(x.min(), x.max(), 240)  # pontos intermediários
y_interp = spline(x_interp)  # valores interpolados

# 6️⃣ Derivada com np.gradient
derivada = np.gradient(y, x)

# 7️⃣ Detectar pontos críticos onde variação é muito alta
limiar = 1.5 * np.std(derivada)  # 2 vezes o desvio padrão
picos = np.where(np.abs(derivada) > limiar)[0]  # índices com variação crítica

# 8️⃣ Plotar gráficos
plt.figure(figsize=(14, 5))

# Gráfico 1: curva original + interpolada
plt.subplot(1, 2, 1)
plt.plot(x, y, 'o', label='Original')
plt.plot(x_interp, y_interp, '--', label='Spline Cúbico')
plt.title("Interpolação de Consumo (23/01/2025)")
plt.xlabel("Hora"); plt.ylabel("Consumo (kW)")
plt.legend()

# Gráfico 2: derivada com pontos críticos
plt.subplot(1, 2, 2)
plt.plot(x, derivada, label='Derivada', color='red')
plt.axhline(limiar, color='gray', linestyle='--', label='±2σ')
plt.axhline(-limiar, color='gray', linestyle='--')
plt.scatter(picos, derivada[picos], color='black', label='Pontos Críticos')
plt.title("Derivada do Consumo")
plt.xlabel("Hora"); plt.ylabel("ΔConsumo/Δt")
plt.legend()

plt.tight_layout()
plt.show()

# 9️⃣ Mostrar os pontos críticos encontrados
print(f"Pontos com variação crítica (> 2σ): {picos.tolist()}")