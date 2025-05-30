import pandas as pd

from classes.cubic_spline import CubicSpline
import matplotlib.pyplot as plt

class Custom:
    def __init__(self, ys: list[float]):
        self.ys = ys
        self.xs = [float(i) for i in range(len(ys))]

        self.spline = CubicSpline(self.xs, self.ys)

    def plot_interpolated_curve(self, title: str = "Interpolated Curve"):
        # Criar pontos intermediários para uma curva suave
        x_dense = []
        y_dense = []

        # Gerar 100 pontos entre cada par de pontos originais
        for i in range(len(self.xs) - 1):
            start = self.xs[i]
            end = self.xs[i + 1]
            segment_x = [start + (end - start) * j / 100 for j in range(100)]
            segment_y = [self.spline.interpolated_function(x) for x in segment_x]

            x_dense.extend(segment_x)
            y_dense.extend(segment_y)

        # Adicionar o último ponto
        x_dense.append(self.xs[-1])
        y_dense.append(self.ys[-1])

        # Plotar os pontos originais e a curva interpolada
        plt.figure(figsize=(10, 6))
        plt.scatter(self.xs, self.ys, color='red', s=80, label='Pontos Originais')
        plt.plot(x_dense, y_dense, '--', linewidth=2, label='Spline Cúbico')

        # Configurações do gráfico
        plt.title('Interpolação por Spline Cúbico', fontsize=14)
        plt.xlabel('x', fontsize=12)
        plt.ylabel('y', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

        # Destacar os pontos de controle
        for i, (x, y) in enumerate(zip(self.xs, self.ys)):
            plt.annotate(f'P{i}', (x, y), xytext=(5, 5),
                         textcoords='offset points', fontsize=10)

        plt.tight_layout()
        plt.show()

def main():
    df = pd.read_csv('../data/Power_Consumption_and_Generation_23_01_2025.csv')
    df.columns = df.columns.str.strip()

    ys = df['Real-time Power Usage (kW)'].to_numpy().tolist()

    custom = Custom(ys)
    custom.plot_interpolated_curve("Custom Interpolated Curve")

if __name__ == "__main__":
    main()