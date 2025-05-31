import pandas as pd

from classes.cubic_spline import CustomCubicSpline
import matplotlib.pyplot as plt

class Custom:
    def __init__(self, ys: list[float]):
        self.ys = ys
        self.xs = [float(i) for i in range(len(ys))]

        self.spline = CustomCubicSpline(self.xs, self.ys)
        self.critical_points = self.spline.find_critical_points()

    def plot_interpolated_curve(self):
        print("\nPontos Críticos Identificados:")
        print("-----------------------------")
        for i, (x, y, ptype) in enumerate(self.critical_points):
            print(f"Ponto {i + 1} ({'Máximo' if ptype == 'max' else 'Mínimo'}):")
            print(f"  x = {x:.6f}")
            print(f"  y = {y:.6f}")
            print(f"  Tipo = {ptype}")
            print()

        # Preparar dados para plotagem
        x_min, x_max = min(self.xs), max(self.xs)
        num_points = 500

        # Implementação manual do linspace
        step = (x_max - x_min) / (num_points - 1)
        x_dense = [x_min + i * step for i in range(num_points)]

        # Calcular curva e derivada
        curve = [self.spline.interpolated_function(x) for x in x_dense]
        derivative = [self.spline.derivative(x) for x in x_dense]

        # Criar figura com eixos duplos
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Plotar curva interpolada (eixo esquerdo)
        color_curve = 'tab:blue'
        ax1.plot(x_dense, curve, color=color_curve, linewidth=3, label='Curva Interpolada')
        ax1.scatter(self.xs, self.ys, color='red', s=100, zorder=5,
                    edgecolor='black', label='Pontos Originais')

        # Plotar pontos críticos
        for x, y, point_type in self.critical_points:
            if point_type == 'max':
                ax1.scatter([x], [y], color='lime', s=200, marker='^',
                            edgecolor='black', zorder=6, label='Máximo Local')
            else:
                ax1.scatter([x], [y], color='purple', s=200, marker='v',
                            edgecolor='black', zorder=6, label='Mínimo Local')

        # Configurar eixo da curva
        ax1.set_xlabel('x', fontsize=14)
        ax1.set_ylabel('Valor da Função', color=color_curve, fontsize=14)
        ax1.tick_params(axis='y', labelcolor=color_curve)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Criar segundo eixo para a derivada
        ax2 = ax1.twinx()
        color_deriv = 'tab:green'
        ax2.plot(x_dense, derivative, color=color_deriv, linewidth=2,
                 linestyle='--', label='Derivada')

        # Configurar eixo da derivada
        ax2.set_ylabel('Taxa de Variação (Derivada)', color=color_deriv, fontsize=14)
        ax2.tick_params(axis='y', labelcolor=color_deriv)

        # Adicionar linha de referência em y=0 para derivada
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

        # Destacar pontos onde derivada = 0 (em AMARELO)
        for x, y, _ in self.critical_points:
            ax2.scatter([x], [0], color='yellow', s=80, marker='o',
                        edgecolor='black', zorder=5, label='Derivada Zero')

        # Configurar título e layout
        plt.title('Análise Completa: Curva Interpolada com Derivada e Pontos Críticos',
                  fontsize=16, pad=20)

        # Combinar legendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines = lines1 + lines2
        all_labels = labels1 + labels2

        # Remover duplicatas usando dicionário
        unique_labels = []
        unique_lines = []
        for label, line in zip(all_labels, all_lines):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_lines.append(line)

        # Posicionar legenda na parte inferior central
        plt.legend(unique_lines, unique_labels, loc='upper center',
                   bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)

        # Ajustar layout
        fig.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        plt.savefig('analise_completa.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    df = pd.read_csv('../data/Power_Consumption_and_Generation_Dataset.csv')
    df.columns = df.columns.str.strip()

    ys = df['Real-time Power Usage (kW)'].to_numpy().tolist()

    custom = Custom(ys)
    custom.plot_interpolated_curve()

if __name__ == "__main__":
    main()