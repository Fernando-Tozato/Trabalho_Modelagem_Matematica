"""
1 - interpolação
2 - derivada
3 - correlação com estabilidade
4 - análise de erro
5 - análise de processamento

6 - comparação de erros e processamento

7 - comparação de dados entre métodos
"""
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline as NativeCubicSpline

from console_user_interface import ConsoleUserInterface
from cubic_spline import CustomCubicSpline


class AppController:
    cui: ConsoleUserInterface = None # Console User Interface instance

    target_column: str = None # Data to be analyzed
    period: dict[str, datetime] = None # Period for the analysis, with keys 'start' and 'end'

    xs: list[float] = None # X values for the analysis
    ys: list[float] = None # Y values for the analysis
    xs_dense: list[float] = None # Dense X values for the analysis
    inter_native: np.array = None # Interpolated Y values for the native method
    inter_custom: np.array = None # Interpolated Y values for the custom method
    deriv_native: np.array = None # Derivative values for the native method
    deriv_custom: np.array = None # Derivative values for the custom method

    benchmark_results: dict = None # Benchmark time for the native method
    error_results: dict = None # Error results for the native method

    stability = None

    def __init__(self):
        self.cui = ConsoleUserInterface()

        self.data = pd.read_csv('../data/Power_Consumption_and_Generation_Dataset.csv')
        self.data.columns = self.data.columns.str.strip()
        self.data['Timestamp'] = pd.to_datetime(self.data['Timestamp'])

    def run(self):
        self.get_data_and_analyze()

        while True:
            self.cui.clear_console()
            option = self.cui.main_menu()

            match option:
                case 1:
                    self.plot_data_analysis('nativo')
                case 2:
                    self.plot_data_analysis('customizado')
                case 3:
                    self.plot_data_analysis('comparação')
                case 4:
                    self.plot_error()
                case 5:
                    self.plot_benchmark()
                case 6:
                    self.plot_correlation_with_stability()
                case 7:
                    self.get_data_and_analyze()
                case 0:
                    self.cui.clear_console()
                    print("Saindo do programa...")
                    sys.exit()

    def get_data_and_analyze(self):
        self.target_column, self.period = self.cui.choose_column_and_period()

        self.cui.clear_console()
        print('Preparando os dados para análise...')
        self.prepare_data()

        self.cui.clear_console()
        print('Analisando os dados...')
        self.analyze_data()

    def plot_data_analysis(self, method: str):
        # Configurações de plotagem
        plt.figure(figsize=(10, 6))

        # Plotar pontos originais (sempre primeiro)
        plt.scatter(self.xs, self.ys, color='purple', s=80, zorder=3, label='Pontos Originais')

        # Casos para cada método
        if method == 'nativo':
            if self.inter_native is None or self.deriv_native is None:
                raise ValueError("Dados nativos não calculados")

            plt.plot(self.xs_dense, self.inter_native, 'b-', label='Interpolação Nativa')
            plt.plot(self.xs_dense, self.deriv_native, 'r--', label='Derivada Nativa')

        elif method == 'customizado':
            if self.inter_custom is None or self.deriv_custom is None:
                raise ValueError("Dados customizados não calculados")

            plt.plot(self.xs_dense, self.inter_custom, 'y-', label='Interpolação Custom')
            plt.plot(self.xs_dense, self.deriv_custom, 'g--', label='Derivada Custom')

        elif method == 'comparação':
            if (self.inter_native is None or self.deriv_native is None or
                    self.inter_custom is None or self.deriv_custom is None):
                raise ValueError("Dados para comparação incompletos")

            plt.plot(self.xs_dense, self.inter_native, 'b-', label='Interpolação Nativa')
            plt.plot(self.xs_dense, self.deriv_native, 'r--', label='Derivada Nativa')
            plt.plot(self.xs_dense, self.inter_custom, 'y-', label='Interpolação Custom')
            plt.plot(self.xs_dense, self.deriv_custom, 'g--', label='Derivada Custom')

        else:
            raise ValueError("Método inválido. Use 'native', 'custom' ou 'compare'")

        # Configurações do gráfico
        plt.title(f'Análise de Interpolação e Derivação - Método: {method}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./img/data_analysis_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_error(self):
        inter_errors = self.error_results['interpolation']
        deriv_errors = self.error_results['derivative']

        metrics = ['MSE', 'MAE', 'MaxError']
        inter_values = [inter_errors[metric] for metric in metrics]
        deriv_values = [deriv_errors[metric] for metric in metrics]

        # Configurações do gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = np.arange(len(metrics))

        # Plotar barras para interpolação e derivação
        bar1 = ax.bar(index, inter_values, bar_width, color='blue', label='Interpolação')
        bar2 = ax.bar(index + bar_width, deriv_values, bar_width, color='red', label='Derivação')

        # Adicionar valores nas barras
        self._add_value_labels(ax, bar1)
        self._add_value_labels(ax, bar2)

        # Configurações do eixo
        ax.set_title('Análise de Erros entre Métodos Nativo e Customizado')
        ax.set_xlabel('Métricas de Erro')
        ax.set_ylabel('Valor do Erro')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        # Ajustar escala do eixo Y se necessário
        max_val = max(max(inter_values), max(deriv_values))
        if max_val > 100 * min(val for val in inter_values + deriv_values if val > 0):
            ax.set_yscale('log')
            ax.set_ylabel('Valor do Erro (escala log)')

        plt.tight_layout()
        plt.savefig(f'./img/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def _add_value_labels(ax, bars):
        """Adiciona rótulos de valor nas barras"""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2e}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Deslocamento vertical
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)

    def plot_benchmark(self):
        build_data = self.benchmark_results['build']
        eval_data = self.benchmark_results['evaluation']
        eval_pp_data = self.benchmark_results['evaluation_per_point']

        # Configurar figura com subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Análise de Desempenho: Comparação Nativo vs Customizado', fontsize=16)

        # ==============================================
        # Primeiro gráfico: Tempos de construção e avaliação
        # ==============================================
        categories = ['Construção', 'Avaliação']
        native_means = [build_data['native_mean'], eval_data['native_mean']]
        custom_means = [build_data['custom_mean'], eval_data['custom_mean']]
        native_stds = [build_data['native_std'], eval_data['native_std']]
        custom_stds = [build_data['custom_std'], eval_data['custom_std']]
        speedups = [build_data['speedup'], eval_data['speedup']]

        # Posições das barras
        x = np.arange(len(categories))
        width = 0.35

        # Plotar barras com desvios padrão
        bar1 = ax1.bar(x - width / 2, native_means, width, yerr=native_stds,
                       capsize=5, color='blue', label='Nativo')
        bar2 = ax1.bar(x + width / 2, custom_means, width, yerr=custom_stds,
                       capsize=5, color='orange', label='Customizado')

        # Adicionar valores de speedup
        for i, speedup in enumerate(speedups):
            height = max(native_means[i] + native_stds[i], custom_means[i] + custom_stds[i])
            ax1.text(x[i], height * 1.1, f"Speedup: {speedup:.2f}x",
                     ha='center', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

        # Configurações do gráfico
        ax1.set_title('Tempos de Processamento')
        ax1.set_ylabel('Tempo (segundos)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.3)

        # ==============================================
        # Segundo gráfico: Tempo por ponto de avaliação
        # ==============================================
        categories_pp = ['Nativo', 'Customizado']
        times_pp = [eval_pp_data['native_mean'], eval_pp_data['custom_mean']]

        # Plotar barras
        bars = ax2.bar(categories_pp, times_pp, color=['blue', 'orange'])

        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height * 1.05,
                     f'{height:.2e} s/ponto',
                     ha='center', va='bottom', fontsize=10)

        # Configurações do gráfico
        ax2.set_title('Tempo Médio por Ponto (Avaliação)')
        ax2.set_ylabel('Tempo (segundos por ponto)')
        ax2.grid(True, linestyle='--', alpha=0.3)

        # Ajustar layout
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])  # Ajustar para o título superior
        plt.savefig(f'./img/benchmark_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_correlation_with_stability(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle('Correlação entre Variação e Estabilidade da Rede', fontsize=16)

        # Painel superior: Derivada e pontos de alta variação
        ax1.plot(self.xs_dense, self.deriv_native, 'r-', alpha=0.7, label='Derivada (Nativo)')
        ax1.plot(self.xs_dense, self.deriv_custom, 'g-', alpha=0.7, label='Derivada (Custom)')

        # Calcular limiar para alta variação (1.5 desvios padrão)
        threshold = 1.5 * np.std(self.deriv_native)

        # Identificar pontos de alta variação
        high_variation_mask = np.abs(self.deriv_native) > threshold
        high_variation_points = np.array(self.xs_dense)[high_variation_mask]

        # Plotar pontos de alta variação
        ax1.scatter(
            high_variation_points,
            self.deriv_native[high_variation_mask],
            color='purple', s=50, zorder=5,
            label='Alta Variação'
        )

        # Linha de limiar
        ax1.axhline(y=threshold, color='b', linestyle='--', alpha=0.5, label='Limiar Alta Variação')
        ax1.axhline(y=-threshold, color='b', linestyle='--', alpha=0.5)

        ax1.set_ylabel('Taxa de Variação')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Painel inferior: Status de estabilidade
        # Mapear índices densos para horas originais
        hour_indices = np.floor(self.xs_dense).astype(int)
        stability_dense = np.array([self.stability[min(i, len(self.stability) - 1)]
                                    for i in hour_indices])

        # Plotar status de estabilidade
        ax2.plot(self.xs_dense, stability_dense, 'k-', alpha=0.7, label='Status Estabilidade')

        # Destacar períodos instáveis
        unstable_mask = stability_dense == 1
        unstable_points = np.array(self.xs_dense)[unstable_mask]

        ax2.scatter(
            unstable_points,
            stability_dense[unstable_mask],
            color='red', s=30, zorder=5,
            label='Instável'
        )

        # Destacar alta variação que coincide com instabilidade
        coincidence_mask = high_variation_mask & unstable_mask
        coincidence_points = np.array(self.xs_dense)[coincidence_mask]

        ax2.scatter(
            coincidence_points,
            stability_dense[coincidence_mask],
            color='purple', s=100, marker='X', zorder=6,
            label='Alta Variação + Instável'
        )

        ax2.set_xlabel('Tempo (horas)')
        ax2.set_ylabel('Estabilidade\n(0 = Estável, 1 = Instável)')
        ax2.set_yticks([0, 1])
        ax2.set_ylim(-0.5, 1.5)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Calcular e mostrar taxa de coincidência
        coincidence_rate = 0.0
        if np.any(high_variation_mask):
            coincidence_rate = np.sum(coincidence_mask) / np.sum(high_variation_mask)

        ax2.text(
            0.95, 0.95,
            f'Taxa de coincidência: {coincidence_rate:.2%}',
            transform=ax2.transAxes,
            ha='right', va='top',
            bbox=dict(facecolor='white', alpha=0.8)
        )

        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        plt.savefig('./img/correlation_stability.png', dpi=300, bbox_inches='tight')
        plt.show()

    def prepare_data(self):
        mask = (self.data['Timestamp'] >= self.period['start']) & (self.data['Timestamp'] <= self.period['end'])
        period_data = self.data.loc[mask].copy().sort_values('Timestamp')


        self.ys = period_data[self.target_column].values.astype(float)
        self.stability = period_data['Power_Stability_Status'].values.astype(int)

        self.xs = list(range(len(self.ys)))

    def analyze_data(self):
        self.xs_dense = list(np.linspace(min(self.xs), max(self.xs), 500))

        # Create instances of the native and custom cubic spline classes
        native_spline = NativeCubicSpline(self.xs, self.ys)
        custom_spline = CustomCubicSpline(self.xs, self.ys)

        # Calculate interpolated curve for both native and custom spline methods
        self.inter_native = native_spline(self.xs_dense)
        self.inter_custom = np.array([custom_spline.interpolated_function(x) for x in self.xs_dense])

        # Calculate derivatives for both methods
        self.deriv_native = native_spline(self.xs_dense, 1)
        self.deriv_custom = np.array([custom_spline.derivative(x) for x in self.xs_dense])

        # Métricas de erro
        self.error_results = {
            'interpolation': {
                'MSE': np.mean((self.inter_native - self.inter_custom) ** 2),
                'MAE': np.mean(np.abs(self.inter_native - self.inter_custom)),
                'MaxError': np.max(np.abs(self.inter_native - self.inter_custom))
            },
            'derivative': {
                'MSE': np.mean((self.deriv_native - self.deriv_custom) ** 2),
                'MAE': np.mean(np.abs(self.deriv_native - self.deriv_custom)),
                'MaxError': np.max(np.abs(self.deriv_native - self.deriv_custom))
            }
        }

        # Benchmarking
        n_runs = 100
        test_points = 1000

        build_native_times = []
        build_custom_times = []

        for _ in range(n_runs):
            # Benchmark construção nativa
            start = time.perf_counter()
            NativeCubicSpline(self.xs, self.ys)
            build_native_times.append(time.perf_counter() - start)

            # Benchmark construção customizada
            start = time.perf_counter()
            CustomCubicSpline(self.xs, self.ys)
            build_custom_times.append(time.perf_counter() - start)

        # Benchmark avaliação
        eval_native_times = []
        eval_custom_times = []

        for _ in range(n_runs):
            # Benchmark avaliação nativa
            start = time.perf_counter()
            native_spline(self.xs_dense)
            eval_native_times.append(time.perf_counter() - start)

            # Benchmark avaliação customizada
            start = time.perf_counter()
            for x in self.xs_dense:
                custom_spline.interpolated_function(x)
            eval_custom_times.append(time.perf_counter() - start)

        # Processar resultados de benchmark
        self.benchmark_results = {
            'build': {
                'native_mean': np.mean(build_native_times),
                'native_std': np.std(build_native_times),
                'custom_mean': np.mean(build_custom_times),
                'custom_std': np.std(build_custom_times),
                'speedup': np.mean(build_custom_times) / np.mean(build_native_times)
            },
            'evaluation': {
                'native_mean': np.mean(eval_native_times),
                'native_std': np.std(eval_native_times),
                'custom_mean': np.mean(eval_custom_times),
                'custom_std': np.std(eval_custom_times),
                'speedup': np.mean(eval_custom_times) / np.mean(eval_native_times)
            },
            'evaluation_per_point': {
                'native_mean': np.mean(eval_native_times) / test_points,
                'custom_mean': np.mean(eval_custom_times) / test_points
            }
        }