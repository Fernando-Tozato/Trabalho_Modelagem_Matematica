import os

from datetime_selector import DateTimeSelector


class ConsoleUserInterface:
    data_dict = {
        1: "Real-time Power Usage (kW)",
        2: "Peak Demand (kW)",
        3: "Load Factor (%)",
        4: "Voltage Fluctuations (V)",
        5: "Current Load (A)",
        6: "Power Factor",
        7: "Reactive Power (kVAR)",
        8: "Energy Consumption per Hour (kWh)",
        9: "Real-time Power Generation (kW)",
        10: "Renewable Energy Contribution (%)",
        11: "Battery Storage Level (%)",
        12: "Grid Frequency (Hz)",
        13: "Solar Radiation (W/m²)",
        14: "Wind Speed (m/s)",
        15: "Fuel Consumption Rate (L/h or kg/h)",
        16: "Carbon Emissions (gCO₂/kWh)",
        17: "Generation Forecast Accuracy (%)"
    }
    dt_selector: DateTimeSelector = None

    def __init__(self):
        self.dt_selector = DateTimeSelector(self.clear_console)

    @staticmethod
    def clear_console():
        os.system('cls' if os.name == 'nt' else 'clear')

    def loop(self, text: str, possible_responses: list[int]) -> int:
        while True:
            print(text)
            response = int(input("Digite sua resposta: "))

            if response in possible_responses:
                return response
            else:
                self.clear_console()
                print("Resposta inválida. Tente novamente.\n\n")

    def choose_column_and_period(self):
        self.clear_console()
        text = (
            "Escolha os dados que deseja analisar:\n"
            "1 - Uso de Energia em Tempo Real (kW)\n"
            "2 - Demanda de Pico (kW)\n"
            "3 - Fator de Carga (%)\n"
            "4 - Flutuações de Voltagem (V)\n"
            "5 - Carga Atual (A)\n"
            "6 - Fator de Potência\n"
            "7 - Potência Reativa (kVAR)\n"
            "8 - Consumo de Energia por Hora (kWh)\n"
            "9 - Geração de Energia em Tempo Real (kW)\n"
            "10 - Contribuição de Energia Renovável (%)\n"
            "11 - Nível de Armazenamento da Bateria (%)\n"
            "12 - Frequência da Rede (Hz)\n"
            "13 - Radiação Solar (W/m²)\n"
            "14 - Velocidade do Vento (m/s)\n"
            "15 - Taxa de Consumo de Combustível (L/h ou kg/h)\n"
            "16 - Emissões de Carbono (gCO₂/kWh)\n"
            "17 - Precisão da Previsão de Geração (%)\n"
        )
        possible_responses = list(range(1, 18))

        response = self.loop(text, possible_responses)

        self.clear_console()

        data = self.data_dict.get(response, "Dados não encontrados")
        period = self.dt_selector.selecionar_periodo()

        return data, period

    def main_menu(self):
        self.clear_console()
        text = (
            "Bem-vindo ao Sistema de Análise da Rede Elétrica!\n"
            "Escolha uma opção:\n"
            "1 - Analisar dados a partir de método nativo\n"
            "2 - Analisar dados a partir de método customizado\n"
            "3 - Comparar dados nos dois métodos\n"
            "-------------------------------------------------------------------\n"
            "4 - Comparar erro dos dois métodos\n"
            "5 - Comparar desempenho dos dois métodos\n"
            '6 - Comparar correlação com estabilidade dos dois métodos\n'
            "-------------------------------------------------------------------\n"
            '7 - Trocar dados e período da análise\n'
            "0 - Sair\n"
        )
        possible_responses = list(range(8))

        response = self.loop(text, possible_responses)

        return response

