from datetime import datetime, timedelta

class DateTimeSelector:
    """
    Classe para selecionar um período inicial e final via console.
    O período permitido é entre 01/01/2025 00:00 e 11/02/2025 15:00.
    O intervalo mínimo entre início e fim é de 3 horas.
    """

    DATE_FORMAT = "%d/%m/%Y %H:%M:%S"

    MIN_START_DATETIME = datetime(2025, 1, 1, 0, 0, 0)  # Mínimo para início
    MAX_START_DATETIME = datetime(2025, 2, 11, 12, 0, 0)  # Máximo para início

    MIN_END_DATETIME = datetime(2025, 1, 1, 3, 0, 0)  # Mínimo para fim
    MAX_END_DATETIME = datetime(2025, 2, 11, 15, 0, 0)  # Máximo para fim

    MIN_INTERVAL = timedelta(hours=3)

    def __init__(self, clear_console: callable):
        self.start_datetime = None
        self.end_datetime = None
        self.clear_console = clear_console

    def _informar_periodo_permitido(self):
        self.clear_console()

        print("Período permitido:")
        print(f"  Início mínimo: {self.MIN_START_DATETIME.strftime(self.DATE_FORMAT)}")
        print(f"  Fim máximo:    {self.MAX_END_DATETIME.strftime(self.DATE_FORMAT)}")
        print(f"Intervalo mínimo entre início e fim: {self.MIN_INTERVAL} (3 horas)")
        print()

    def _ler_datetime(self, prompt: str, is_start: bool = False) -> datetime | None:
        """
        Lê uma string do usuário e tenta converter em datetime.
        Repete até o usuário informar um valor válido no formato correto
        e dentro do período permitido.
        """
        while True:
            self._informar_periodo_permitido()
            texto = input(prompt).strip()
            try:
                dt = datetime.strptime(texto, self.DATE_FORMAT)
            except ValueError:
                print(f"Formato inválido. Use '{self.DATE_FORMAT}'. Tente novamente.")
                continue

            if is_start:
                if dt < self.MIN_START_DATETIME or dt > self.MAX_START_DATETIME:
                    print(
                        "Data/hora fora do período permitido.\n"
                        f"Informe entre {self.MIN_START_DATETIME.strftime(self.DATE_FORMAT)} "
                        f"e {self.MAX_START_DATETIME.strftime(self.DATE_FORMAT)}."
                    )
                    continue
            else:
                if dt < self.MIN_END_DATETIME or dt > self.MAX_END_DATETIME:
                    print(
                        "Data/hora fora do período permitido.\n"
                        f"Informe entre {self.MIN_END_DATETIME.strftime(self.DATE_FORMAT)} "
                        f"e {self.MAX_END_DATETIME.strftime(self.DATE_FORMAT)}."
                    )
                    continue

            return dt

    def selecionar_periodo(self) -> dict[str, datetime]:
        """
        Solicita ao usuário a data/hora inicial e final, garantindo que:
         - Ambas estejam no intervalo permitido.
         - O fim seja posterior ao início.
         - O intervalo seja de pelo menos 3 horas.
        """

        print("Informe a data/hora de início.")
        self.start_datetime = self._ler_datetime("  Início (DD/MM/YYYY HH:MM:SS): ", is_start=True)

        while True:
            print("Informe a data/hora de término.")
            end_dt = self._ler_datetime("  Término (DD/MM/YYYY HH:MM:SS): ")

            if end_dt <= self.start_datetime:
                print("A data/hora de término deve ser posterior à de início. Tente novamente.")
                continue

            if (end_dt - self.start_datetime) < self.MIN_INTERVAL:
                print(
                    f"O intervalo entre início e término deve ser de pelo menos {self.MIN_INTERVAL}. "
                    "Tente novamente."
                )
                continue

            self.end_datetime = end_dt
            break

        self._confirmar_selecao()

        return {'start': self.start_datetime, 'end': self.end_datetime}

    def _confirmar_selecao(self):
        self.clear_console()
        print(
            "Período selecionado:\n"
            f"  Início: {self.start_datetime.strftime(self.DATE_FORMAT)}\n"
            f"  Término: {self.end_datetime.strftime(self.DATE_FORMAT)}\n"
            f"  Duração: {self.end_datetime - self.start_datetime}\n\n"
            "Pressione Enter para continuar..."
        )
        input()

# Exemplo de uso:
if __name__ == "__main__":
    selector = DateTimeSelector()
    selector.selecionar_periodo()
