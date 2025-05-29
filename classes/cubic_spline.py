"""
Essa classe implementa o método de interpolação por splines cúbicos customizado.

Como o eixo X é uma sequência de datetimes, decidimos usar uma sequência de inteiros
para representar os índices dos datetimes.

Entradas:
 - xs: lista de inteiros representando os índices dos datetimes.
 - ys: lista de floats representando os valores correspondentes aos índices.
"""
class CubicSpline:
    xs: list[int]
    ys: list[float]
    n: int
    spaces: int
    equations: int
    H_I = 1
    deltas_y: list[float] = []

    def __init__(self, xs: list, ys: list[float]):
        self.xs = [i for i in range(len(xs))]
        self.ys = ys

        if self.inputs_are_not_valid():
            raise ValueError("The lengths of xs and ys must be the same.")

        self.n = len(self.xs)
        self.spaces = self.n - 1
        self.equations = self.n - 2

        self.deltas_y = [(ys[i + 1] - ys[i]) for i in range(self.n - 1)]

    def inputs_are_not_valid(self):
        if len(self.xs) != len(self.ys):
            return True
        return False
