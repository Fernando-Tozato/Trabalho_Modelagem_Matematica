"""
Essa classe calcula os coeficientes do spline cúbico e permite a interpolação.

Como o eixo X é uma sequência de datetimes, decidimos usar uma sequência
igualmente espaçada para representar os índices dos datetimes.

Entradas:
 - xs: lista de inteiros representando os índices dos datetimes.
 - ys: lista de floats representando os valores correspondentes aos índices.
"""
class CubicSpline:
    xs: list[float]
    ys: list[float]
    n: int
    H_I = 1 # Espaçamento entre os pontos no eixo X, constante para simplificar o cálculo (valores de Xi são igualmente espaçados)
    deltas_y: list[float] = []
    SUPER_DIAGONAL: int = 1  # Super diagonal da matriz do sistema, constante para simplificar o cálculo (gerado a partir de H_I)
    MAIN_DIAGONAL: int = 4 # Diagonal principal da matriz do sistema, constante para simplificar o cálculo (gerado a partir de H_I)
    SUB_DIAGONAL: int = 1 # Sub diagonal da matriz do sistema, constante para simplificar o cálculo (gerado a partir de H_I)
    independent_terms_vector: list[float] = []
    c_prime: list[float] # Coeficientes c' do spline cúbico
    d_prime: list[float] # Coeficientes d' do spline cúbico
    c_internal: list[float] # Coeficientes c interno do spline cúbico

    coef_a: list[float] = []  # Coeficientes a do spline cúbico
    coef_b: list[float] = []  # Coeficientes b do spline cúbico
    coef_c: list[float] = []  # Coeficientes c do spline cúbico
    coef_d: list[float] = []  # Coeficientes d do spline cúbico

    def __init__(self, xs: list[float], ys: list[float]):
        self.xs = xs
        self.ys = ys

        if self.inputs_are_not_valid():
            raise ValueError("The lengths of xs and ys must be the same.")

        self.n = len(self.xs)

        self.c_prime = []
        self.d_prime = []
        self.c_internal = []

        self.deltas_y = [(ys[i + 1] - ys[i]) for i in range(self.n - 1)]
        self.independent_terms_vector = [3 * (self.deltas_y[i + 1] - self.deltas_y[i]) for i in range(self.n - 2)]

        if self.n >= 3:
            self.solve_equation_system()
            self.coef_c = [0.0] + self.c_internal + [0.0]
        else:
            # Spline não aplicável para menos de 3 pontos
            self.coef_c = [0.0] * self.n

        self.coef_a = ys[:-1]  # Coeficientes a são os valores de ys, exceto o último

        for i in range(self.n - 1):
            self.coef_b.append(
                self.deltas_y[i] - (self.coef_c[i + 1] + 2 * self.coef_c[i]) / 3.0
            )
            self.coef_d.append(
                (self.coef_c[i + 1] - self.coef_c[i]) / 3.0
            )


    def inputs_are_not_valid(self):
        if len(self.xs) != len(self.ys):
            return True
        return False

    """
    Essa etapa do algoritmo resolve o sistema de equações para encontrar o coeficiente c interno.
    Sistema de equações = Matriz * Xi = Di
    
    A = subdiagonal
    B = diagonal principal
    C = super diagonal
    D = termos independentes
    X = coeficientes internos c
    
    Ela é divida em duas partes:
    - Eliminação direta: resolve o sistema de equações para encontrar os coeficientes c' e d'.
        quando i=1:
            c'[i] = C / B
            d'[i] = D[0] / B
            
        quando i=2,3,...,n-1:
            c'[i] = C / (B - A * c'[i-1])
            d'[i] = (D[i] - A * d'[i-1]) / (B - A * c'[i-1])
    - Substituição reversa: usa os coeficientes c' e d' para calcular os coeficientes internos c.
        quando i=n:
            X[n] = d'[n]
        quando i=n-1,n-2,...,1:
            X[i] = d'[i] - c'[i] * X[i+1]
    
    Dessa forma, o coeficiente c interno é calculado de forma eficiente, evitando a necessidade de resolver o sistema de equações diretamente.
    """
    def solve_equation_system(self):
        n_eq = self.n - 2

        self.c_prime = [0.0] * n_eq
        self.d_prime = [0.0] * n_eq
        self.c_internal = [0.0] * n_eq

        if (self.n-2) <= 0:
            return

#       -- Eliminação direta --
        self.c_prime[0] = self.SUPER_DIAGONAL / self.MAIN_DIAGONAL
        self.d_prime[0] = self.independent_terms_vector[0] / self.MAIN_DIAGONAL

        # Equações restantes
        for i in range(1, n_eq):
            denom = self.MAIN_DIAGONAL - self.SUB_DIAGONAL * self.c_prime[i - 1]
            self.c_prime[i] = self.SUPER_DIAGONAL / denom
            self.d_prime[i] = (self.independent_terms_vector[i] - self.SUB_DIAGONAL * self.d_prime[i - 1]) / denom

#       -- Substituição reversa --
        self.c_internal[n_eq - 1] = self.d_prime[n_eq - 1]
        for i in range(n_eq - 2, -1, -1):
            self.c_internal[i] = self.d_prime[i] - self.c_prime[i] * self.c_internal[i + 1]

    def interpolated_function(self, x: float) -> float:
        """
        Calcula o valor interpolado para um dado x usando os coeficientes do spline cúbico.
        :param x: Valor de x para o qual se deseja calcular o valor interpolado.
        :return: Valor interpolado correspondente a x.
        """
        if x < self.xs[0] or x > self.xs[-1]:
            raise ValueError("x must be within the range of xs.")

        if x == self.xs[-1]:  # Último ponto
            i = self.n - 2
        else:
            i = int(x)

        dx = x - i
        return (self.coef_a[i] +
                self.coef_b[i] * dx +
                self.coef_c[i] * dx**2 +
                self.coef_d[i] * dx**3)