import numpy as np
import matplotlib.pyplot as plt
from numpy.random import uniform
from time import sleep

# Параметры параллельного контура
R_true = 11          # Сопротивление в Омах
L_true = 100         # Индуктивность в Генри
C_true = 0.015       # Емкость в Фарадах
I0_const = 0.05      # Амплитуда источника тока
w = 0.09             # Угловая частота источника тока
E0_const = 400       # Амплитуда источника напряжения

# Функции производных для напряжения и тока
def derivatives_U(state, t, R, L, C):
    V, dVdt = state
    b = 1 / (2 * R * C)
    w02 = 1 / (L * C)
    dV = dVdt
    ddVdt = (I0_const * np.cos(w * t) / C) - (2 * b * dVdt) - (w02 * V)
    return np.array([dV, ddVdt])

def derivatives_I(state, t, R, L, C):
    q, I = state
    b = R / (2 * L)
    w02 = 1 / (L * C)
    dq = I
    dI = (E0_const * np.cos(w * t) / L) - (2 * b * I) - (w02 * q)
    return np.array([dq, dI])

# Шаг Рунге-Кутты 4-го порядка
def rk4_step(func, state, h, t, R, L, C):
    k1 = func(state, t, R, L, C)
    k2 = func(state + h * 0.5 * k1, t + 0.5 * h, R, L, C)
    k3 = func(state + h * 0.5 * k2, t + 0.5 * h, R, L, C)
    k4 = func(state + h * k3, t + h, R, L, C)
    return state + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Начальные условия
def initial_conditions(R, L, C):
    phi_U = np.arctan((1 / (w * L) - w * C) * R)
    dVdt0 = I0_const * np.cos(w * 0 + phi_U) / np.sqrt(1 / R**2 + (w * C - 1 / (w * L))**2)
    V0 = I0_const * np.cos(w * 0 + phi_U) / (w * np.sqrt(1 / R**2 + (w * C - 1 / (w * L))**2))

    phi_I = np.arctan(R / (w * L - 1 / (w * C)))
    omega_I = -(phi_I + np.pi / 2)
    I0 = E0_const * np.cos(w * 0 - omega_I) / np.sqrt(R**2 + (w * L - 1 / (w * C))**2)
    q0 = E0_const * np.cos(w * 0 + phi_I) / (w * np.sqrt(R**2 + (w * L - 1 / (w * C))**2))
    return [V0, dVdt0], [q0, I0]

# Генерация "наблюдаемых" данных
def generate_data(t, R, L, C):
    V0, dVdt0 = initial_conditions(R, L, C)[0]
    q0, I0 = initial_conditions(R, L, C)[1]

    results_U = np.zeros((len(t), 2))
    results_I = np.zeros((len(t), 2))
    results_U[0] = [V0, dVdt0]
    results_I[0] = [q0, I0]

    for i in range(1, len(t)):
        results_I[i] = rk4_step(derivatives_I, results_I[i - 1], h, t[i], R, L, C)
        results_U[i] = rk4_step(derivatives_U, results_U[i - 1], h, t[i], R, L, C)

    return results_I[:, 0], results_U[:, 0]

# Решение системы линейных уравнений через LU-разложение
def solve_linear_system(A, b):
    n = len(b)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))
        for k in range(i, n):
            L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    return x

# Метод Ньютона-Гаусса: только для тока
def gauss_newton_I(t, observed_I, initial_guess, max_iter=10, tol=1e-6):
    params = np.array(initial_guess)
    for iteration in range(max_iter):
        R, L, C = params
        model_I, _ = generate_data(t, R, L, C)

        residuals = observed_I - model_I

        # Построение матрицы Якоби
        J = np.zeros((len(t), 3))
        delta = 1e-6
        for i, p in enumerate(params):
            dp = np.zeros_like(params)
            dp[i] = delta
            model_I_plus, _ = generate_data(t, *(params + dp))
            model_I_minus, _ = generate_data(t, *(params - dp))

            J[:, i] = (model_I_plus - model_I_minus) / (2 * delta)

        JTJ = J.T @ J
        JTr = J.T @ residuals
        delta_params = solve_linear_system(JTJ, JTr)
        params += delta_params

        if np.linalg.norm(delta_params) < tol:
            break

        print(f"Iteration {iteration + 1} (I), params: {params}")
        line_model.set_ydata(model_I)
        plt.draw()
        plt.pause(1)

    return params


# Метод Ньютона-Гаусса: только для напряжения
def gauss_newton_U(t, observed_U, initial_guess, max_iter=10, tol=1e-6):
    params = np.array(initial_guess)
    for iteration in range(max_iter):
        R, L, C = params
        _, model_U = generate_data(t, R, L, C)

        residuals = observed_U - model_U

        # Построение матрицы Якоби
        J = np.zeros((len(t), 3))
        delta = 1e-6
        for i, p in enumerate(params):
            dp = np.zeros_like(params)
            dp[i] = delta
            _, model_U_plus = generate_data(t, *(params + dp))
            _, model_U_minus = generate_data(t, *(params - dp))

            J[:, i] = (model_U_plus - model_U_minus) / (2 * delta)

        JTJ = J.T @ J
        JTr = J.T @ residuals
        delta_params = solve_linear_system(JTJ, JTr)
        params += delta_params

        if np.linalg.norm(delta_params) < tol:
            break

        print(f"Iteration {iteration + 1} (U), params: {params}")
        line_model2.set_ydata(model_U)
        plt.draw()
        plt.pause(1)

    return params


# Главная функция
# Главная функция
if __name__ == "__main__":
    start, end, h = 0, 300, 0.1
    points = np.arange(start, end, h)

    # Генерация наблюдаемых данных
    noise_data_I, noise_data_U = generate_data(points, R_true, L_true, C_true)
    noise_data_I += np.random.normal(0, 0.01, size=len(noise_data_I))
    noise_data_U += np.random.normal(0, 0.01, size=len(noise_data_U))

    initial_guess = [R_true * uniform(0.99, 1.01), L_true * uniform(0.99, 1.01), C_true * uniform(0.99, 1.01)]
    print(f"Initial guess: {initial_guess}")

    # Визуализация данных
    plt.figure(figsize=(12, 6))

    # Наблюдаемые данные
    plt.plot(points, noise_data_I, label="Наблюдаемый ток", color='cyan', linewidth=1.5)
    plt.plot(points, noise_data_U, label="Наблюдаемое напряжение", color='orange', linewidth=1.5)

    # Модельные данные
    line_model, = plt.plot(points, np.zeros_like(points), '--', label="Модельный ток", color='blue', linewidth=2)
    line_model2, = plt.plot(points, np.zeros_like(points), '--', label="Модельное напряжение", color='red', linewidth=2)

    plt.legend()
    plt.grid()
    plt.xlabel("Время (t)")
    plt.ylabel("Амплитуда")
    plt.title("Сравнение наблюдаемых и модельных данных для тока и напряжения")

    # Запуск метода Ньютона-Гаусса для тока и напряжения
    print("\nВосстановление параметров по току:")
    fitted_params_I = gauss_newton_I(points, noise_data_I, initial_guess)
    print(f"Recovered parameters (I): R = {fitted_params_I[0]:.4f}, L = {fitted_params_I[1]:.4f}, C = {fitted_params_I[2]:.6f}")

    print("\nВосстановление параметров по напряжению:")
    fitted_params_U = gauss_newton_U(points, noise_data_U, initial_guess)
    print(f"Recovered parameters (U): R = {fitted_params_U[0]:.4f}, L = {fitted_params_U[1]:.4f}, C = {fitted_params_U[2]:.6f}")

    plt.show()
