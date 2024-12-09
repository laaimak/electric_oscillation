from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from numpy.ma.core import arctan
from time import sleep

# Параметры параллельного контура
R_true = 11          # Сопротивление в Омах
L_true = 100         # Индуктивность в Генри
C_true = 0.015       # Емкость в Фарадах
I0_const = 0.05 # Амплитуда источника тока
w = 0.09        # Угловая частота источника тока
E0_const = 400

# Генерация возмущений на параметры
R_obs = R_true * 1.01  # +1% к R
L_obs = L_true * 1.01  # +1% к L
C_obs = C_true * 1.01  # +1% к C

# Функции производных для напряжения и тока
def derivatives_U(state, t, R, L, C):
    V, dVdt = state
    b = 1 / (2 * R * C)
    w02 = 1 / (L * C)
    dV = dVdt
    ddVdt = (I0_const * cos(w * t) / C) - (2 * b * dVdt) - (w02 * V)
    return np.array([dV, ddVdt])

def derivatives_I(state, t, R, L, C):
    q, I = state
    b = R / (2 * L)
    w02 = 1 / (L * C)
    dq = I
    dI = (E0_const * cos(w * t) / L) - (2 * b * I) - (w02 * q)
    return np.array([dq, dI])

# Шаг Рунге-Кутты 4-го порядка
def rk4_step_U(state, h, t, R, L, C):
    k1 = derivatives_U(state, t, R, L, C)
    k2 = derivatives_U(state + h * 1/2 * k1, t + h * 1/2, R, L, C)
    k3 = derivatives_U(state + h * 1/2 * k2, t + h * 1/2, R, L, C)
    k4 = derivatives_U(state + h * 1 * k3, t + h * 1, R, L, C)
    return state + h * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)

def rk4_step_I(state, h, t, R, L, C):
    k1 = derivatives_I(state, t, R, L, C)
    k2 = derivatives_I(state + h * 1/2 * k1, t + h * 1/2, R, L, C)
    k3 = derivatives_I(state + h * 1/2 * k2, t + h * 1/2, R, L, C)
    k4 = derivatives_I(state + h * 1 * k3, t + h * 1, R, L, C)
    return state + h * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4)

# Начальные условия
def initial_conditions(R,L,C):
    phi_U = arctan((1 / (w * L) - w * C) * R)
    dVdt0 = I0_const * cos(w * 0 + phi_U) / (sqrt(1 / R**2 + pow((w * C - 1 / (w * L)), 2)))
    V0 = I0_const * cos(w * 0 + phi_U) / (w * sqrt(1 / R**2 + pow((w * C - 1 / (w * L)), 2)))

    phi_I = arctan(R / (w * L - 1 / (w * C)))
    omega_I = -(phi_I + pi / 2)
    I0 = E0_const * cos(w * 0 - omega_I) / (sqrt(pow(R, 2) + pow((w * L - 1 / (w * C)), 2)))
    q0 = E0_const * cos(w * 0 + phi_I) / (w * sqrt(pow(R, 2) + pow((w * L - 1 / (w * C)), 2)))
    return [V0, dVdt0], [q0, I0]

# Функция для расчета невязок
def residuals(params, t, observed_I, observed_U):
    R, L, C = params
    V0, dVdt0 = initial_conditions(R,L,C)[0]
    q0, I0 = initial_conditions(R,L,C)[1]

    results_U = np.zeros((len(t), 2))
    results_I = np.zeros((len(t), 2))
    results_U[0] = [V0, dVdt0]
    results_I[0] = [q0, I0]

    for i in range(1, len(t)):
        results_I[i] = rk4_step_I(results_I[i - 1], h, t[i], R, L, C)
        results_U[i] = rk4_step_U(results_U[i - 1], h, t[i], R, L, C)

    model_I = results_I[:, 0]
    model_U = results_U[:, 0]
    line_model.set_ydata(results_I[:, 0])
    line_model2.set_ydata(results_U[:, 0])
    plt.title(f"R={params[0]:.4f}, L={params[1]:.4f}, C={params[2]:.6f}")
    plt.pause(0.1)
    # print([observed_I - model_I, observed_U - model_U])
    return np.concatenate([observed_I - model_I, observed_U - model_U])

# Главная функция
if __name__ == "__main__":
    start = 0
    end = 300
    h = 0.1
    points = np.arange(start, end, h)

    # Получаем начальные условия
    V0, dVdt0 = initial_conditions(R_true,L_true,C_true)[0]
    q0, I0 = initial_conditions(R_true,L_true,C_true)[1]

    # Генерация "наблюдаемых" данных с возмущенными параметрами
    results_U_obs = np.zeros((len(points), 2))
    results_I_obs = np.zeros((len(points), 2))
    results_U_obs[0] = [V0, dVdt0]
    results_I_obs[0] = [q0, I0]

    for i in range(1, len(points)):
        results_I_obs[i] = rk4_step_I(results_I_obs[i - 1], h, points[i], R_obs, L_obs, C_obs)
        results_U_obs[i] = rk4_step_U(results_U_obs[i - 1], h, points[i], R_obs, L_obs, C_obs)

    # Добавляем шум
    noise_data_I = results_I_obs[:, 0] + np.random.normal(0, 0.1, size=len(results_I_obs[:, 0]))
    noise_data_U = results_U_obs[:, 0] + np.random.normal(0, 0.1, size=len(results_U_obs[:, 0]))
    
    
    
    # Графики
    plt.figure(figsize=(12, 6))
    plt.plot(points, noise_data_I, label="Наблюдаемый ток с шумом")
    plt.plot(points, noise_data_U, label="Наблюдаемое напряжение с шумом")
    line_model, = plt.plot(points, np.zeros_like(points), label="Модельный ток", color='blue')
    line_model2, = plt.plot(points, np.zeros_like(points), label="Модельное напряжение", color='red')

    plt.legend()
    plt.grid()
    sleep(5)
    # Начальные приближения для оптимизации
    params_initial = [15, 320, 0.1]  # Начальные догадки

    # Решаем обратную задачу методом Гаусса-Ньютона
    result = least_squares(residuals, params_initial, args=(points, noise_data_I, noise_data_U))

    # Выводим восстановленные параметры
    R_fit, L_fit, C_fit = result.x
    print(f"Восстановленные параметры: R = {R_fit:.4f}, L = {L_fit:.4f}, C = {C_fit:.6f}")

    plt.show()