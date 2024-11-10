from math import *
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import arctan

# Параметры контура
R = 11              # Сопротивление в Омах
L = 100             # Индуктивность в Генри
C = 0.015           # Емкость в Фарадах
I0_const = 0.05     # Амплитуда источника тока
w = 0.3             # Угловая частота источника тока
E0_const = 400

def derivatives_U(state, t):
    V, dVdt = state
    b = 1 / (2 * R * C)
    w02 = 1 / (L * C)
    dV = dVdt
    ddVdt = (I0_const * cos(w * t) / C) - (2 * b * dVdt) - (w02 * V)
    return np.array([dV, ddVdt])

def derivatives_I(state, t):
    q, I = state
    b = R/(2*L)
    w02 = 1/(L*C)
    dq = I
    dI = (E0_const * cos(w * t) / L) - (2 * b * I) - (w02 * q)
    return np.array([dq, dI])


def rk4_step_U(state, h, t):
    k1 = derivatives_U(state, t)
    k2 = derivatives_U(state + h * 1/2 * k1, t + h * 1/2)
    k3 = derivatives_U(state + h * 1/2 * k2, t + h * 1/2)
    k4 = derivatives_U(state + h * 1 * k3, t + h * 1)
    new_state = np.array(
        state + h * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4))
    return new_state

def rk4_step_I(state, h, t):
    k1 = derivatives_I(state, t)
    k2 = derivatives_I(state + h * 1/2 * k1, t + h * 1/2)
    k3 = derivatives_I(state + h * 1/2 * k2, t + h * 1/2)
    k4 = derivatives_I(state + h * 1 * k3, t + h * 1)
    new_state = np.array(
        state + h * (1/6 * k1 + 1/3 * k2 + 1/3 * k3 + 1/6 * k4))
    return new_state

def main():
    start = 0
    end = 300
    h = 0.1  # Шаг интегрирования
    points = np.arange(start, end, h)
    num_steps = len(points)

    # Начальные значения напряжения и его производной
    phi_U = arctan((1 / (w * L)-w * C) * R)
    omega_U = -(phi_U + pi/2)
    dVdt0 = I0_const * cos(w * start + phi_U) / (sqrt(1 / R **
                                            2 + pow((w * C - 1 / (w * L)), 2)))
    V0 = I0_const * sin(w * start + phi_U) / (w*(sqrt(1 / R **
                                            2 + pow((w * C - 1 / (w * L)), 2))))
    
    phi_I = arctan(R/(w*L - 1/(w*C)))
    omega_I = -(phi_I + pi/2)
    I0 = E0_const * cos(w * start - omega_I) / \
        (sqrt(pow(R, 2) + (pow(w*L - 1/(w*C), 2))))
    q0 = E0_const * cos(w * start + phi_I) / \
        (w * sqrt(pow(R, 2) + (pow(w*L - 1/(w*C), 2))))


    # Инициализация массива результатов
    results_U = np.zeros((num_steps, 2))
    results_I = np.zeros((num_steps, 2))
    results_U[0] = np.array([V0, dVdt0])
    results_I[0] = np.array([q0, I0])
    

    # Решение системы методом Рунге-Кутты
    for i in range(1, len(results_I)):
        results_I[i] = rk4_step_I(results_I[i - 1], h, points[i])
        results_U[i] = rk4_step_U(results_U[i - 1], h, points[i])

    # Построение графиков
    plt.figure(figsize=(12, 6))

    # Графики напряжения и тока
    plt.subplot(2, 1, 1)
    plt.plot(points, results_I[:, 0], label='Ток I(t)', color='b')
    plt.plot(points, results_U[:, 0], label='Напряжение V(t)', color='r')
    plt.xlabel('Время t (с)')
    plt.ylabel('Амплитуда')
    plt.title('Напряжение и ток в параллельном и последовательном RLC контуре')
    plt.legend()
    plt.grid(True)

    # Графики с шумом
    plt.subplot(2, 1, 2)
    noise_data_I = np.array([v + np.random.normal(0, 0.9) for v in results_I[:, 0]])
    noise_data_U = np.array([v + np.random.normal(0, 0.9) for v in results_U[:, 0]])
    plt.plot(points, noise_data_U, label='Напряжение с шумом', color='r')
    plt.plot(points, noise_data_I, label='Напряжение с шумом', color='b')
    plt.xlabel('Время t (с)')
    plt.ylabel('Амплитуда')
    plt.title('Напряжение в параллельном RLC контуре с шумом')
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
