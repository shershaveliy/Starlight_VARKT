import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
import pandas as pd

import pathlib



# Константы
g = 9.81  # ускорение свободного падения (м/с²)
rho_0 = 1.225  # плотность воздуха на уровне моря (кг/м³)
G = 6.67430e-11  # Гравитационная постоянная
M_kerbin = 5.2915793e22  # Масса Кербина в кг
R_kerbin = 600000  # Радиус Кербина в метрах
A = 10.0  # Площадь поперечного сечения ракеты, m² (примерное значение)
C_d = 3.3 # Коэффициент аэродинамического сопротивления

# Масса и характеристики ступеней
stages = [
    {"wet_mass": 75010, "fuel_mass": 46779, "thrust": 1_410_000, "burn_time": 105, "ejection_force": 200, "area": 10},
    {"wet_mass": 36672, "fuel_mass": 12400, "thrust": 470_000, "burn_time": 83, "ejection_force": 250, "area": 8},
]

# Функция для расчета плотности воздуха в зависимости от высоты
def air_density(h):
    return rho_0 * np.exp(-h / 4700)

# Функция для расчета угла наклона (pitch) в зависимости от высоты
def calculate_pitch(altitude):
    if altitude < 70000:
        return 90 * (1 - altitude / 70000)  # Чем выше высота, тем меньше наклон
    return 0

# Функция для расчета гравитационного ускорения
def gravitational_acceleration(height):
    r = R_kerbin + height
    return G * M_kerbin / r**2

# Функция для системы уравнений

def rocket_equations(y, t, stage_index):
    # Распаковка состояния
    x_coord, horizontal_velocity, y_coord, vertical_velocity = y
    
    # Получаем данные по текущей ступени
    stage = stages[stage_index]
    fuel_mass = stage["fuel_mass"]
    thrust = stage["thrust"]
    burn_time = stage["burn_time"]
    drain_speed = fuel_mass / burn_time
    ejection_force = stage["ejection_force"]
    area = stage["area"]

    
    cur_mass = start_mass - drain_speed * t
    vel = pow(horizontal_velocity, 2) + pow(vertical_velocity, 2)
    
    pitch = calculate_pitch(y_coord)

    # Расчет гравитационного ускорения и сопротивления
    force_gravity = cur_mass * gravitational_acceleration(y_coord)
    air_density_value = air_density(y_coord)
    drag_force = 0.5 * C_d * air_density_value * vel * area

    # Расчет ускорений
    radius = R_kerbin + y_coord
    centrifugal_force = (cur_mass * horizontal_velocity**2) / radius
    acceleration_vertical = ((thrust - drag_force) * np.sin(np.radians(pitch)) + centrifugal_force - force_gravity) / cur_mass
    acceleration_horizontal = ((thrust - drag_force) * np.cos(np.radians(pitch))) / cur_mass

    # Обновление значений
    dxcoord = horizontal_velocity
    dhorizontal_velocity = acceleration_horizontal
    dycoord = vertical_velocity
    dvertical_velocity = acceleration_vertical
    
    if t == burn_time:
        # После окончания сжигания топлива применяем выброс (ejection)
        dhorizontal_velocity += (ejection_force / cur_mass) * np.cos(np.radians(pitch))
        dvertical_velocity += (ejection_force / cur_mass) * np.sin(np.radians(pitch))

    return [dxcoord, dhorizontal_velocity, dycoord, dvertical_velocity]

# Начальные условия
start_mass = 111_683
initial_conditions = [0, 0, 0, 0]  # x_coord, horizontal_velocity, y_coord, vertical_velocity

# Время интегрирования
time_span = (0, stages[0]["burn_time"])  # Время первой ступени
time_eval = np.linspace(time_span[0], time_span[1], 1000)


# Решение системы уравнений для первой ступени
result_first_stage = odeint(rocket_equations, initial_conditions, time_eval, args=(0,))

time_first_stage = time_eval

start_mass = stages[1]["wet_mass"]
time_span = (0, stages[1]["burn_time"])  # Время второй ступени
time_eval = np.linspace(time_span[0], time_span[1], 1000)


# Решение для второй ступени
result_second_stage = odeint(rocket_equations, result_first_stage[-1, :], time_eval, args=(1,))

time_second_stage = time_eval

# Объединение результатов
time = np.concatenate([time_first_stage, time_first_stage[-1] + time_second_stage])
x_coords = np.concatenate([result_first_stage[:, 0], result_second_stage[:, 0]])
horizontal_velocities = np.concatenate([result_first_stage[:, 1], result_second_stage[:, 1]])
y_coords = np.concatenate([result_first_stage[:, 2], result_second_stage[:, 2]])
vertical_velocities = np.concatenate([result_first_stage[:, 3], result_second_stage[:, 3]])

# Получение данных из симуляции KSP
PATH = str(pathlib.Path().resolve().joinpath("ksp_flight_data.csv"))
data = pd.read_csv(PATH)

time_data_ksp = data['Time']
altitude_data_ksp = data['Altitude']
vertical_velocity_data_ksp = data['Vertical Velocity']
horizontal_velocity_data_ksp = data['Horizontal Velocity']
displacement_data_ksp = data['Displacement']



# Построение графиков
plt.figure(figsize=(15, 10))

# График высоты
plt.subplot(3, 2, 1)
plt.plot(time, y_coords, label='Высота', color='blue')
plt.plot(time_data_ksp, altitude_data_ksp, label='Высота KSP', color='orange')
plt.title('Высота от времени')
plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')
plt.legend()

# График вертикальной скорости
plt.subplot(3, 2, 2)
plt.plot(time, vertical_velocities, label='Скорость по вертикали', color='blue')
plt.plot(time_data_ksp, vertical_velocity_data_ksp, label='Скорость по вертикали KSP', color='orange')
plt.title('Скорость по вертикали от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.legend()

# График горизонтальной скорости
plt.subplot(3, 2, 3)
plt.plot(time, horizontal_velocities, label='Скорость по горизонтали', color='blue')
plt.plot(time_data_ksp, horizontal_velocity_data_ksp, label='Скорость по горизонтали KSP', color='orange')
plt.title('Скорость по горизонтали от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.legend()

# График смещения
plt.subplot(3, 2, 4)
plt.plot(time, x_coords, label='Смещение по горизонтали', color='blue')
plt.plot(time_data_ksp, displacement_data_ksp, label='Смещение по горизонтали KSP',color='orange')
plt.title('Смещение по горизонтали от времени')
plt.xlabel('Время (с)')
plt.ylabel('Смещение по X (м)')
plt.legend()

plt.tight_layout(pad=1.5)
plt.savefig("final.png")
plt.show()
