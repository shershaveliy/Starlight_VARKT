import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
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
x_velocities = np.concatenate([result_first_stage[:, 1], result_second_stage[:, 1]])
y_coords = np.concatenate([result_first_stage[:, 2], result_second_stage[:, 2]])
y_velocities = np.concatenate([result_first_stage[:, 3], result_second_stage[:, 3]])

# Получение данных из симуляции KSP
PATH = str(pathlib.Path().resolve().joinpath("ksp_flight_data.csv"))
data = pd.read_csv(PATH)

time_ksp = data['Time']
x_coords_ksp = data['Displacement']
x_velocities_ksp = data['Horizontal Velocity']
y_coords_ksp = data['Altitude']
y_velocities_ksp = data['Vertical Velocity']

i = 0
while time_ksp[0] > time[i]:
    i += 1

time = time[i:]
x_coords = x_coords[i:]
x_velocities = x_velocities[i:]
y_coords = y_coords[i:]
y_velocities = y_velocities[i:]


# утилиты

def remap(v, x, y, a, b):
    return (v - x) / (y - x) * (b - a) + a

def lerp(t, a, b):
    return a + (b - a) * (1 - t)

assert remap(1, 0, 2, 0, 10) == 5
assert lerp(0.5, 0, 2) == 1

# интерполяция данных
time_remap = []
x_coords_remap = []
x_velocities_remap = []
y_coords_remap = []
y_velocities_remap = []

idx_ksp = 0
for idx in range(0, len(time) - 1):
    ran_out = False
    while time_ksp[idx_ksp + 1] < time[idx]:
        idx_ksp += 1
        if idx_ksp >= len(time_ksp) - 1:
            ran_out = True
            break
    
    if ran_out:
        break

    dt = remap(time[idx], time_ksp[idx_ksp], time_ksp[idx_ksp + 1], 0, 1)
    
    x_coord = lerp(dt, x_coords_ksp[idx_ksp], x_coords_ksp[idx_ksp + 1])
    x_velocity = lerp(dt, x_velocities_ksp[idx_ksp], x_velocities_ksp[idx_ksp + 1])
    y_coord = lerp(dt, y_coords_ksp[idx_ksp], y_coords_ksp[idx_ksp + 1])
    y_velocity = lerp(dt, y_velocities_ksp[idx_ksp], y_velocities_ksp[idx_ksp + 1])
    
    time_remap.append(time[idx])
    x_coords_remap.append(x_coord)
    x_velocities_remap.append(x_velocity)
    y_coords_remap.append(y_coord)
    y_velocities_remap.append(y_velocity)

# Вычисление значений погрешностей

def abs_error(values):
    return abs(values[1] - values[0])

def rel_error(values):
    if values[0] < 1:
        res = abs_error(values) * 100
    else:
        res = abs_error(values) * 100 / values[0]
    return max(min(res, 100), -100)

y_velocities_abs_error = list(map(abs_error, zip(y_velocities, y_velocities_remap)))
y_coords_abs_error = list(map(abs_error, zip(y_coords, y_coords_remap)))
x_velocities_abs_error = list(map(abs_error, zip(x_velocities, x_velocities_remap)))
x_coords_abs_error = list(map(abs_error, zip(x_coords, x_coords_remap)))

y_velocities_rel_error = list(map(rel_error, zip(y_velocities, y_velocities_remap)))
y_coords_rel_error = list(map(rel_error, zip(y_coords, y_coords_remap)))
x_velocities_rel_error = list(map(rel_error, zip(x_velocities, x_velocities_remap)))
x_coords_rel_error = list(map(rel_error, zip(x_coords, x_coords_remap)))

# Построение графиков
plt.figure(figsize=(15, 10))

# График высоты
plt.subplot(3, 2, 1)
plt.plot(time_remap, y_coords_abs_error, label='Погрешность',color='red')
plt.plot(time, y_coords, label='Высота', color='blue')
plt.plot(time_remap, y_coords_remap, label='Высота KSP', color='orange')
plt.title('Высота от времени')
plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')
plt.legend()

# График вертикальной скорости
plt.subplot(3, 2, 2)
plt.plot(time_remap, y_velocities_abs_error, label='Погрешность',color='red')
plt.plot(time, y_velocities, label='Скорость по вертикали', color='blue')
plt.plot(time_remap, y_velocities_remap, label='Скорость по вертикали KSP', color='orange')
plt.title('Скорость по вертикали от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.legend()

# График смещения
plt.subplot(3, 2, 3)
plt.plot(time_remap, x_coords_abs_error, label='Погрешность',color='red')
plt.plot(time, x_coords, label='Смещение по горизонтали', color='blue')
plt.plot(time_ksp, x_coords_ksp, label='Смещение по горизонтали KSP',color='orange')
plt.title('Смещение по горизонтали от времени')
plt.xlabel('Время (с)')
plt.ylabel('Смещение по X (м)')
plt.legend()

# График горизонтальной скорости
plt.subplot(3, 2, 4)
plt.plot(time_remap, x_velocities_abs_error, label='Погрешность',color='red')
plt.plot(time, x_velocities, label='Скорость по горизонтали', color='blue')
plt.plot(time_ksp, x_velocities_ksp, label='Скорость по горизонтали KSP', color='orange')
plt.title('Скорость по горизонтали от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.legend()

# График относительных погрешностей
plt.subplot(3, 2, 5)
plt.plot(time_remap, y_coords_rel_error, label='Высота', color='blue')
plt.plot(time_remap, y_velocities_rel_error, label='Скорость по вертикали',color='orange')
plt.plot(time_remap, x_coords_rel_error, label='Смещение по горизонтали', color='red')
plt.plot(time_remap, x_velocities_rel_error, label='Скорость по горизонтали',color='green')
plt.title('Относительные погрешности')
plt.xlabel('Время (с)')
plt.ylabel('Погрешность (%)')
plt.legend()

plt.tight_layout(pad=1.5)
plt.savefig("final.png")
plt.show()
