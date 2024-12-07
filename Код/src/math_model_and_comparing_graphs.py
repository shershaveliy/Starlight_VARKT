import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

# Константы
g = 9.81  # ускорение свободного падения (м/с²)
rho_0 = 1.225  # плотность воздуха на уровне моря (кг/м³)
G = 6.67430e-11  # Гравитационная постоянная
M_kerbin = 5.2915793e22  # Масса Кербина в кг
R_kerbin = 600000  # Радиус Кербина в метрах
A = 10.0  # Площадь поперечного сечения ракеты, m² (примерное значение)

# Масса и характеристики ступеней
stages = [
    {"wet_mass": 75010, "fuel_mass": 46779, "thrust": 1_410_000, "burn_time": 105, "ejection_force": 200},
    {"wet_mass": 36672, "fuel_mass": 12400, "thrust": 470_000, "burn_time": 83, "ejection_force": 250},
]

# Функция для расчета плотности воздуха в зависимости от высоты
def air_density(h):
    return rho_0 * np.exp(-h / 5000)

# Функция для расчета угла наклона (pitch) в зависимости от высоты
def calculate_pitch(altitude):
    if altitude < 70000:
        return 90 * (1 - altitude / 70000)  # Чем выше высота, тем меньше наклон
    return 0

# Функция для расчета гравитационного ускорения
def gravitational_acceleration(height):
    # Расстояние от центра Кербина до ракеты
    r = R_kerbin + height
    # Гравитационное ускорение на этой высоте
    return G * M_kerbin / r**2

# Коэффициент сопротивления
def get_drag_coefficient(angle_of_attack):
    # Примерная функция, возвращающая коэффициент сопротивления для заданного угла атаки
    return 0.5 + 0.01 * angle_of_attack

def calculate_angle_of_attack(vertical_velocity, horizontal_velocity):
    return np.degrees(np.arctan2(vertical_velocity, horizontal_velocity))

# Инициализация переменных
vertical_velocity = 0
horizontal_velocity = 0
altitude = 0
displacement = 0
time = 0
mass = 111_683

# Открываем файл для записи данных
with open('src/flight_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Altitude", "Vertical Velocity", "Horizontal Velocity",
                     "Total Velocity", "Drag", "Displacement"])

    # Симуляция полета по ступеням
    for i in range(len(stages)):
        
        #Получаем данные по ступеням
        stage = stages[i]
        wet_mass = stage["wet_mass"]
        fuel_mass = stage["fuel_mass"]
        dry_mass = wet_mass - fuel_mass
        
        
        thrust = stage["thrust"]
        burn_time = stage["burn_time"]
        ejection_force = stage["ejection_force"]

        for t in range(burn_time):
            # Рассчет угла наклона
            pitch = calculate_pitch(altitude)

            # Разделение тяги на вертикальную и горизонтальную составляющие
            thrust_vertical = thrust * np.sin(np.radians(pitch))
            thrust_horizontal = thrust * np.cos(np.radians(pitch))
            
            # Радиус от центра Кербина
            radius = R_kerbin + altitude
            
            # Центробежная сила
            centrifugal_force = (mass * horizontal_velocity**2) / radius
            
            # Сила тяжести
            force_gravity = mass * gravitational_acceleration(altitude)

            # Расчет коэффициента сопротивления
            angle_of_attack = calculate_angle_of_attack(vertical_velocity, horizontal_velocity)
            C_drag = 3.3 #get_drag_coefficient(angle_of_attack)
            
            # Плотность воздуха на текущей высоте
            air_density_value = air_density(altitude)  
            
            # Расчет сопротивления
            drag_force = 0.5 * air_density_value * (vertical_velocity**2 + horizontal_velocity**2) * C_drag * (10 if stage == 1 else 8)
            
            # Расчет ускорений
            acceleration_vertical = (thrust_vertical - force_gravity + centrifugal_force -
                                     drag_force * np.sin(np.radians(pitch))) / mass
            acceleration_horizontal = (thrust_horizontal - drag_force * np.cos(np.radians(pitch))) / mass

            # Обновление скоростей, высоты, и горизонтального смещения
            vertical_velocity += acceleration_vertical
            horizontal_velocity += acceleration_horizontal
            altitude += vertical_velocity
            displacement += horizontal_velocity
            
            # Линейное расходование топлива
            mass -= fuel_mass / burn_time 

            # Длина общего вектора скорости
            total_velocity = np.sqrt(vertical_velocity**2 + horizontal_velocity**2)

            # Логгирование данных в файл
            writer.writerow([time, altitude, vertical_velocity, horizontal_velocity, total_velocity, 
                             drag_force, displacement])
            
            # Логгирование данных для графиков
            time += 1

        # Применение силы выброса после сгорания топлива
        vertical_velocity += (ejection_force / mass) * np.sin(np.radians(pitch))
        horizontal_velocity += (ejection_force / mass) * np.cos(np.radians(pitch))
        
        #Сброс сухой массы ступени
        mass -= dry_mass

# Получения данных по мат модели
data = pd.read_csv('src/flight_data.csv')

time_data = data['Time']
altitude_data = data['Altitude']
vertical_velocity_data = data['Vertical Velocity']
horizontal_velocity_data = data['Horizontal Velocity']
total_velocity_data = data['Total Velocity']
drag_data = data['Drag']
displacement_data = data['Displacement']


# Получение данных из симуляции KSP
data = pd.read_csv('src/ksp_flight_data_existing.csv')

time_data_ksp = data['Time']
altitude_data_ksp = data['Altitude']
vertical_velocity_data_ksp = data['Vertical Velocity']
horizontal_velocity_data_ksp = data['Horizontal Velocity']
total_velocity_data_ksp = data['Total Velocity']
drag_data_ksp = data['Drag']
displacement_data_ksp = data['Displacement']


# Построение графиков
plt.figure(figsize=(15, 15))

# График высоты
plt.subplot(3, 2, 1)
plt.plot(time_data, altitude_data, label="Высота (м)")
plt.plot(time_data_ksp, altitude_data_ksp, color='orange')
plt.title('Высота от времени')
plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')

# График вертикальной скорости
plt.subplot(3, 2, 2)
plt.plot(time_data, vertical_velocity_data, label='Скорость по вертикали', color='green')
plt.plot(time_data_ksp, vertical_velocity_data_ksp, label='Скорость по вертикали ksp', color='red')
plt.title('Скорость по вертикали от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.legend()

# График горизонтальной скорости
plt.subplot(3, 2, 3)
plt.plot(time_data, horizontal_velocity_data, label='Скорость по горизонтали', color='blue')
plt.plot(time_data_ksp, horizontal_velocity_data_ksp, label='Скорость по горизонтали ksp', color='orange')
plt.title('Скорость по горизонтали от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.legend()

# График скорости
plt.subplot(3, 2, 4)
plt.plot(time_data, total_velocity_data, label="Скорость (м/с)")
plt.plot(time_data_ksp, total_velocity_data_ksp, color='orange')
plt.title('Скорость от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')

# График сопротивления
plt.subplot(3, 2, 5)
plt.plot(time_data, drag_data, label="Сопротивление (Н)")
plt.plot(time_data_ksp, drag_data_ksp, color='orange')
plt.title('Сопротивление от времени')
plt.xlabel('Время (с)')
plt.ylabel('Сопротивление (Н)')

# График смещения
plt.subplot(3, 2, 6)
plt.plot(time_data, displacement_data)
plt.plot(time_data_ksp, displacement_data_ksp, color='orange')
plt.title('Смещение по горизонтали от времени')
plt.xlabel('Время (с)')
plt.ylabel('Смещение по X (м)')


plt.tight_layout()
plt.show()