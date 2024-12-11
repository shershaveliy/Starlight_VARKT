import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import bisect

# Параметры ракеты и атмосферы
g = 9.81  # ускорение свободного падения (м/с²)
rho_0 = 1.225  # плотность воздуха на уровне моря (кг/м³)

# Масса и характеристики ступеней
stages = [
    {"wet_mass": 59410 + 15600, "fuel_mass": 46779, "thrust": 1_410_000, "burn_time": 105, "ejection_force": 200},
    {"wet_mass": 36672, "fuel_mass": 12400, "thrust": 470_000, "burn_time": 83, "ejection_force": 250},
]

# Функция для расчета плотности воздуха в зависимости от высотыs
def air_density(h):
    return rho_0 * np.exp(-h / 5000)

# Функция для расчета угла наклона (pitch) в зависимости от высоты
def calculate_pitch(altitude):
    if altitude < 70000:
        return 90 * (1 - altitude / 70000)  # Чем выше высота, тем меньше наклон
    return 0
    # if altitude < 1000:
    #     return 90
    # elif 1000 <= altitude < 10000:
    #     return 85
    # elif 10000 <= altitude < 20000:
    #     return 60
    # elif 20000 <= altitude < 30000:
    #     return 45
    # elif 30000 <= altitude < 50000:
    #     return 20
    # elif 50000 <= altitude < 70000:
    #     return 5
    # else:
    #     return 0

# Константы
G = 6.67430e-11  # Гравитационная постоянная
M_kerbin = 5.2915793e22  # Масса Кербина в кг
R_kerbin = 600000  # Радиус Кербина в метрах

def gravitational_acceleration(height):
    # Расстояние от центра Кербина до ракеты
    r = R_kerbin + height
    # Гравитационное ускорение на этой высоте
    return G * M_kerbin / r**2

# Инициализация переменных
vertical_velocity = 0
horizontal_velocity = 0
altitude = 0
displacement = 0
time = 0
mass = 111_683
ksp_data = pd.read_csv('src/for_development/ksp_flight_data_full.csv')
time_data_ksp = list(ksp_data['Time'])
drag_data_ksp = list(ksp_data['Drag'])
lift_data_ksp = list(ksp_data['Lift'])

# Открываем файл для записи данных
with open('src/for_development/flight_data_full.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time (s)", "Altitude (m)", "Vertical Velocity (m/s)", "Horizontal Velocity (m/s)",
                     "Total Velocity (m/s)", "Mass (kg)", "Drag", "Pitch", "Heading", "Displacement",
                     "Thrust Horizontal", "Thrust Vertical", "Thrust"])

    # Симуляция полета по ступеням
    for i in range(len(stages)):
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
            
            radius = R_kerbin + altitude  # Радиус от центра Кербина
            centrifugal_force = (mass * horizontal_velocity**2) / radius
            # Расчет сил и ускорений
            
            force_gravity = mass * gravitational_acceleration(altitude)
            
            
            A = 10.0  # Площадь поперечного сечения ракеты, m² (примерное значение)

            # Коэффициенты
            def get_drag_coefficient(angle_of_attack):
                # Примерная функция, возвращающая коэффициент сопротивления для заданного угла атаки
                return 0.5 + 0.01 * angle_of_attack  # Значения нужно уточнять экспериментально или эмпирически

            def get_lift_coefficient(angle_of_attack):
                # Примерная функция, возвращающая коэффициент подъемной силы для заданного угла атаки
                return 0.1 * np.sin(np.radians(angle_of_attack))  # Значения нужно уточнять экспериментально или эмпирически
            
            def calculate_angle_of_attack(vertical_velocity, horizontal_velocity):
                return np.degrees(np.arctan2(vertical_velocity, horizontal_velocity))
            angle_of_attack = calculate_angle_of_attack(vertical_velocity, horizontal_velocity)

            # Расчет коэффициентов
            C_drag = get_drag_coefficient(angle_of_attack)
            C_lift = get_lift_coefficient(angle_of_attack)
            # Расчет аэродинамических сил
            air_density_value = air_density(altitude)  # Плотность воздуха на текущей высоте
            drag_force = 0.5 * air_density_value * (vertical_velocity**2 + horizontal_velocity**2) * C_drag * A
            #lift_force = 0.5 * air_density_value * (vertical_velocity**2 + horizontal_velocity**2) * C_lift * A



            #force_drag = 0.5 * air_density(altitude) * (vertical_velocity**2 + horizontal_velocity**2) * 20  # Лобовое сопротивление
            ind = bisect.bisect_left(time_data_ksp, time)
            drag_force = drag_data_ksp[ind]
            #lift_force = lift_data_ksp[ind]
            
            acceleration_vertical = (thrust_vertical - force_gravity + centrifugal_force - 
                                     drag_force * np.sin(np.radians(pitch))) / mass
            acceleration_horizontal = (thrust_horizontal - drag_force * np.cos(np.radians(pitch))) / mass

            # Обновление скоростей и высоты
            vertical_velocity += acceleration_vertical
            horizontal_velocity += acceleration_horizontal
            altitude += vertical_velocity
            displacement += horizontal_velocity
            mass -= fuel_mass / burn_time  # Линейное расходование топлива

            # Длина общего вектора скорости
            total_velocity = np.sqrt(vertical_velocity**2 + horizontal_velocity**2)

            # Логгирование данных в файл
            writer.writerow([time, altitude, vertical_velocity, horizontal_velocity, total_velocity, 
                             mass, drag_force, pitch, 90, displacement, thrust_horizontal, thrust_vertical, thrust])
            
            # Логгирование данных для графиков
            time += 1

        # Применение силы выброса после сгорания топлива
        vertical_velocity += (ejection_force / mass) * np.sin(np.radians(pitch))
        horizontal_velocity += (ejection_force / mass) * np.cos(np.radians(pitch))
        mass -= dry_mass
        print(f"Stage separation: Applied ejection force of {ejection_force} N")

# Построение графиков на основе данных из файла
data = np.genfromtxt('src/for_development/flight_data_full.csv', delimiter=',', skip_header=1)
time_data = data[:, 0]
altitude_data = data[:, 1]
vertical_velocity_data = data[:, 2]
horizontal_velocity_data = data[:, 3]
total_velocity_data = data[:, 4]
mass_data = data[:, 5]
drag_data = data[:, 6]
pitch_data = data[:, 7]
heading_data = data[:, 8]
displacement_data = data[:, 9]
thrust_horizontal_data = data[:, 10]
thrust_vertical_data = data[:, 11]
thrust_data = data[:, 12]
#lift_data = data[:, 13]


data = pd.read_csv('src/for_development/ksp_flight_data_full.csv')

time_data_ksp = data['Time']
altitude_data_ksp = data['Altitude']
speed_data_ksp = data['Speed']
drag_data_ksp = data['Drag']
drag_x_data_ksp = data['Drag x']
drag_y_data_ksp = data['Drag y']
drag_z_data_ksp = data['Drag z']
mass_data_ksp = data['Mass']
pitch_data_ksp = data['Pitch']
heading_data_ksp = data['Heading']
x_displacement_data_ksp = data['X Displacement']
x_velocity_data_ksp = data['X Velocity']
y_velocity_data_ksp = data['Y Velocity']
thrust_vertical_data_ksp = data['Thrust Vertical']
thrust_horizontal_data_ksp = data['Thrust Horizontal']
available_thrust_vertical_data_ksp = data['Available Thrust Vertical']
available_thrust_horizontal_data_ksp = data['Available Thrust Horizontal']
available_thrust_data_ksp = data['Available Thrust']
thrust_data_ksp = data['Thrust']
lift_data_ksp = data['Lift']


x_velocity_data_ksp_list = list(x_velocity_data_ksp)
time_data_ksp_list = list(ksp_data['Time'])

calculated_displacement_ksp = [0]
displacement = 0
for i in range(1, len(x_velocity_data_ksp_list)):
    displacement += x_velocity_data_ksp_list[i] * (time_data_ksp_list[i] - time_data_ksp_list[i - 1])
    calculated_displacement_ksp.append(displacement)



#'Available Thrust Vertical', 
# 'Available Thrust Horizontal', 'Thrust Vertical', 'Thrust Horizontal', 'Available Thrust', 'Thrust'

# plt.figure(figsize=(10, 5))

# # График скорости
# plt.subplot(1, 2, 1)
# plt.plot(time_data, total_velocity_data, label="Скорость (м/с)")
# plt.plot(data['Time'], data['Speed'], label='Скорость (м/с)', color='orange')
# plt.xlabel("Время (с)")
# plt.ylabel("Скорость (м/с)")
# plt.title("Зависимость скорости от времени")
# plt.grid(True)

# # График высоты
# plt.subplot(1, 2, 2)
# plt.plot(time_data, altitude_data, label="Высота (м)")
# plt.plot(data['Time'], data['Altitude'], label='Высота (м)', color="orange")
# plt.xlabel("Время (с)")
# plt.ylabel("Высота (м)")
# plt.title("Зависимость высоты от времени")
# plt.grid(True)

# plt.tight_layout()
# plt.show()



# Построение графиков
plt.figure(figsize=(15, 15))

# График высоты
plt.subplot(4, 2, 1)
plt.plot(time_data, altitude_data, label="Высота (м)")
plt.plot(time_data_ksp, altitude_data_ksp, color='orange')
plt.title('Высота от времени')
plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')

# График скорости
plt.subplot(4, 2, 2)
plt.plot(time_data, total_velocity_data, label="Скорость (м/с)")
plt.plot(time_data_ksp, speed_data_ksp, color='orange')
plt.title('Скорость от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')

# График сопротивления
plt.subplot(4, 2, 3)
plt.plot(time_data, drag_data)
plt.plot(time_data_ksp, drag_data_ksp, color='orange')
plt.title('Сопротивление от времени')
plt.xlabel('Время (с)')
plt.ylabel('Сопротивление (Н)')

# # График коэффициента сопротивления
# plt.subplot(4, 2, 4)
# plt.plot(time_data, drag_coefficient_data)
# plt.title('Коэффициент сопротивления от времени')
# plt.xlabel('Время (с)')
# plt.ylabel('Сопротивление (Н)')

# График массы
plt.subplot(4, 2, 4)
plt.plot(time_data, mass_data, label="Масса (кг)")
plt.plot(time_data_ksp, mass_data_ksp, color='orange')
plt.title('Масса от времени')
plt.xlabel('Время (с)')
plt.ylabel('Масса (кг)')

# График угла наклона
plt.subplot(4, 2, 5)
plt.plot(time_data, pitch_data)
plt.plot(time_data_ksp, pitch_data_ksp, color='orange')
plt.title('Угол наклона от времени')
plt.xlabel('Время (с)')
plt.ylabel('Угол наклона (градусы)')

# График курса
plt.subplot(4, 2, 6)
plt.plot(time_data, heading_data)
plt.plot(time_data_ksp, heading_data_ksp, color='orange')
plt.title('Курс от времени')
plt.xlabel('Время (с)')
plt.ylabel('Курс (градусы)')

# График смещения по X
plt.subplot(4, 2, 7)
plt.plot(time_data, displacement_data)
plt.plot(time_data_ksp, x_displacement_data_ksp, color='orange')
plt.title('Смещение по X от времени')
plt.xlabel('Время (с)')
plt.ylabel('Смещение по X (м)')

# График скоростей по X и Y
plt.subplot(4, 2, 8)
plt.plot(time_data, horizontal_velocity_data, label='Скорость по X', color='blue')
plt.plot(time_data, vertical_velocity_data, label='Скорость по Y', color='green')
plt.plot(time_data_ksp, x_velocity_data_ksp, label='Скорость по X ksp', color='orange')
plt.plot(time_data_ksp, y_velocity_data_ksp, label='Скорость по Y ksp', color='red')
plt.title('Скорости по X и Y от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.legend()

plt.tight_layout()
plt.show()


print('second graph')
plt.figure(figsize=(15, 15))

# График горизонтальной тяги
plt.subplot(4, 2, 1)
plt.plot(time_data, thrust_horizontal_data)
plt.plot(time_data_ksp, thrust_horizontal_data_ksp, color='orange')
plt.xlabel("Время (с)")
plt.ylabel("Горизонтальная Тяга (Н)")
plt.title("Зависимость горизонтальной тяги от времени")
plt.grid(True)

# График вертикальной тяги
plt.subplot(4, 2, 2)
plt.plot(time_data, thrust_vertical_data)
plt.plot(time_data_ksp, thrust_vertical_data_ksp, color='orange')
plt.xlabel("Время (с)")
plt.ylabel("Вертикальная Тяга (Н)")
plt.title("Зависимость вертикальной тяги от времени")
plt.grid(True)

# График тяги
plt.subplot(4, 2, 3)
plt.plot(time_data, thrust_data)
plt.plot(time_data_ksp, thrust_data_ksp, color='orange')
plt.xlabel("Время (с)")
plt.ylabel("Тяга (Н)")
plt.title("Зависимость тяги от времени")
plt.grid(True)


# График смещения по X 2
plt.subplot(4, 2, 4)
plt.plot(time_data, displacement_data)
plt.plot(time_data_ksp, calculated_displacement_ksp, color='orange')
plt.title('Смещение по X от времени')
plt.xlabel('Время (с)')
plt.ylabel('Смещение по X (м)')

# График сопротивления
plt.subplot(4, 2, 5)
plt.plot(time_data, drag_data)
plt.plot(time_data_ksp, drag_data_ksp, color='orange')
plt.title('Сопротивление drag от времени')
plt.xlabel('Время (с)')
plt.ylabel('Сопротивление drag(Н)')

# График сопротивления
# plt.subplot(4, 2, 6)
# plt.plot(time_data, lift_data)
# plt.plot(time_data_ksp, lift_data_ksp, color='orange')
# plt.title('Сопротивление lift от времени')
# plt.xlabel('Время (с)')
# plt.ylabel('Сопротивление lift(Н)')

plt.tight_layout()
plt.show()

# # График горизонтальной тяги
# plt.subplot(4, 2, 4)
# plt.plot(time_data, thrust_horizontal_data)
# plt.plot(time_data_ksp, available_thrust_horizontal_data_ksp, color='orange')
# plt.xlabel("Время (с)")
# plt.ylabel("Горизонтальная Тяга (Н)")
# plt.title("Зависимость горизонтальной тяги от времени(available)")
# plt.grid(True)

# # График вертикальной тяги
# plt.subplot(4, 2, 5)
# plt.plot(time_data, thrust_vertical_data)
# plt.plot(time_data_ksp, available_thrust_vertical_data_ksp, color='orange')
# plt.xlabel("Время (с)")
# plt.ylabel("Вертикальная Тяга (Н)")
# plt.title("Зависимость вертикальной тяги от времени(available)")
# plt.grid(True)

# # График тяги
# plt.subplot(4, 2, 6)
# plt.plot(time_data, thrust_data)
# plt.plot(time_data_ksp, available_thrust_data_ksp, color='orange')
# plt.xlabel("Время (с)")
# plt.ylabel("Тяга (Н)")
# plt.title("Зависимость тяги от времени(available)")
# plt.grid(True)

# # График тяг
# plt.subplot(4, 2, 7)
# plt.plot(time_data_ksp, thrust_data_ksp)
# plt.plot(time_data_ksp, available_thrust_data_ksp, color='orange')
# plt.xlabel("Время (с)")
# plt.ylabel("Тяга (Н)")
# plt.title("Available и обычная тяги(available)")
# plt.grid(True)

# plt.tight_layout()
# plt.show()



# plt.figure(figsize=(15, 15))

# # График сопротивления
# plt.subplot(4, 2, 1)
# plt.plot(time_data, drag_data)
# plt.plot(time_data_ksp, drag_data_ksp, color='orange')
# plt.title('Сопротивление от времени')
# plt.xlabel('Время (с)')
# plt.ylabel('Сопротивление (Н)')

# # График сопротивления x
# plt.subplot(4, 2, 2)
# plt.plot(time_data_ksp, drag_x_data_ksp, color='orange')
# plt.title('Сопротивление от времени x')
# plt.xlabel('Время (с)')
# plt.ylabel('Сопротивление (Н)')

# # График сопротивления y
# plt.subplot(4, 2, 3)
# plt.plot(time_data, drag_data)
# plt.plot(time_data_ksp, drag_y_data_ksp, color='orange')
# plt.title('Сопротивление от времени y')
# plt.xlabel('Время (с)')
# plt.ylabel('Сопротивление (Н)')

# # График сопротивления z
# plt.subplot(4, 2, 4)
# plt.plot(time_data_ksp, drag_z_data_ksp, color='orange')
# plt.title('Сопротивление от времени z')
# plt.xlabel('Время (с)')
# plt.ylabel('Сопротивление (Н)')


# plt.tight_layout()
# plt.show()