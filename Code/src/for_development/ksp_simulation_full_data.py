import krpc
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import math

# Подключаемся к игре
conn = krpc.connect(name='Автопилот Венера-7')
vessel = conn.space_center.active_vessel

# Настраиваем файлы для записи данных
time_data = []
altitude_data = []
speed_data = []
drag_data = []
#drag_coefficient_data = []
mass_data = []
pitch_data = []
heading_data = []
available_thrust_vertical_data = []
available_thrust_horizontal_data = []
thrust_vertical_data = []
thrust_horizontal_data = []
available_thrust_data = []
thrust_data = []
x_displacement_data = []
x_velocity_data = []
y_velocity_data = []

with open('src/for_development/ksp_flight_data_full.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'Altitude', 'Speed', 'Drag', 'Drag x', 'Drag y', 'Drag z', 'Mass', 'Pitch', 'Heading',
                     'X Displacement', 'X Velocity', 'Y Velocity', 'Available Thrust Vertical',
                     'Available Thrust Horizontal', 'Thrust Vertical', 'Thrust Horizontal', 'Available Thrust', 'Thrust', 'Lift'])

    # Счетчик времени
    start_time = conn.space_center.ut

    # Начальные позиции для расчета смещения
    initial_position = vessel.position(vessel.orbit.body.reference_frame)

    # Подготовка к запуску
    #vessel.control.sas = True
    vessel.control.rcs = False
    vessel.control.throttle = 1.0
    
    print('Запуск через 3...2...1...')
    vessel.control.activate_next_stage()  # Запуск двигателей первой ступени
    time.sleep(0.3)
    vessel.control.activate_next_stage()  # Отрубаем клеммы
    time.sleep(0.7)
    stage_main_engines = ['', 'R7 B V G D Engine Cluster', 'R7 Block A Engine', 'R7 Block I']
    stage = 1
    
    #vessel.auto_pilot.pid_gains = (0.5, 0.5, 0.5)
    #vessel.auto_pilot.target_roll = float('nan')
    # Основной цикл полета
    while True:
        ut = conn.space_center.ut
        elapsed_time = ut - start_time
        altitude = vessel.flight().mean_altitude
        speed = vessel.flight(vessel.orbit.body.reference_frame).speed
        # Сбор данных
        drag_x, drag_y, drag_z = vessel.flight().drag
        drag = sqrt(drag_x ** 2 + drag_y ** 2 + drag_z ** 2)
        
        lift_x, lift_y, lift_z = vessel.flight().lift
        lift = sqrt(lift_x ** 2 + lift_y ** 2 + lift_z ** 2)
        
        #drag_coefficient = vessel.flight().drag_coefficient
        mass = vessel.mass
        pitch = vessel.flight().pitch  # Текущий угол наклона
        heading = vessel.flight().heading  # Текущий курс

        # Текущее положение и скорость для расчета смещения и горизонтальной/вертикальной скорости
        current_position = vessel.position(vessel.orbit.body.reference_frame)
        velocity = vessel.velocity(vessel.orbit.body.reference_frame)
        
        # Расчет смещения по X в направлении pitch
        horizontal_displacement = current_position[0] - initial_position[0]
        # dx = current_position[0] - initial_position[0]
        # dz = current_position[2] - initial_position[2]
        # horizontal_displacement = math.sqrt(dx**2 + dz**2)
        
        # Скорость в направлениях X и Y
        # x_velocity = velocity[0]
        # y_velocity = velocity[1]
        
        vertical_speed = vessel.flight(vessel.orbit.body.reference_frame).vertical_speed
        horizontal_speed = vessel.flight(vessel.orbit.body.reference_frame).horizontal_speed
        #horizontal_speed = vessel.flight(vessel.orbit.body.reference_frame)
        # Расчет вертикальной и горизонтальной тяги
        available_thrust_vertical = vessel.available_thrust * np.sin(np.radians(pitch))
        available_thrust_horizontal = vessel.available_thrust * np.cos(np.radians(pitch))
        thrust_vertical = vessel.thrust * np.sin(np.radians(pitch))
        thrust_horizontal = vessel.thrust * np.cos(np.radians(pitch))
        available_thrust = vessel.available_thrust
        thrust = vessel.thrust

        # Записываем данные в файл
        writer.writerow([elapsed_time, altitude, speed, drag, drag_x, drag_y, drag_z, mass, pitch, heading,
                         horizontal_displacement, horizontal_speed, vertical_speed, available_thrust_vertical,
                         available_thrust_horizontal, thrust_vertical, thrust_horizontal, available_thrust, thrust, lift])

        # Сохраняем данные для построения графиков
        time_data.append(elapsed_time)
        altitude_data.append(altitude)
        speed_data.append(speed)
        drag_data.append(drag)
        #drag_coefficient_data.append(drag_coefficient)
        mass_data.append(mass)
        pitch_data.append(pitch)
        heading_data.append(heading)
        x_displacement_data.append(horizontal_displacement)
        x_velocity_data.append(horizontal_speed)
        y_velocity_data.append(vertical_speed)
        available_thrust_vertical_data.append(available_thrust_vertical)
        available_thrust_horizontal_data.append(available_thrust_horizontal)
        thrust_vertical_data.append(thrust_vertical)
        thrust_horizontal_data.append(thrust_horizontal)
        available_thrust_data.append(available_thrust)
        thrust_data.append(thrust)
        
        # Гравитационный маневр
        vessel.auto_pilot.target_roll = 0
        vessel.auto_pilot.engage()
        if altitude < 70000:
            target_pitch = 90 * (1 - altitude / 70000)  # Чем выше высота, тем меньше наклон
            vessel.auto_pilot.target_pitch_and_heading(target_pitch, 90)
        else:
            vessel.auto_pilot.target_pitch_and_heading(0, 90)
        # if altitude < 1000:
        #     vessel.auto_pilot.target_pitch_and_heading(90, 90)
        # elif altitude > 1000 and altitude < 10000:
        #     vessel.auto_pilot.target_pitch_and_heading(85, 90) 
        # elif altitude >= 10000 and altitude < 20000:
        #     vessel.auto_pilot.target_pitch_and_heading(60, 90) #75
        # elif altitude >= 20000 and altitude < 30000:
        #     vessel.auto_pilot.target_pitch_and_heading(45, 90) #60
        # elif altitude >= 30000 and altitude < 50000:
        #     vessel.auto_pilot.target_pitch_and_heading(20, 90) #45
        # elif altitude >= 50000 and altitude < 70000:
        #     vessel.auto_pilot.target_pitch_and_heading(5, 90) #30
        # elif altitude >= 70000:
        #     vessel.auto_pilot.target_pitch_and_heading(0, 90)
        
        
        # Функция для проверки, есть ли топливо в текущей ступени
        def stage_has_fuel():
            for engine in vessel.parts.engines:
                if engine.has_fuel and engine.part.title == stage_main_engines[stage]:
                    return True
            print(engine.part.title)
            return False
        
        if not stage_has_fuel():  # Если топлива нет, активируем следующую ступень
            vessel.control.activate_next_stage()
            stage += 1
            print("Stage separation")

        # Проверка на орбиту и завершение
        if vessel.orbit.apoapsis_altitude > 150000:
            vessel.control.throttle = 0.0
            print('Достигнута требуемая апоцентрическая высота.')
            break

        time.sleep(0.1)

# Построение графиков
plt.figure(figsize=(15, 15))

# График высоты
plt.subplot(4, 2, 1)
plt.plot(time_data, altitude_data)
plt.title('Высота от времени')
plt.xlabel('Время (с)')
plt.ylabel('Высота (м)')

# График скорости
plt.subplot(4, 2, 2)
plt.plot(time_data, speed_data)
plt.title('Скорость от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')

# График сопротивления
plt.subplot(4, 2, 3)
plt.plot(time_data, drag_data)
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
plt.plot(time_data, mass_data)
plt.title('Масса от времени')
plt.xlabel('Время (с)')
plt.ylabel('Масса (кг)')

# График угла наклона
plt.subplot(4, 2, 5)
plt.plot(time_data, pitch_data)
plt.title('Угол наклона от времени')
plt.xlabel('Время (с)')
plt.ylabel('Угол наклона (градусы)')

# График курса
plt.subplot(4, 2, 6)
plt.plot(time_data, heading_data)
plt.title('Курс от времени')
plt.xlabel('Время (с)')
plt.ylabel('Курс (градусы)')

# График смещения по X
plt.subplot(4, 2, 7)
plt.plot(time_data, x_displacement_data)
plt.title('Смещение по X от времени')
plt.xlabel('Время (с)')
plt.ylabel('Смещение по X (м)')

# График скоростей по X и Y
plt.subplot(4, 2, 8)
plt.plot(time_data, x_velocity_data, label='Скорость по X')
plt.plot(time_data, y_velocity_data, label='Скорость по Y')
plt.title('Скорости по X и Y от времени')
plt.xlabel('Время (с)')
plt.ylabel('Скорость (м/с)')
plt.legend()

plt.tight_layout()
plt.show()
