import krpc
import time
import csv
from math import sqrt
import numpy as np
import pathlib

# Подключаемся к игре
conn = krpc.connect(name='Автопилот Венера-7')
vessel = conn.space_center.active_vessel

# Создаем файл для записи данных
PATH = str(pathlib.Path().resolve().joinpath("ksp_flight_data.csv"))
with open(PATH, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Altitude", "Vertical Velocity", "Horizontal Velocity",
                     "Total Velocity", "Drag", "Displacement"])

    # Счетчик времени
    start_time = conn.space_center.ut

    # Начальная позиция для расчета смещения
    initial_position = vessel.position(vessel.orbit.body.reference_frame)
    # Длина вектора
    initial_position_vec_length = np.linalg.norm(initial_position)
    
    # Подготовка к запуску
    vessel.control.sas = False
    vessel.control.rcs = False
    vessel.control.throttle = 1.0
    
    print('Запуск через 3...2...1...')
    vessel.control.activate_next_stage()  # Запуск двигателей первой ступени
    time.sleep(0.3)
    vessel.control.activate_next_stage()  # Освобождение от стартовых клемм
    time.sleep(0.7)
    
    stage_main_engines = ['', 'R7 B V G D Engine Cluster', 'R7 Block A Engine', 'R7 Block I']
    stage = 1
    
    # Основной цикл полета
    while True:
        
        # Настоящее время
        ut = conn.space_center.ut
        
        # Прошедшее время с начала
        elapsed_time = ut - start_time
        
        # Сбор данных
        altitude = vessel.flight().mean_altitude
        speed = vessel.flight(vessel.orbit.body.reference_frame).speed
        drag_x, drag_y, drag_z = vessel.flight().drag
        drag = sqrt(drag_x ** 2 + drag_y ** 2 + drag_z ** 2)

        # Текущее положение для расчета смещения
        current_position = vessel.position(vessel.orbit.body.reference_frame)
        
        # Расчет смещения
        current_position = current_position / np.linalg.norm(current_position) * initial_position_vec_length
        horizontal_displacement = np.linalg.norm(current_position - initial_position)
        
        # Получение скоростей
        vertical_speed = vessel.flight(vessel.orbit.body.reference_frame).vertical_speed
        horizontal_speed = vessel.flight(vessel.orbit.body.reference_frame).horizontal_speed
        
        # Записываем данные в файл
        writer.writerow([elapsed_time, altitude, vertical_speed, horizontal_speed, speed, drag, horizontal_displacement])

        # Наклон ракеты в зависимости от высоты
        vessel.auto_pilot.target_roll = 0
        vessel.auto_pilot.engage()
        if altitude < 70000:
            target_pitch = 90 * (1 - altitude / 70000)  # Чем выше высота, тем меньше наклон
            vessel.auto_pilot.target_pitch_and_heading(target_pitch, 90)
        else:
            vessel.auto_pilot.target_pitch_and_heading(0, 90)

        
        # Проверяем есть ли топливо в одном из движков ступени
        has_fuel = False
        for engine in vessel.parts.engines:
            if engine.has_fuel and engine.part.title == stage_main_engines[stage]:
                has_fuel = True
                break
        
        # Если топлива нет, активируем следующую ступень
        if not has_fuel:
            vessel.control.activate_next_stage()
            stage += 1
            print("Отделение ступени")
        
        if stage == 3:
            print('Конец')
            vessel.control.throttle = 0.0
            break
        
        time.sleep(0.1)