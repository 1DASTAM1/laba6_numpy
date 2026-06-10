import numpy as np

print("Задание 1:\n")

dt = np.dtype([('ts', 'int32'), ('les_id', 'int16'), ('area', 'float32'), ('ery', 'float32'), ('mel', 'float32'), ('mois', 'float32')])

data = np.genfromtxt('c:/Users/21das/OneDrive/Desktop/123/Proga/laba6/Code/data.csv', delimiter=',', dtype=dt, skip_header=1, invalid_raise=False)

sizeData = data.shape

print(f"Количество строк в файле: {sizeData[0]}")

sizeByte = data.nbytes
sizeMBute = sizeByte / 1024 / 1024

print(f"Вес файла в байтах: {sizeByte} и мегабайтах: {sizeMBute}")

sumNan = np.isnan(data['area']).sum() + np.isnan(data['ery']).sum() + np.isnan(data['mel']).sum() + np.isnan(data['mois']).sum()
sumInf = np.isinf(data['area']).sum() + np.isinf(data['ery']).sum() + np.isinf(data['mel']).sum() + np.isinf(data['mois']).sum()

print(f"Количество NaN: {sumNan}")
print(f"Количество Inf: {sumInf}")

data['area'] = np.where(np.isnan(data['area']), 0, data['area'])
data['area'] = np.where(np.isinf(data['area']), 0, data['area'])

data['ery'] = np.where(np.isnan(data['ery']), 0, data['ery'])
data['ery'] = np.where(np.isinf(data['ery']), 0, data['ery'])

data['mel'] = np.where(np.isnan(data['mel']), 0, data['mel'])
data['mel'] = np.where(np.isinf(data['mel']), 0, data['mel'])

data['mois'] = np.where(np.isnan(data['mois']), 0, data['mois'])
data['mois'] = np.where(np.isinf(data['mois']), 0, data['mois'])

if (sumNan + sumInf > sizeData[0] * len(data.dtype.names) * 3 / 100):
    print("Педупреждение! NaN и Inf больше 3 % от всего файла")

print("\nЗадание 2:\n")

mask_area = data['area'] < 0
mask_ery = (data['ery'] < 0) | (data['ery'] > 100)
mask_mel = (data['mel'] < 0) | (data['mel'] > 100)
anomaly_sum = np.sum(mask_area | mask_ery | mask_mel)

print(f"Количество строк с анамалиями: {anomaly_sum}")
print(f"Доля строк с анамалиями от общего набора: {anomaly_sum / len(data)}")

data['area'] = np.where(data['area'] < 0, 0, data['area'])
data['ery'] = np.clip(data['ery'], 0, 100)
data['mel'] = np.clip(data['mel'], 0, 100)

print("\nЗадание 3:\n")

groups = np.unique(data['les_id'])

print(f"Количество групп: {len(groups)}")

z_score = np.zeros(len(data))

for g in groups:
    mask = data['les_id'] == g

    count = np.sum(mask)

    mean_area = np.mean(data['area'][mask])
    max_ery = np.max(data['ery'][mask])

    mean_dev = np.mean(np.abs(data['area'][mask] - mean_area))

    print(f"\nГруппа {g}")
    print(f"Количество записей: {count}")
    print(f"Средняя площадь: {mean_area}")
    print(f"Максимальная эритема: {max_ery}")
    print(f"Среднее отклонение: {mean_dev}")

    std_area = np.std(data['area'][mask])

    if std_area != 0:
        z_score[mask] = (data['area'][mask] - mean_area) / std_area

print(f"\nZ-score нормализаци: {z_score}")

print("\nЗадание 4:\n")

k = 25

cs = np.cumsum(np.insert(data['area'], 0, 0))

moving_avg = (cs[k:] - cs[:-k]) / k

moving_avg = np.pad(moving_avg,(k - 1, 0),mode='edge')

mel_diff = np.diff(data['mel'],prepend=data['mel'][0])

print(f"Скользящие средние: {moving_avg}")

new_dt = np.dtype([('ts', 'int32'), ('les_id', 'int16'), ('area', 'float32'), ('ery', 'float32'), ('mel', 'float32'), ('mois', 'float32'), ('mel_diff', 'float32')])

new_data = np.empty(data.shape, dtype=new_dt)

for name in data.dtype.names:
    new_data[name] = data[name]

new_data['mel_diff'] = mel_diff

data = new_data

print(f"\nСкорость изменения заданной характеристики: {data['mel_diff']}")

print("\nЗадание 5:\n")

inflam_density = (data['ery'] / (data['area'] + 1e-8)) #Сколько покраснения приходится на 1 см² поражения

hydration_per_area = data['mois'] / (data['area'] + 1e-8) #Показывает уровень гидратации на единицу площади поражения.

inflam_density = np.nan_to_num(inflam_density, nan=0, posinf=0, neginf=0) 

severity_index = np.nan_to_num(hydration_per_area, nan=0, posinf=0, neginf=0)

print("Созданы признаки:")
print(f"inflam_density: {inflam_density}")
print(f"hydration_per_area: {severity_index}")

print("\nЗадание 6:\n")

additional_mask = (data['ery'] > 50) #Доп условие

result = []

for g in groups:
    mask = ((data['les_id'] == g) & additional_mask)

    vals = data['area'][mask]

    if len(vals) > 0:
        mean_val = np.mean(vals)
        median_val = np.median(vals)
        p90 = np.percentile(vals, 90)

        result.append([g, mean_val, median_val, p90])

result = np.array(result)

print(result)

print("\nЗадание 7:\n")

lag_area = np.roll(data['area'],1)

lag_area[0] = data['area'][0]

delta_area = (data['area'] - lag_area)

growth = np.mean(delta_area > 0)
decline = np.mean(delta_area < 0)

print(f"Доля роста: {growth}")

print(f"Доля падения: {decline}")

signs, counts = np.unique(np.sign(delta_area), return_counts=True)

print("\nРаспределение знаков:")

for s, c in zip(signs, counts):
    print(s, c)

print("\nЗадание 8:\n")

total_replaced = 0

for g in groups:
    mask = data['les_id'] == g

    vals = data['area'][mask]

    q1 = np.percentile(vals, 25)
    q3 = np.percentile(vals, 75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = ((vals < lower) | (vals > upper))

    replace_count = np.sum(outliers)

    total_replaced += replace_count

    median_val = np.median(vals)

    data['area'][mask] = np.where(outliers, median_val, data['area'][mask])

    print(f"Группа {g} : {replace_count}")

print(f"\nОбщая доля заменённых записей: {total_replaced / len(data)}")

print("\nЗадание 9:\n")

logic_mask = ((data['area'] == 0) & ((data['ery'] > 80) | (data['mel'] > 80))) #Площадь поражения равна 0, но при этом наблюдается очень высокая эритемаили или очень высокая пигментация.

violations = np.sum(logic_mask)

print(f"Нарушений: {violations}")

print(f"Доля нарушений: {violations / len(data)}")

data['ery'] = np.where(logic_mask, 50, data['ery']) #50 - нейтральное значение

data['mel'] = np.where(logic_mask, 50, data['mel']) #50 - нейтральное значение

print("\nЗадание 10:\n")

categories, counts = np.unique(data['les_id'], return_counts=True)

freq = counts / len(data)

rare = categories[freq < 0.01]

print(f"Редких категорий: {len(rare)}")

data['les_id'] = np.where(np.isin(data['les_id'], rare), 255, data['les_id'])

print(f"Редкие категории объединены в код 255: {data[data['les_id'] == 255]}")