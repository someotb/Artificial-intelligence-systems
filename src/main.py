from matplotlib import pyplot as plt
from ucimlrepo import fetch_ucirepo

"""
age - Возраст
sex - Пол
cp - Церебральный паралич
trestbps - Артериальное давление в состоянии покоя
chol - Желчь
fbs - Уровень сахара в крови натощак
restecg -
thalach - Максимальное достигнутое сердцебиение
exang - Стенокардия, вызванная физической нагрузкой
oldpeak - Депрессия сегмента ST, вызванная физической нагрузкой
slope -
ca - Количество крупных сосудов (0-3)
thal -
num - Диагностика сердечных заболеваний(0 - 4), где 0 это отсутствие
"""

heart_disease = fetch_ucirepo(id=45)
# access metadata
print(heart_disease.metadata.additional_info.summary, "\n")

# access variable info in tabular format
print(heart_disease.variables, "\n")  # Вызывает наши признаки и их подробное описание

# print(heart_disease.data.features)  # Так вызываем только признаки
# print(heart_disease.data.targets)  # Так вызываем только значения num - диагноз
print(heart_disease.data.original, "\n")  # Так вызывает все вместе

age = heart_disease.data.original.age
sex = heart_disease.data.original.sex
cp = heart_disease.data.original.cp
trestbps = heart_disease.data.original.trestbps
chol = heart_disease.data.original.chol
fbs = heart_disease.data.original.fbs
restecg = heart_disease.data.original.restecg
thalach = heart_disease.data.original.thalach
exang = heart_disease.data.original.exang
oldpeak = heart_disease.data.original.oldpeak
slope = heart_disease.data.original.slope
ca = heart_disease.data.original.ca
thal = heart_disease.data.original.thal
num = heart_disease.data.original.num

print(
    "Features:\n",
    age[:1],
    "\n\n",
    sex[:1],
    "\n\n",
    cp[:1],
    "\n\n",
    trestbps[:1],
    "\n\n",
    chol[:1],
    "\n\n",
    fbs[:1],
    "\n\n",
    restecg[:1],
    "\n\n",
    thalach[:1],
    "\n\n",
    exang[:1],
    "\n\n",
    oldpeak[:1],
    "\n\n",
    slope[:1],
    "\n\n",
    ca[:1],
    "\n\n",
    thal[:1],
    "\n\n",
    num[:1],
    "\n\n",
)

plt.figure(1)
(ages,) = plt.plot(age, num, "ro", label="Age")
(sexs,) = plt.plot(sex, num, "bs", label="Sex")
(cps,) = plt.plot(cp, num, "g^", label="cpc")
(trestbpss,) = plt.plot(trestbps, num, "s", label="trestbps")
(chols,) = plt.plot(chol, num, "bo", label="chol")
(fbss,) = plt.plot(fbs, num, "o", label="fbs")
(restecgs,) = plt.plot(restecg, num, "rs", label="restecg")
(thalachs,) = plt.plot(thalach, num, "r^", label="thalach")
(exangs,) = plt.plot(exang, num, "g^", label="exang")
(oldpeaks,) = plt.plot(oldpeak, num, "or", label="oldpeak")
(slopes,) = plt.plot(slope, num, "gs", label="slope")
(cas,) = plt.plot(ca, num, "bo", label="ca")
(thals,) = plt.plot(thal, num, "gs", label="thal")
plt.legend()
plt.show()
