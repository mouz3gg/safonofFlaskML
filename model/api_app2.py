from requests import get

height = input('Введите height = ')
weight = input('Введите weight = ')
sex = input('Введите sex(м млм ж) = ')
if sex == 'м':
    sex = 0
elif sex == 'ж':
    sex = 1
print(get(f'http://127.0.0.1:5000/obuv_api', json={'height': height, 'weight': weight, 'sex': sex}).json())
