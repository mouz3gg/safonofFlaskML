from requests import get
wind = input('Введите wind = ')
wet = input('Введите wet = ')
cloud = input('Введите cloud = ')
print(get(f'http://127.0.0.1:5000/weather_api', json={'wind':wind,'wet':wet, 'cloud':cloud}).json())