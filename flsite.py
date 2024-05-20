import pickle
from math import sqrt
from model.fit import *
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify

app = Flask(__name__)

menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"}]

loaded_model_tree = pickle.load(open('model/weather_model', 'rb'))
loaded_model_obuv = pickle.load(open('model/obuv_model', 'rb'))


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2'])]])
        pred = prediction(X_new, train, 3)
        return render_template('lab1.html', title="Метод k -ближайших соседей (KNN)", menu=menu,
                               class_model="Это: " + pred)


@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new2 = np.array([[float(request.form['list1']),
                            float(request.form['list2']),
                            float(request.form['list3'])]])
        pred2 = loaded_model_tree.predict(X_new2)
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + pred2)


@app.route("/p_lab3", methods=['POST', 'GET'])
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new3 = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred3 = str(loaded_model_obuv.predict(X_new3))
        return render_template('lab3.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + pred3)


if __name__ == "__main__":
    app.run(debug=True)
