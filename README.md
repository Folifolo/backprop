# Практическая работа №1: Реализация метода обратного распространения ошибки для двухслойной полностью связанной нейронной сети

## Задача
Требуется вывести расчетные формулы и спроектировать программную реализацию метода обратного распространения ошибки для двухслойной полносвязной нейронной сети. Обучение и тестирование сети происходит на наборе данных MNIST, функция активации скрытого слоя – relu, функция активации выходного слоя – softmax, функция ошибки – кросс-энтропия.
![]()
## Математическая модель
Модель нейрона описывается следующими уравнениями: 

![](https://latex.codecogs.com/gif.latex?u_k%3D%5Csum%5Climits_%7Bj%3D1%7D%5Enw_%7Bk%2Cj%7Dx_j%2C%5Cqquad%20y_k%3D%5Cvarphi%20%28u_k&plus;b_k%29)

где ![](https://latex.codecogs.com/gif.latex?x_j) – входной сигнал, ![](https://latex.codecogs.com/gif.latex?w_%7Bk%2Cj%7D) – синаптический вес сигнала ![](https://latex.codecogs.com/gif.latex?x_j), ![](https://latex.codecogs.com/gif.latex?%5Cvarphi) – функция активации, ![](https://latex.codecogs.com/gif.latex?b_k) – смещение

### Прямой ход
Для получения предсказания сети, производится прямой ход: для каждого нейрона последовательно, от начальных слоёв к конечным, вычисляется линейная активация входных сигналов, к ней применяется функция активации, после чего этот сигнал передаётся на следующий слой. В случае данной архитектуры:

![](https://latex.codecogs.com/gif.latex?v_s%3D%5Cvarphi%5E%7B%281%29%7D%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7BN%7Dw_%7Bsi%7D%5E%7B%281%29%7Dx_i%5Cright%29)

![](https://latex.codecogs.com/gif.latex?u_j%3D%5Cvarphi%5E%7B%282%29%7D%5Cleft%28%5Csum_%7Bs%3D0%7D%5E%7BK%7Dw_%7Bjs%7D%5E%7B%282%29%7Dv_s%5Cright%29%3D%5Cvarphi%5E%7B%282%29%7D%5Cleft%28%5Csum_%7Bs%3D0%7D%5E%7BK%7Dw_%7Bjs%7D%5E%7B%282%29%7D%5Cvarphi%5E%7B%281%29%7D%5Cleft%28%5Csum_%7Bi%3D1%7D%5E%7BN%7Dw_%7Bsi%7D%5E%7B%281%29%7Dx_i%5Cright%29%5Cright%29)

где ![](https://latex.codecogs.com/gif.latex?v_s) – выход скрытого слоя, ![](https://latex.codecogs.com/gif.latex?%5Cvarphi%5E%7B%281%29%7D) – функция активации скрытого слоя (relu), ![](https://latex.codecogs.com/gif.latex?x_j) – вход сети, ![](https://latex.codecogs.com/gif.latex?u_j) – выход сети, ![](https://latex.codecogs.com/gif.latex?%5Cvarphi%5E%7B%282%29%7D) – функция активации выходного слоя (softmax), 

### Метод обратного распространения ошибки

Метод обратного распространения ошибки определяет стратегию выбора весов сети 𝑤 с использованием градиентных методов оптимизации.

Схема обратного распространения ошибки состоит из следующих этапов:
1. Прямой проход по нейронной сети. На данном этапе вычисляются значения выходных сигналов каждого слоя, а так же производные их функций активации.
2. Вычисление значений целевой функции и её производной.

    Целевая функция – кросс-энтропия, вычисляется как 
    
    ![](https://latex.codecogs.com/gif.latex?E%28w%29%3D-%5Csum%5Climits_%7Bj%3D1%7D%5EMy_j%5Cln%7Bu_j%7D)
    
    где ![](https://latex.codecogs.com/gif.latex?y_j) – ожидаемый выход (метки)
    
    Производную целевой функции по весам можно вывести следующим образом:
    
    ##### По весам второго слоя:
    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bjs%7D%5E%7B%282%29%7D%7D%20%3D%20-%5Csum_%7Bj%27%3D0%7D%5E%7BM%7Dy_%7Bj%27%7D%5Cfrac%7B%5Cpartial%20%5Cln%7Bu_%7Bj%27%7D%7D%7D%7B%5Cpartial%20w_%7Bjs%7D%5E%7B%282%29%7D%7D%3D%20-%5Csum_%7Bj%27%3D0%7D%5E%7BM%7Dy_%7Bj%27%7D%5Cfrac%7B1%7D%7Bu_%7Bj%27%7D%7D%5Cfrac%7B%5Cpartial%20u_%7Bj%27%7D%7D%7B%5Cpartial%20w_%7Bjs%7D%5E%7B%282%29%7D%7D)
    
    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20u_%7Bj%27%7D%7D%7B%5Cpartial%20w_%7Bjs%7D%5E%7B%282%29%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20%5Cvarphi%5E%7B%282%29%7D%28g_%7Bj%27%7D%29%7D%7B%5Cpartial%20g_j%7D%5Cfrac%7B%5Cpartial%20g_%7Bj%7D%7D%7B%5Cpartial%20w_%7Bjs%7D%5E%7B%282%29%7D%7D%3D%5Cvarphi%28g_%7Bj%27%7D%29%28%5Cdelta_%7Bj%2Cj%27%7D%20-%20%5Cvarphi%28g_%7Bj%7D%29%29%20%5Cfrac%7B%5Cpartial%20g_%7Bj%7D%7D%7B%5Cpartial%20w_%7Bjs%7D%5E%7B%282%29%7D%7D)
    
    ![](https://latex.codecogs.com/gif.latex?g_%7Bj%7D%20%3D%20%5Csum_%7Bs%3D0%7D%5E%7BK%7Dw_%7Bij%7D%5E%7B%282%29%7Dv_s%2C%20%5Cqquad%20%5Cfrac%7B%5Cpartial%20g_%7Bj%7D%7D%7B%5Cpartial%20w_%7Bjs%7D%5E%7B%282%29%7D%7D%20%3D%20v_s)
    
    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bjs%7D%5E%7B%282%29%7D%7D%20%3D%20-%5Csum_%7Bj%27%3D0%7D%5E%7BM%7Dy_%7Bj%27%7D%28%5Cdelta%20_%7Bj%27%2Cj%7D-%5Cvarphi%5E%7B%282%29%7D%28g_%7Bj%7D%29%29v_s%20%3D%5Cleft%20%28%20%5Cvarphi%5E%7B%282%29%7D%28g_%7Bj%7D%29%5Csum_%7Bj%27%3D0%7D%5E%7BM%7Dy_%7Bj%27%7D-y_%7Bj%7D%20%5Cright%20%29v_s)
    
    из условия ![](https://latex.codecogs.com/gif.latex?%5Csum_%7Bj%3D0%7D%5E%7BM%7Dy_%7Bj%7D%3D1) получаем
    
    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bjs%7D%5E%7B%282%29%7D%7D%20%3D%20%5Cleft%20%28%20%5Cvarphi%5E%7B%282%29%7D%28g_%7Bj%7D%29-y_%7Bj%7D%20%5Cright%20%29v_s)
    
    #### По весам первого слоя
    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bsi%7D%5E%7B%281%29%7D%7D%20%3D%20-%5Csum_%7Bj%27%3D0%7D%5E%7BM%7Dy_%7Bj%27%7D%5Cfrac%7B%5Cpartial%20%5Cln%7Bu_%7Bj%27%7D%7D%7D%7B%5Cpartial%20w_%7Bsi%7D%5E%7B%281%29%7D%7D%3D%20-%5Csum_%7Bj%27%3D0%7D%5E%7BM%7Dy_%7Bj%27%7D%5Cfrac%7B1%7D%7Bu_%7Bj%27%7D%7D%5Cfrac%7B%5Cpartial%20u_%7Bj%27%7D%7D%7B%5Cpartial%20w_%7Bsi%7D%5E%7B%281%29%7D%7D)
    
    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20u_%7Bj%27%7D%7D%7B%5Cpartial%20w_%7Bsi%7D%5E%7B%281%29%7D%7D%20%3D%20%5Cfrac%7B%5Cpartial%20%5Cvarphi%5E%7B%282%29%7D%28g_%7Bj%27%7D%29%7D%7B%5Cpartial%20g_%7Bj%7D%7D%20%5Cfrac%7B%5Cpartial%20g_%7Bj%7D%7D%7B%5Cpartial%20w_%7Bsi%7D%5E%7B%281%29%7D%7D%20%3D%20%5Cvarphi%5E%7B%282%29%7D%28g_%7Bj%27%7D%29%28%5Cdelta%20_%7Bj%27%2Cj%7D-%5Cvarphi%5E%7B%282%29%7D%28g_%7Bj%7D%29%29%20w_%7Bjs%7D%5E%7B%282%29%7D%5Cfrac%7B%5Cpartial%20%5Cvarphi_%7Bi%7D%5E%7B%281%29%7D%7D%7B%5Cpartial%20w_%7Bsi%7D%5E%7B%281%29%7D%7Dx_i)
    
    ![](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bsi%7D%5E%7B%281%29%7D%7D%20%3D%20%5Cleft%20%28%5Csum_%7Bj%27%3D0%7D%5E%7BM%7D%28y_%7Bj%27%7D-u_%7Bj%27%7D%29w_%7Bj%27s%7D%5E%7B%282%29%7D%20%5Cright%20%29%5Cfrac%7B%5Cpartial%20%5Cvarphi_%7Bi%7D%5E%7B%281%29%7D%7D%7B%5Cpartial%20w_%7Bsi%7D%5E%7B%281%29%7D%7Dx_i)
3. Обратный проход нейронной сети и корректировка весов
4. Повторение этапов 1-3 до выполнения условия останова

## Описание программной реализации
#### Network.py
Содержит реализацию нейронной сети

Класс NN содержит данные и методы для работы с сетью

**Поля класса NN:**

*_input_size* – размер входного слоя

*_hidden_size* – размер скрытого слоя

*_output_size* – размер выходного слоя

*_w1, _b1* – массивы для хранения весов и смещений первого слоя

*_w2, _b2* – массивы для хранения весов и смещений второго слоя

**Методы класса NN:**

*_forward(input)* – прямой проход сети. Возвращает выходной сигнал первого и второго слоя

*_calculate_dE(input, label, output1, output2)* – вычисление градиента функции ошибки. Возвращает градиент функции по весам и биасам первого и второго слоёв

*_backprop(learning_rate, size, dEb1, dEb2)* – корректировка весов сети при помощи посчитанных градиентов

*init_weights()* – инициализация весов нормальным распределением с дисперсией 1/10

*fit(input, label, validate_data = None, batch_size = 100, learning_rate = 0.1, epochs = 100)* – пакетное обучение сети на epochs эпохах, скоростью обучения learning_rate, размером пакета batch_size. Выводит точность и значение целевой функции на каждой эпохе

*predict(input)* – получение предсказания сети

#### utils.py:
Содержит вспомогательные функции.

*relu(X)* – функция relu

*reluD(X)* – производная функции relu

*calcilate_E(predict, label)* – подсчёт функции ошибки на основании предсказания сети и верной разметки

*calculate_acc(prediction ,label)* – посчёт точности на основании предсказания сети и верной разметки

#### main.py
Обучает сеть из класса NN на MNIST с параметрами из аргументов запуска. Измеряет время обучения.

**Аргументы:**

* *--hidden* – количество нейронов в скрытом слое

* *--epochs* – количество эпох обучения

* *--lr* – скорость обучения

* *--batch* – размер пакета

**Как вызывать:**

        python main.py --hidden 30 --epochs 20 --lr 0,1 --batch 100
(в примере указаны параметры по умолчанию)

## Эксперименты

### Размер скрытого слоя: 30, эпох: 20, скорость обучения: 0,1, размер пакета: 100

train accuracy:  0.9729 train error:  0.0913
validate accuracy:  **0.9613** validate error:  0.1238

Time:  28.309726  seconds

### Размер скрытого слоя: 10, эпох: 20, скорость обучения: 0,1, размер пакета: 100

train accuracy:  0.9469 train error:  0.1805
validate accuracy:  **0.9404** validate error:  0.2013

Time:  18.23942  seconds

### Размер скрытого слоя: 10, эпох: 20, скорость обучения: 0,1, размер пакета: 100

train accuracy:  0.9874 train error:  0.0467
validate accuracy:  **0.9721** validate error:  0.0878

Time:  58.233283  seconds

### Размер скрытого слоя: 30, эпох: 50, скорость обучения: 0,1, размер пакета: 100

train accuracy:  0.9875 train error:  0.0418
validate accuracy:  **0.9678** validate error:  0.1216

Time:  70.004101  seconds

### Размер скрытого слоя: 30, эпох: 20, скорость обучения: 0,5, размер пакета: 100

train accuracy:  0.9766 train error:  0.0732
validate accuracy:  **0.9622** validate error:  0.1614

Time:  26.213141  seconds

### Размер скрытого слоя: 30, эпох: 30, скорость обучения: 0,05, размер пакета: 100

train accuracy:  0.971 train error:  0.0998
validate accuracy:  **0.9625** validate error:  0.1225

Time:  40.577051  seconds

### Размер скрытого слоя: 30, эпох: 20, скорость обучения: 0,1, размер пакета: 10

train accuracy:  0.9724 train error:  0.0928
validate accuracy:  **0.9533** validate error:  0.2423

Time:  65.179444  seconds

### Размер скрытого слоя: 30, эпох: 20, скорость обучения: 0,1, размер пакета: 1000

train accuracy:  0.9268 train error:  0.2588
validate accuracy:  **0.9264** validate error:  0.2576

Time:  25.455479  seconds
