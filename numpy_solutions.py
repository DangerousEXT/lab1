import numpy as np
import matplotlib.pyplot as plt

'''
Задача 1. Дан набор из p матриц размерностью (n,n) и p векторов размерностью (n,1).
Найти сумму произведений матриц на векторы.
Написать тесты для кода
'''

def sum_prod(X, V):
	"""
	Вычисляет сумму произведений матриц на векторы.
	Параметры:
    X - список массивов размерностью (n, n) - квадратные матрицы
	V - список массивов размерностью (n, 1) - векторы-столбцы
	Возвращает массив размерностью (n, 1) - сумма всех произведений
	Валидация введенных данных реализована подробным образом
	"""
	if not isinstance(X, list) or not isinstance(V, list):
		raise ValueError("X и V должны быть списками")
	if len(X) == 0 or len(V) == 0:
		raise ValueError("X и V не должны быть пустыми")
	if len(X) != len(V):
		raise ValueError("X и V должны иметь одинаковую длину")
	for i, (matrix, vector) in enumerate(zip(X, V)):
		if not isinstance(matrix, np.ndarray) or not isinstance(vector, np.ndarray):
			raise ValueError(f"Элемент {i}: матрица и вектор должны быть numpy массивами")
		if matrix.ndim != 2 or vector.ndim != 2:
			raise ValueError(f"Элемент {i}: матрица должна быть двумерной, вектор должен быть двумерным")
		if matrix.shape[0] != matrix.shape[1]:
			raise ValueError(f"Элемент {i}: матрица должна быть квадратной")
		if vector.shape[1] != 1:
			raise ValueError(f"Элемент {i}: вектор должен быть вектором-столбцом с размерностью (n, 1)")
		if matrix.shape[1] != vector.shape[0]:
			raise ValueError(f"Элемент {i}: несовместимые размерности для матрично-векторного умножения")
	result = np.zeros_like(V[0])
	for matrix, vector in zip(X, V):
		result += np.matmul(matrix, vector)
	return result

def test_sum_prod():
	X = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
	V = [np.array([[1], [2]]), np.array([[2], [1]])]
	result = sum_prod(X, V)
	expected = np.array([[21], [33]])
	assert np.array_equal(result, expected)
	X_single = [np.array([[2, 0], [0, 3]])]
	V_single = [np.array([[4], [5]])]
	result_single = sum_prod(X_single, V_single)
	expected_single = np.array([[8], [15]])
	assert np.array_equal(result_single, expected_single)
	X_3x3 = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])]
	V_3x3 = [np.array([[5], [10], [15]])]
	result_3x3 = sum_prod(X_3x3, V_3x3)
	expected_3x3 = np.array([[5], [10], [15]])
	assert np.array_equal(result_3x3, expected_3x3)
	try:
		sum_prod([], [])
		assert False
	except ValueError:
		pass
	
	try:
		sum_prod([np.array([[1, 2], [3, 4]])], [np.array([[1], [2]]), np.array([[3], [4]])])
		assert False
	except ValueError:
		pass
	try:
		sum_prod([np.array([[1, 2], [3, 4]])], [np.array([[1], [2], [3]])])
		assert False
	except ValueError:
		pass
	try:
		sum_prod([np.array([[1, 2, 3], [4, 5, 6]])], [np.array([[1], [2]])])
		assert False
	except ValueError:
		pass
	try:
		sum_prod(["не список"], [np.array([[1], [2]])])
		assert False
	except ValueError:
		pass
	print("Все тесты задачи 1 пройдены")

test_sum_prod()

'''
Задача 2. Дана матрица M, напишите функцию, которая бинаризует матрицу по некоторому threshold
(то есть, все значения большие threshold становятся равными 1, иначе 0).
Напишите тесты для кода
'''

def binarize(M, threshold=0.5):
	"""
	Бинаризует матрицу по заданному порогу.
	Параметры:
	M: numpy массив - входная матрица любой размерности
	threshold: float - пороговое значение (по умолчанию 0.5)
	Возвращает: numpy массив той же размерности - бинаризованная матрица
	Валидация входных данных реализована.
	"""
	if not isinstance(M, np.ndarray):
		raise ValueError("M должна быть numpy массивом")
	if M.size == 0:
		raise ValueError("M не должна быть пустой")
	if not isinstance(threshold, (int, float)):
		raise ValueError("threshold должен быть числом")
	if not np.issubdtype(M.dtype, np.number):
		raise ValueError("M должна содержать числовые значения")
	
	return (M > threshold).astype(int)

def test_binarize():
	M = np.array([[0.1, 0.6], [0.9, 0.3]])
	result = binarize(M, threshold=0.5)
	expected = np.array([[0, 1], [1, 0]])
	assert np.array_equal(result, expected)
	M_default = np.array([[0.4, 0.5], [0.6, 0.7]])
	result_default = binarize(M_default)
	expected_default = np.array([[0, 0], [1, 1]])
	assert np.array_equal(result_default, expected_default)
	M_3d = np.array([[[0.2, 0.8], [0.9, 0.1]], [[0.6, 0.4], [0.3, 0.7]]])
	result_3d = binarize(M_3d, threshold=0.5)
	expected_3d = np.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
	assert np.array_equal(result_3d, expected_3d)
	M_zeros = np.zeros((2, 2))
	result_zeros = binarize(M_zeros, threshold=0.0)
	expected_zeros = np.zeros((2, 2), dtype=int)
	assert np.array_equal(result_zeros, expected_zeros)
	try:
		binarize([[0.1, 0.2], [0.3, 0.4]])
		assert False
	except ValueError:
		pass
	try:
		binarize(np.array([]))
		assert False
	except ValueError:
		pass
	try:
		binarize(np.array([[1, 2], [3, 4]]), threshold="некорректно")
		assert False
	except ValueError:
		pass
	try:
		binarize(np.array([["a", "b"], ["c", "d"]]))
		assert False
	except ValueError:
		pass
	print("Все тесты задачи 2 пройдены")

test_binarize()

'''
Задача 3. Напишите функцию, которая возвращает уникальные элементы из каждой строки матрицы.
Напишите такую же функцию, но для столбцов.
Напишите тесты для кода
'''

def unique_rows(mat):
    arr = np.array(mat)
    return [np.unique(row).tolist() for row in arr]

def unique_columns(mat):
    arr = np.array(mat)
    return [np.unique(col).tolist() for col in arr.T]

def test_unique_functions():
    mat = [
        [1, 2, 2, 3],
        [4, 4, 5, 6],
        [7, 8, 8, 7]
    ]
    expected_rows = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8]
    ]
    expected_columns = [
        [1, 4, 7],
        [2, 4, 8],
        [2, 5, 8],
        [3, 6, 7]
    ]
    result_rows = unique_rows(mat)
    assert result_rows == expected_rows
    result_columns = unique_columns(mat)
    assert result_columns == expected_columns
    mat2 = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    expected_rows2 = [[1], [1], [1]]
    expected_columns2 = [[1], [1], [1]]
    assert unique_rows(mat2) == expected_rows2
    assert unique_columns(mat2) == expected_columns2
    mat3 = []
    assert unique_rows(mat3) == []
    assert unique_columns(mat3) == []
    print("Все тесты задачи 3 пройдены")

test_unique_functions()

'''
Задача 4. Напишите функцию, которая заполняет матрицу с размерами (m,n) случайными числами, 
распределенными по нормальному закону.
Затем считает мат. ожидание и дисперсию для каждого из столбцов и строк, 
а также строит для каждой строки и столбца гистограмму значений (использовать функцию hist из модуля matplotlib.plot)
'''

import numpy as np
import matplotlib.pyplot as plt


def analyze_normal_matrix(m, n):
	"""
	Заполняет матрицу случайными числами из нормального распределения,
	вычисляет статистики и строит гистограммы.
	Параметры:
	m (int) - количество строк матрицы
	n (int) - количество столбцов матрицы
	Возвращает:
	tuple (матрица, мат.ожидания строк, дисперсии строк, мат.ожидания столбцов, дисперсии столбцов)
	Валидация входных данных написана
	"""
	if not isinstance(m, int) or not isinstance(n, int):
		raise ValueError("m и n должны быть целыми числами")
	if m <= 0 or n <= 0:
		raise ValueError("m и n должны быть положительными числами")
	mat = np.random.normal(loc=0, scale=1, size=(m, n))
	print("Сгенерированная матрица:")
	print(mat)
	row_means = np.mean(mat, axis=1)
	row_vars = np.var(mat, axis=1)
	col_means = np.mean(mat, axis=0)
	col_vars = np.var(mat, axis=0)
	print("Статистика по строкам:")
	for i in range(m):
		print(f"  Строка {i + 1}: мат. ожидание = {row_means[i]:.4f}, дисперсия = {row_vars[i]:.4f}")
	
	print("Статистика по столбцам:")
	for j in range(n):
		print(f"  Столбец {j + 1}: мат. ожидание = {col_means[j]:.4f}, дисперсия = {col_vars[j]:.4f}")
	
	for i in range(m):
		plt.figure(figsize=(8, 5))
		plt.hist(mat[i, :], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
		plt.title(f"Гистограмма для строки {i + 1}")
		plt.xlabel("Значение")
		plt.ylabel("Частота")
		plt.grid(True, linestyle='--', alpha=0.6)
		plt.tight_layout()
		plt.show()
	
	for j in range(n):
		plt.figure(figsize=(8, 5))
		plt.hist(mat[:, j], bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
		plt.title(f"Гистограмма для столбца {j + 1}")
		plt.xlabel("Значение")
		plt.ylabel("Частота")
		plt.grid(True, linestyle='--', alpha=0.6)
		plt.tight_layout()
		plt.show()
	return mat, row_means, row_vars, col_means, col_vars

analyze_normal_matrix(4, 5)

'''
Задача 5. Напишите функцию, которая заполняет матрицу (m,n) в шахматном порядке заданными числами a и b.
Напишите тесты для кода
'''

def chess(m, n, a, b):
	"""
	Создает шахматную матрицу с чередующимися значениями.
	Параметры:
	m: int - количество строк
	n: int - количество столбцов
	a: любой тип - значение для четных позиций (когда i+j четное)
	b: любой тип - значение для нечетных позиций (когда i+j нечетное)
	Возвращает: numpy массив размерностью (m, n) с шахматным расположением значений a и b
	"""
	if not isinstance(m, int) or not isinstance(n, int):
		raise ValueError("m и n должны быть целыми числами")
	if m <= 0 or n <= 0:
		raise ValueError("m и n должны быть положительными числами")
	mat = np.zeros((m, n), dtype=type(a))
	for i in range(m):
		for j in range(n):
			if (i + j) % 2 == 0:
				mat[i, j] = a
			else:
				mat[i, j] = b
	return mat


def test_chess():
	expected1 = np.array([[1, 0],
	                      [0, 1]])
	assert np.array_equal(chess(2, 2, 1, 0), expected1)
	expected2 = np.array([[1, 2, 1],
	                      [2, 1, 2],
	                      [1, 2, 1]])
	assert np.array_equal(chess(3, 3, 1, 2), expected2)
	expected3 = np.array([[5, 9, 5],
	                      [9, 5, 9]])
	assert np.array_equal(chess(2, 3, 5, 9), expected3)
	expected4 = np.array([[7]])
	assert np.array_equal(chess(1, 1, 7, 0), expected4)
	try:
		chess(0, 5, 1, 0)
		assert False
	except ValueError:
		pass
	try:
		chess(5, -3, 1, 0)
		assert False
	except ValueError:
		pass
	try:
		chess(2.5, 3, 1, 0)
		assert False
	except ValueError:
		pass
	try:
		chess(3, "строка", 1, 0)
		assert False
	except ValueError:
		pass
	try:
		chess(-1, -1, 1, 0)
		assert False
	except ValueError:
		pass
	print("Все тесты задачи 5 пройдены")

test_chess()

'''
Задача 6. 
Напишите функцию, которая отрисовывает прямоугольник с заданными размерами (a, b) на изображении размера (m, n), 
цвет фона задайте в схеме RGB, как и цвет прямоугольника. 
Цвета также должны быть параметрами функции. 
Напишите аналогичную функцию но для овала с полуосями a и b. 
Напишите тесты для кода. 
Примечание: уравнение эллипса (границы овала) можно записать как: (x-x0)^2/a^2 + (y-y0)^2/b^2 = 1
'''

def draw_rectangle(a, b, m, n, rectangle_color, background_color):
	"""
	Отрисовывает прямоугольник на изображении с заданными цветами.
	Параметры:
	a: int - ширина прямоугольника
	b: int - высота прямоугольника
	m: int - высота изображения
	n: int - ширина изображения
	rectangle_color: tuple - цвет прямоугольника в RGB (r, g, b), значения 0-255
	background_color: tuple - цвет фона в RGB (r, g, b), значения 0-255
	Возвращает: numpy массив размерностью (m, n, 3) - RGB изображение
	Валидация данных реализована
	"""
	if not all(isinstance(x, int) for x in [a, b, m, n]):
		raise ValueError("a, b, m, n должны быть целыми числами")
	if a <= 0 or b <= 0 or m <= 0 or n <= 0:
		raise ValueError("a, b, m, n должны быть положительными")
	if a > n or b > m:
		raise ValueError("размеры прямоугольника не должны превышать размеры изображения")
	if not (isinstance(rectangle_color, tuple) and len(rectangle_color) == 3):
		raise ValueError("rectangle_color должен быть кортежем из 3 элементов")
	if not (isinstance(background_color, tuple) and len(background_color) == 3):
		raise ValueError("background_color должен быть кортежем из 3 элементов")
	if not all(0 <= c <= 255 for c in rectangle_color + background_color):
		raise ValueError("значения RGB должны быть в диапазоне 0-255")
	image = np.zeros((m, n, 3), dtype=np.uint8)
	image[:, :] = background_color
	start_y = (m - b) // 2
	start_x = (n - a) // 2
	end_y = start_y + b
	end_x = start_x + a
	image[start_y:end_y, start_x:end_x] = rectangle_color
	return image


def draw_ellipse(a, b, m, n, ellipse_color, background_color):
	"""
	Отрисовывает эллипс на изображении с заданными цветами.
	Параметры:
	a: int - полуось эллипса по горизонтали
	b: int - полуось эллипса по вертикали
	m: int - высота изображения
	n: int - ширина изображения
	ellipse_color: tuple - цвет эллипса в RGB (r, g, b), значения 0-255
	background_color: tuple - цвет фона в RGB (r, g, b), значения 0-255
	Возвращает: numpy массив размерностью (m, n, 3) - RGB изображение
	Валидация данных реализована
	"""
	if not all(isinstance(x, int) for x in [a, b, m, n]):
		raise ValueError("a, b, m, n должны быть целыми числами")
	if a <= 0 or b <= 0 or m <= 0 or n <= 0:
		raise ValueError("a, b, m, n должны быть положительными")
	if 2 * a > n or 2 * b > m:
		raise ValueError("размеры эллипса не должны превышать размеры изображения")
	if not (isinstance(ellipse_color, tuple) and len(ellipse_color) == 3):
		raise ValueError("ellipse_color должен быть кортежем из 3 элементов")
	if not (isinstance(background_color, tuple) and len(background_color) == 3):
		raise ValueError("background_color должен быть кортежем из 3 элементов")
	if not all(0 <= c <= 255 for c in ellipse_color + background_color):
		raise ValueError("значения RGB должны быть в диапазоне 0-255")
	image = np.zeros((m, n, 3), dtype=np.uint8)
	image[:, :] = background_color
	x0 = n // 2
	y0 = m // 2
	for y in range(m):
		for x in range(n):
			if ((x - x0) ** 2) / (a ** 2) + ((y - y0) ** 2) / (b ** 2) <= 1:
				image[y, x] = ellipse_color
	return image

def test_draw_rectangle():
	img = draw_rectangle(4, 3, 10, 10, (255, 0, 0), (0, 0, 255))
	assert img.shape == (10, 10, 3)
	assert np.array_equal(img[0, 0], [0, 0, 255])
	assert np.array_equal(img[5, 5], [255, 0, 0])
	img2 = draw_rectangle(2, 2, 5, 5, (100, 150, 200), (50, 75, 100))
	assert img2.shape == (5, 5, 3)
	try:
		draw_rectangle(15, 10, 10, 10, (255, 0, 0), (0, 0, 255))
		assert False
	except ValueError:
		pass
	try:
		draw_rectangle(5, 5, 10, 10, (300, 0, 0), (0, 0, 255))
		assert False
	except ValueError:
		pass
	try:
		draw_rectangle(-5, 5, 10, 10, (255, 0, 0), (0, 0, 255))
		assert False
	except ValueError:
		pass
	print("Все тесты задачи 6 (прямоугольник) пройдены!")


def test_draw_ellipse():
	img = draw_ellipse(3, 2, 10, 10, (0, 255, 0), (255, 255, 255))
	assert img.shape == (10, 10, 3)
	assert np.array_equal(img[0, 0], [255, 255, 255])
	img2 = draw_ellipse(4, 4, 20, 20, (128, 128, 128), (0, 0, 0))
	assert img2.shape == (20, 20, 3)
	try:
		draw_ellipse(10, 10, 10, 10, (0, 255, 0), (255, 255, 255))
		assert False
	except ValueError:
		pass
	try:
		draw_ellipse(5, 5, 10, 10, (0, 255), (255, 255, 255))
		assert False
	except ValueError:
		pass
	print("Все тесты задачи 6 (эллипс) пройдены!")

test_draw_rectangle()
test_draw_ellipse()

'''
Задача 7. Дан некий временной ряд. 
Для данного ряда нужно найти его: математическое ожидание, дисперсию, СКО, найти все локальные максимумы и минимумы
(локальный максимум - это точка, которая больше своих соседних точек, 
а локальный минимум - это точка, которая меньше своих соседей), 
а также вычислить для данного ряда другой ряд, получаемый методом скользящего среднего с размером окна p.
Примечание: метод скользящего среднего подразумевает нахождение среднего из подмножетсва ряда размером p
'''

def analyze_time_series(series, window_size):
	"""
	Анализирует временной ряд: вычисляет статистики, находит экстремумы и скользящее среднее.
	Параметры:
	series: numpy массив - временной ряд
	window_size: int - размер окна для скользящего среднего
	Возвращает:
	Словарь с ключами 'mean', 'variance', 'std', 'local_maxima', 'local_minima', 'moving_average'
	Валидация данных реализована
	"""
	if not isinstance(series, np.ndarray):
		raise ValueError("series должен быть numpy массивом")
	if series.ndim != 1:
		raise ValueError("series должен быть одномерным массивом")
	if len(series) == 0:
		raise ValueError("series не должен быть пустым")
	if not isinstance(window_size, int):
		raise ValueError("window_size должен быть целым числом")
	if window_size <= 0 or window_size > len(series):
		raise ValueError("window_size должен быть положительным и не превышать длину ряда")
	
	mean = np.mean(series)
	variance = np.var(series)
	std = np.std(series)
	local_maxima = []
	local_minima = []
	for i in range(1, len(series) - 1):
		if series[i] > series[i - 1] and series[i] > series[i + 1]:
			local_maxima.append(i)
		elif series[i] < series[i - 1] and series[i] < series[i + 1]:
			local_minima.append(i)
	moving_average = np.convolve(series, np.ones(window_size) / window_size, mode='valid')
	return {
		'mean': mean,
		'variance': variance,
		'std': std,
		'local_maxima': local_maxima,
		'local_minima': local_minima,
		'moving_average': moving_average
	}

'''
Задача 8. Дан некоторый вектор с целочисленными метками классов.
Напишите функцию, которая выполняет one-hot-encoding для данного вектора
One-hot-encoding - представление, в котором на месте метки некоторого класса стоит 1, в остальных позициях стоит 0.
Например для вектора [0, 2, 3, 0] one-hot-encoding выглядит как: 
[[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
'''

def one_hot_encode(labels):
	"""
	Выполняет one-hot-encoding для вектора меток классов.
	Параметры: labels: numpy массив - вектор с целочисленными метками классов
	Возвращает: numpy массив размерностью (n, num_classes) - one-hot представление
	Валидация данных реализована
	"""
	if not isinstance(labels, np.ndarray):
		raise ValueError("labels должен быть numpy массивом")
	if labels.ndim != 1:
		raise ValueError("labels должен быть одномерным массивом")
	if len(labels) == 0:
		raise ValueError("labels не должен быть пустым")
	if not np.issubdtype(labels.dtype, np.integer):
		raise ValueError("labels должен содержать целочисленные значения")
	if np.any(labels < 0):
		raise ValueError("метки классов должны быть неотрицательными")
	
	num_classes = labels.max() + 1
	one_hot = np.eye(num_classes)[labels]
	return one_hot.astype(int)