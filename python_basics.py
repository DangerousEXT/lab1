'''
Задача 1
Написать функцию на вход которой подается строка, состоящая из латинских букв. 
Функция должна вернуть количество гласных букв (a, e, i, o, u) в этой строке. 
Написать тесты для кода
'''

vowels = "aeiou"
def count_vowels(line:str) -> int:
    counter = 0
    if not isinstance(line, str): raise ValueError("На вход подается строка")
    for letter in line:
        if letter in vowels:
            counter += 1
    return counter

def test_vowels():
    assert count_vowels("") == 0
    assert count_vowels("!@#$%^&*()01xybcd") == 0
    assert count_vowels("abcdefghijklmnopqrstuvwxyz") == 5
    assert count_vowels("test-case") == 3
    assert count_vowels("   \t\n") == 0
    try:
        count_vowels(123)
        assert False, "Должно было выбросить ValueError"
    except ValueError:
        pass
    print("Все тесты задачи 1 пройдены")

test_vowels()

'''
Задача 2
Написать функцию на вход, которой подается строка. 
Функция должна вернуть true, если каждый символ в строке встречается только 1 раз, иначе должна вернуть false. 
Написать тесты для кода
'''

def has_unique_letters(line: str) -> bool:
    if not isinstance(line, str): raise ValueError("На вход подается строка")
    return len(set(line)) == len(line)

def test_unique_letters():
    assert has_unique_letters("abcde") == True
    assert has_unique_letters("aabbccddee") == False
    try:
        has_unique_letters(1234)
        assert False, "Должно было выбросить ValueError"
    except ValueError:
        pass
    print("Все тесты задачи 2 пройдены")

test_unique_letters()

'''
Задача 3
Написать функцию, которая принимает положительное число и возвращает количество бит равных 1 в этом числе.
Написать тесты для кода
'''
def count_ones(number: int) -> int:
    if not isinstance(number, int) or number <= 0:
        raise ValueError("На вход должно подаваться положительное число")
    return bin(number).count("1")

def test_count_ones():
    assert count_ones(5) == 2
    try:
        count_ones(0)
        assert False, "Должно было выбросить ValueError"
    except ValueError:
        pass
    try:
        count_ones(-128)
        assert False, "Должно было выбросить ValueError"
    except ValueError:
        pass
    try:
        count_ones("1234")
        assert False, "Должно было выбросить ValueError"
    except ValueError:
        pass
    print("Все тесты задачи 3 пройдены")

test_count_ones()

'''
Задача 4
Написать функцию, которая принимает положительное число. 
Функция должна вернуть то, сколько раз необходимо перемножать цифры числа или результат перемножения, 
чтобы получилось число состоящее из одной цифры. 
Например, для входного числа: · 39 функция должна вернуть 3, так как 39=27 => 27=14 => 14=4 · 4 функция должна вернуть 0, 
так как число уже состоит из одной цифры · 999 функция должна вернуть 4, так как 999=729 => 729=126 => 126=12 => 12=2. 
Написать тесты для кода
'''

def count_multiplications(number : int) -> int:
    '''
    Возможно можно было сделать легче, но я решил так. В начале проверяем корректность введенных данных.
    Для подсчета необходимых итераций вводим вспом.счетчики.
    Чтобы работать с числом в цикле преобразовываем его в строку (можно было любую коллекцию)
    Далее используем вложенные циклы. Обработка идет до тех пор, пока результатом не станет единичное число (len == 1)
    temp используется для сохранения промежуточных значений перемножений и изменения тела цикла while
    '''
    if not isinstance(number, int) or number <= 0: raise ValueError("На вход должно подаваться положительное число")
    number_str = str(number)
    result_count = 0
    while len(number_str) != 1:
        temp = 1
        for c in number_str:
            temp *= int(c)
        number_str = str(temp)
        result_count += 1
    return result_count

def test_count_multiplications():
    assert count_multiplications(39) == 3
    assert count_multiplications(4) == 0
    assert count_multiplications(999) == 4
    assert count_multiplications(10) == 1
    assert count_multiplications(25) == 2
    try:
        count_multiplications(-5)
        assert False
    except ValueError:
        pass
    
    try:
        count_multiplications("123")
        assert False
    except ValueError:
        pass
    print("Все тесты задачи 4 пройдены")

test_count_multiplications()

'''
Задача 5
Написать функцию, которая принимает два целочисленных вектора одинаковой длины и возвращает среднеквадратическое отклонение двух векторов. 
Написать тесты для кода
'''

def mse(pred: list, true: list) -> int:
    '''
    Сначала проверяем, одинаковы ли длины векторов (условие)
    На вход поступают списки координат векторов
    Т.к векторы равны, то мы можем пройтись одним циклом и попарно их вычесть
    Далее накопить значение квадратичного отклонения попарных координат - это и есть ответ
    '''
    if len(pred) != len(true):
        raise ValueError("Векторы должны быть одинаковой длины")
    n = len(pred)
    total_squared_error = 0
    for i in range(n):
        error = pred[i] - true[i]
        total_squared_error += error ** 2
    return total_squared_error / n


def test_mse():
    assert mse([1, 2, 3], [1, 2, 3]) == 0
    assert mse([1, 2, 3], [0, 1, 2]) == 1
    assert mse([1, 4, 5], [3, 2, 6]) == 3
    assert mse([5], [8]) == 9
    assert mse([-1, -2], [-3, -4]) == 4
    try:
        mse([1, 2], [1])
        assert False, "Должна быть вызвана ошибка ValueError"
    except ValueError:
        pass
    
    print("Все тесты задачи 5 пройдены")

test_mse()

'''
Задача 6
Написать функцию, принимающая целое положительное число.
Функция должна вернуть строку вида “(n1p1)(n2p2)…(nkpk)” 
представляющая разложение числа на простые множители (если pi == 1, то выводить только ni). 
Например, для числа 86240 функция должна вернуть “(25)(5)(7**2)(11)”. 
Написать тесты для кода
'''

def prime_factorization(n):
    '''
    Для начала делаем привычную валидацию входных данных.
    Обрабатываем спец.случай с "1", оно не является ни простым, ни составным
    '''
    if not isinstance(n, int): raise ValueError("На вход подаётся только целое число")
    if n <= 0: raise ValueError("На вход подано отрицательное число, либо нейтральное")
    if n == 1:
        return ""
    result = []
    factor = 2
    while factor * factor <= n:
        count = 0
        while n % factor == 0:
            count += 1
            n //= factor
        if count > 0:
            if count == 1:
                result.append(f"({factor})")
            else:
                result.append(f"({factor}**{count})")
        factor += 1
    if n > 1:
        result.append(f"({n})")
    return "".join(result)


def test_prime_factorization():
    try:
        prime_factorization(-3)
        assert False, "Должна быть вызвана ошибка ValueError"
    except ValueError:
        pass
    try:
        prime_factorization("garbage")
        assert False, "Должна быть вызвана ошибка ValueError"
    except ValueError:
        pass
    try:
        prime_factorization(0)
        assert False, "Должна быть вызвана ошибка ValueError"
    except ValueError:
        pass
    assert prime_factorization(86240) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(17) == "(17)"
    assert prime_factorization(16) == "(2**4)"
    assert prime_factorization(30) == "(2)(3)(5)"
    assert prime_factorization(2) == "(2)"
    assert prime_factorization(1000) == "(2**3)(5**3)"
    assert prime_factorization(1) == ""
    print("Все тесты задачи 6 пройдены")

test_prime_factorization()

'''
Задача 7
Написать функцию, принимающая целое число n, задающее количество кубиков.
Функция должна определить, можно ли из данного кол-ва кубиков построить пирамиду, 
то есть можно ли представить число n как 1^2+2^2+3^2+…+k^2. 
Если можно, то функция должна вернуть k, иначе строку “It is impossible”. 
Написать тесты для кода
'''

def pyramid(number):
    if not isinstance(number, int): raise ValueError("На вход должно подаваться целое число")
    if number < 1:
        return "It is impossible"
    k = 1
    while True:
        total = k * (k + 1) * (2 * k + 1) // 6
        if total == number:
            return k
        elif total > number:
            return "It is impossible"
        k += 1

def test_pyramid():
    try:
        pyramid(1.52323)
        assert False, "Должна быть вызвана ошибка ValueError"
    except ValueError:
        pass
    try:
        pyramid("garbage")
        assert False, "Должна быть вызвана ошибка ValueError"
    except ValueError:
        pass
    assert pyramid(14) == 3
    assert pyramid(55) == 5
    assert pyramid(15) == "It is impossible"
    assert pyramid(1) == 1
    assert pyramid(0) == "It is impossible"
    assert pyramid(-5) == "It is impossible"
    assert pyramid(506) == 11
    assert pyramid(100) == "It is impossible"
    print("Все тесты задачи 7 пройдены")

test_pyramid()

'''
Задача 8
Функция принимает на вход положительное число и определяет является ли оно сбалансированным, 
т.е. сумма цифр до средних равна сумме цифр после. 
Средними в случае нечетного числа цифр считать одну цифру, в случае четного - две средних. 
Написать тесты для кода
'''

'''
Примеры :
5, 53, 157, 173, 211, 257, 263, 373, 563, 593, 607, 653, 733, 947, 977, 1103
Pn = (Pn-1 + Pn+1)/ 2
'''

def is_balanced_number(number : int) -> bool:
    '''
    Делаем валидацию при помощи isinstance и проверки числа
    Если разрядность входного числа меньше трех, то это в любом случае сбалансированное число, т.к нет чисел до/после средних
    В случае нечетного числа - центр середина (length // 2 индекс)
    Иначе под центр (область, не входящую в l_part/r_part) пойдет length // 2, length // 2 + 1
    Сравниваем результаты левой и правой части, возвращаем ответ
    '''
    if not(isinstance(number, int)) or number <= 0: raise ValueError("На вход подается положительное число")
    number_str = str(number)
    length = len(number_str) 
    if length <= 2:
        return True
    if length % 2 == 1:
        left_part = number_str[:length // 2] 
        right_part = number_str[length // 2 + 1:]
    else:
        left_part = number_str[:length // 2 - 1]
        right_part = number_str[length // 2 + 1:]
    left_sum = sum(int(i) for i in left_part)
    right_sum = sum(int(i) for i in right_part)
    return left_sum == right_sum

def test_is_balanced_number():
    try:
        is_balanced_number(-12.5)
        assert False
    except ValueError:
        pass
    try:
        is_balanced_number("123")
        assert False
    except ValueError:
        pass
    assert is_balanced_number(1) == True
    assert is_balanced_number(99) == True
    assert is_balanced_number(121) == True
    assert is_balanced_number(212) == True
    assert is_balanced_number(7547) == True
    assert is_balanced_number(9229) == True
    assert is_balanced_number(12321) == True
    assert is_balanced_number(123321) == True
    assert is_balanced_number(1234321) == True
    print("Все тесты задачи 8 пройдены")

test_is_balanced_number()