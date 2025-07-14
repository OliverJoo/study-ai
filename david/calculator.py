
def add(a, b):
    return a + b

def subtract(a, b):
    return a-b

def multiply(a, b):
    return a * b

def divide(a, b):
    return a/b
        

def main():
    try:
        num1, num2 = map(float, input('실수형 숫자 2개 입력를 공백으로 구분하여 입력: ').split())
        
        num1 = int(num1)
        num2 = int(num2)

        operator = input("연산자 입력 (+, -, *, /): ")
 
        result = None
        if operator == '+':
            result = add(num1, num2)
        elif operator == '-':
            result = subtract(num1, num2)
        elif operator == '*':
            result = multiply(num1, num2)
        elif operator == '/':
            result = divide(num1, num2)

        if result is not None:
            msg = f'Result: {result}'
        else:
            msg = "Invalid operator."

        print(msg)

    except ValueError:
        print("오류: 정확한 두 개의 '정수'를 공백으로 구분하여 입력해야 합니다.")

    # 0으로 나누려고 시도했을 때 에러 처리
    except ZeroDivisionError:
        print("Error: Division by zero.")
        
if __name__ == "__main__": 
    main()