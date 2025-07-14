
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
        num1, num2 = map(int, input('두 개의 정수를 공백으로 구분하여 입력: ').split())

        operator = input("원하는 연산자를 입력하세요 (+, -, *, /): ")

        if operator == '+': 
            print(f'Result: {add(num1, num2)}')
        elif operator == '-':
            print(f'Result: {subtract(num1, num2)}')
        elif operator == '*':
            print(f'Result: {multiply(num1, num2)}')
        elif operator == '/':
            print(f'Result: {divide(num1, num2)}')
        else:
            print("Invalid operator.")

    except ValueError:
        print("오류: 정확한 두 개의 '정수'를 공백으로 구분하여 입력해야 합니다.")

    # 0으로 나누려고 시도했을 때
    except ZeroDivisionError:
        print("Error: Division by zero.")
        
if __name__ == "__main__": 
    main()