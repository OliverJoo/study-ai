def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ZeroDivisionError
    return a / b

def main():
    while True:
        expression = input("계산할 식 입력 (예: 10 + 5). 종료하려면 'exit' 입력: ")

        if expression.lower() == 'exit':
            print("계산기를 종료합니다.")
            break

        try:
            parts = expression.split()

            if len(parts) != 3:
                raise IndexError

            num1 = float(parts[0])
            operator = parts[1]
            num2 = float(parts[2])

            if operator == '+':
                result = add(num1, num2)
            elif operator == '-':
                result = subtract(num1, num2)
            elif operator == '*':
                result = multiply(num1, num2)
            elif operator == '/':
                result = divide(num1, num2)
            else:
                print(f"오류: '{operator}'는 유효하지 않은 연산자입니다.")
                continue

            print(f'결과: {result}')

        except ValueError:
            print("Error: Invalid number format. Please enter valid numbers.")
        except IndexError:
            print("Error: Invalid format. Please use 'number operator number' (e.g., 2 + 3).")
        except ZeroDivisionError:
            print("Error: Division by zero.")
        except Exception as e:
            print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()