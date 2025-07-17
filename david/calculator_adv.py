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
    try:
        expression = input("Enter expression: ")

        # 제한없이 받아들이고자 할 경우
        # numbers = list(map(float, input('원하는 만큼 실수를 공백으로 구분하여 입력: ').split()))

        parts = expression.split()

        if len(parts) != 3:
            print(
                "Error: Invalid format. Please use 'number operator number' (e.g., 2 + 3)."
            )
            return

        num1 = float(parts[0])
        operator = parts[1]
        num2 = float(parts[2])

        if operator == "+":
            result = add(num1, num2)
        elif operator == "-":
            result = subtract(num1, num2)
        elif operator == "*":
            result = multiply(num1, num2)
        elif operator == "/":
            result = divide(num1, num2)
        else:
            print(f"Error: Invalid operator '{operator}'")
            return

        print(f"Result: {result}")

    except ValueError:
        print("Error: Invalid number format.")
    except IndexError:
        print(
            "Error: Invalid format. Please use 'number operator number' (e.g., 2 + 3)."
        )
    except ZeroDivisionError:
        print("Error: Division by zero.")


if __name__ == "__main__":
    main()
