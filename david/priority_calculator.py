import calculator as calc


def main():
    try:
        input_str = input().split()

        # empty input check
        if not input_str:
            raise ValueError

        operators = {"+", "-", "*", "/"}

        # print(f"Input Check : {input_str}")

        # check float & operator
        for i, token in enumerate(input_str):
            is_even_index = i % 2 == 0

            if is_even_index:
                try:
                    float(token)
                except ValueError:
                    raise
            else:
                if token not in operators:
                    raise ValueError

        # casting numbers to float without operators
        operator_with_number = [
            float(token) if token not in operators else token for token in input_str
        ]

        i = 0
        while i < len(operator_with_number):
            operator = operator_with_number[i]

            if operator == "*":
                result = calc.multiply(
                    operator_with_number[i - 1], operator_with_number[i + 1]
                )
                operator_with_number[i - 1 : i + 2] = [result]
                i = 0
                continue
            elif operator == "/":
                if operator_with_number[i + 1] == 0:
                    raise ZeroDivisionError
                result = calc.divide(
                    operator_with_number[i - 1], operator_with_number[i + 1]
                )
                operator_with_number[i - 1 : i + 2] = [result]
                i = 0
                continue
            i += 1

        # print(f"Priority Check : {operator_with_number}")

        result = operator_with_number[0]
        for i in range(1, len(operator_with_number), 2):
            operator = operator_with_number[i]
            number = operator_with_number[i + 1]

            if operator == "+":
                result = calc.add(result, number)
            elif operator == "-":
                result = calc.subtract(result, number)

        print(f"Result: {result}")

    except (ValueError, IndexError):
        print("Invalid input.")

    except ZeroDivisionError:
        print("Error: Division by zero.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
