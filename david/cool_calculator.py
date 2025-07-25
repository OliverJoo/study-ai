import calculator as calc
import math 

def main():
    try:
        input_tokens = input("Enter a calculation expression (e.g., 10 + 20 * sin 30): ").split()

        if not input_tokens:
            raise ValueError("Input cannot be empty.")

        operators = {"+", "-", "*", "/", "sin", "cos", "tan"} 

        # Process unary operators (sin, cos, tan) first
        processed_tokens = []
        i = 0
        while i < len(input_tokens):
            token = input_tokens[i]
            if token in ["sin", "cos", "tan"]:
                if i + 1 >= len(input_tokens): 
                    raise ValueError(f"Missing number after '{token}' function.")
                
                try:
                    angle = float(input_tokens[i+1])
                except ValueError:
                    raise ValueError(f"The argument for '{token}' must be a number, but got '{input_tokens[i+1]}'.")

                if token == "sin":
                    processed_tokens.append(calc.sin(angle))
                elif token == "cos":
                    processed_tokens.append(calc.cos(angle))
                elif token == "tan":
                    processed_tokens.append(calc.tan(angle))
                i += 2 
            else:
                processed_tokens.append(token)
                i += 1
        
        input_tokens = processed_tokens 

        # Validate remaining tokens (numbers and binary operators)
        for i, token in enumerate(input_tokens):
            is_even_index = i % 2 == 0 

            if is_even_index:
                try:
                    float(token)
                except ValueError:
                    raise ValueError(f"'{token}' is not a valid number at position {i}.")
            else: 
                if token not in operators:
                    raise ValueError(f"'{token}' is not a valid operator at position {i}.")
        
        # Cast numbers to float
        operator_with_number = []
        for token in input_tokens:
            if token in operators:
                operator_with_number.append(token)
            else:
                operator_with_number.append(float(token))

        # Handle multiplication and division (operator precedence)
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
                if isinstance(operator_with_number[i + 1], (int, float)) and operator_with_number[i + 1] == 0:
                    raise ZeroDivisionError("Division by zero is not allowed.")
                
                result = calc.divide(
                    operator_with_number[i - 1], operator_with_number[i + 1]
                )
                operator_with_number[i - 1 : i + 2] = [result]
                i = 0 
                continue
            i += 1 

        # Handle addition and subtraction
        if len(operator_with_number) == 1:
            final_result = operator_with_number[0]
        else:
            final_result = operator_with_number[0]
            for i in range(1, len(operator_with_number), 2): 
                operator = operator_with_number[i]
                number = operator_with_number[i + 1]

                if operator == "+":
                    final_result = calc.add(final_result, number)
                elif operator == "-":
                    final_result = calc.subtract(final_result, number)

        print(f"Result: {final_result}")

    except (ValueError, IndexError) as e:
        print(f"Invalid input: {e}")
    except ZeroDivisionError:
        print("Error: Division by zero is not allowed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()