def main():

    try:
        base = float(input('Enter number: '))
    except ValueError:
        print('Invalid number input')
        return
    
    try:
        exponent = int(input('Enter exponent: '))
    except:
        print('Invalid expponent input')
        return

    result = 1.0

    if exponent >= 0:
        for _ in range(exponent):
            result *= base
    else:
        for _ in range(abs(exponent)):
            result *= base
        if result == 0:
            print('Cannot divide by zero')
            return
        result = 1.0 / result

    if result == int(result):
        print(f'Result: {int(result)}')
    else:
        print(f'Result: {result}')



if __name__ == "__main__":
    main()