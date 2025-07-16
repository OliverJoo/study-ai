def find_min_max(numbers): 
    min_val = numbers[0]
    max_val = numbers[0]
 
    for number in numbers[1:]: 
        if number < min_val:
            min_val = number 
        if number > max_val:
            max_val = number
            
    return min_val, max_val

def main(): 
    try: 
        input_str = input("숫자를 공백으로 구분하여 입력: ")
         
        numbers = [float(num) for num in input_str.split()]
         
        if not numbers:
            print("숫자를 입력해야 합니다.")
            return
 
        min_val, max_val = find_min_max(numbers)
         
        print(f"Min: {min_val}, Max: {max_val}")

    except ValueError: 
        print("Invalid input.")
 
if __name__ == "__main__":
    main()