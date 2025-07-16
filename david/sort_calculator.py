def selection_sort(numbers):
    n = len(numbers)

    for i in range(n):
        min_index = i

        for j in range(i + 1, n):
            if numbers[j] < numbers[min_index]:
                min_index = j

        numbers[i], numbers[min_index] = numbers[min_index], numbers[i]

    return numbers


def main():
    try: 
        input_str = input("숫자를 공백으로 구분하여 입력: ").split()

        numbers = [float(num) for num in input_str]

        if not numbers:
            print("Invalid input.")
            return 

        sorted_numbers = selection_sort(numbers)

        output_str = " ".join(map(str, sorted_numbers))
        print(f"Sorted: {output_str}")

    except ValueError:
        print("Invalid input.")


if __name__ == "__main__":
    main()
