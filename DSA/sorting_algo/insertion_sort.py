def insertion_sort(arr):
    n = len(arr)
    for i in range(n):         
        j = i
        while j > 0 and arr[j-1] > arr[j]:
            arr[j-1], arr[j] = arr[j], arr[j-1]   
            j -= 1

    print("After insertion sort:")
    print(*arr)


# driver code
arr = [13, 46, 24, 52, 20, 9]
print("Before Using insertion Sort:")
print(*arr)
insertion_sort(arr)
