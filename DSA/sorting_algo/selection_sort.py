def selection_sort(arr):
    n = len(arr)
    for i in range(n-1):
        mini = j
        for j in range(i+1,n):
            if arr[j] < arr[mini]:
                mini = j

        arr[mini], arr[i] = arr[i], arr[mini]