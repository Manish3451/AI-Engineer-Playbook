import heapq

def top_k_frequent_heap(nums, k):
    # 1) Count frequencies using a simple dict
    freq = {}
    for x in nums:
        if x in freq:
            freq[x] += 1
        else:
            freq[x] = 1

    # 2) Use a min-heap of size k to keep the k most frequent items.
    # Heap elements are tuples: (count, number)
    min_heap = []

    for number, count in freq.items():
        # If heap has less than k elements, just push the current pair
        if len(min_heap) < k:
            heapq.heappush(min_heap, (count, number))
        else:
            # If current count is larger than the smallest count in heap,
            # replace the smallest with the current pair.
            if count > min_heap[0][0]:
                heapq.heapreplace(min_heap, (count, number))

    # 3) min_heap now contains k pairs (count, number).
    # Sort them by count descending so result is ordered from most -> least frequent
    min_heap.sort(key=lambda pair: pair[0], reverse=True)

    # 4) Extract the numbers in that order and return
    result = [number for count, number in min_heap]
    return result


# Examples / testing
print(top_k_frequent_heap([1,1,1,2,2,3], 2))   # -> [1, 2]
print(top_k_frequent_heap([1], 1))             # -> [1]
print(top_k_frequent_heap([1,2,1,2,1,2,3,1,3,2], 2))  # -> [1, 2]
