import heapq
from typing import List

def merge_k_sorted_arrays(arrs: List[List[int]]) -> List[int]:
    heap = []  # (value, array_idx, elem_idx)
    for a_idx, arr in enumerate(arrs):
        if arr:  # skip empty arrays
            heapq.heappush(heap, (arr[0], a_idx, 0))

    out = []
    while heap:
        val, a_idx, i = heapq.heappop(heap)
        out.append(val)
        nxt_i = i + 1
        if nxt_i < len(arrs[a_idx]):
            heapq.heappush(heap, (arrs[a_idx][nxt_i], a_idx, nxt_i))
    return out

# Example
# print(merge_k_sorted_arrays([[1,4,5],[1,3,4],[2,6]]))  # -> [1,1,2,3,4,4,5,6]
