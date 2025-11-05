import heapq
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = []

        for x in nums:
            if len(heap) < k:
                heapq.heappush(heap,x)
            else:
                if heap[0] < x:
                    heapq.heappop(heap)   
                    heapq.heappush(heap,x) 
        return heap[0]
        