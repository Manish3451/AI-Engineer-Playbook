def reverse_array(arr):
    """
    Reverses the given array in place.

    Parameters:
    arr (list): The array to be reversed.

    Returns:
    list: The reversed array.
    """
    left = 0
    right = len(arr) - 1

    while left < right:
        # Swap the elements at left and right indices
        arr[left], arr[right] = arr[right], arr[left]
        # Move towards the middle
        left += 1
        right -= 1

    return arr