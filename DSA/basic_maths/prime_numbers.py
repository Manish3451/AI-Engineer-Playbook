"""Prime numbers play a fundamental role in computer science, cryptography, and many real-world applications like secure messaging and data protection. In this problem, you’ll design a beginner-friendly function to find all unique prime factors of a given integer.

You are given a positive integer n. Your task is to implement a function that returns a list containing all distinct prime numbers that divide n (i.e., the prime factors of n in ascending order).

For example, for n = 12, the prime factors are 2 and 3, because 12 = 2 × 2 × 3.

Understanding how to efficiently test for primes and factorize numbers is a foundational skill in DSA. You will apply concepts like prime testing and leverage algorithms such as the Sieve of Eratosthenes to generate a list of primes up to the square root of n and use it for factorization.

This exercise is practical: prime factorization is used in data encryption, error correction, and many interview questions. By breaking a number into its prime factors, you’re learning efficient algorithms and building your understanding of how numbers work under the hood.

Input:

A single positive integer n (1 ≤ n ≤ 10^6)
Output:

An array/list of all unique prime factors of n, sorted in ascending order.
For example, if n = 30, the output should be [2, 3, 5].

If n is 1 (which has no prime factors), output an empty list []."""

## brute force 

# ans  = []

# def is_prime(n):
#     for i in range(2,int(n**0.5)+1):
#         if n < 2:
#             return False
#         if n % i == 0:
#             return False
#     return True



# def prime_factors(n):
#     for i in range(2,n):
#         if n % i == 0:
#             if  is_prime(i):
#                 ans.append(i)
#     return (ans) 


# print(prime_factors(12))

def prime_factors(n):
    factors = set()
    divisor = 2
    
    while n > 1:
        while n % divisor == 0:
            factors.add(divisor)
            n //= divisor
        divisor += 1
    
    return list(factors)

print(prime_factors(12))  # Output: [2, 3]
print(prime_factors(30))  # Output: [2, 3, 5]
print(prime_factors(36))  # Output: [2, 3]