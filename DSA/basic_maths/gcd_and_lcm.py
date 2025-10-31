"""Greatest Common Divisor (GCD) and Least Common Multiple (LCM) are two fundamental concepts in number theory and arithmetic. These concepts are widely used in computer science, from simplifying fractions to solving scheduling problems.

In this problem, you will design two efficient functions: one to compute the GCD of two positive integers using the Euclidean Algorithm, and another to compute their LCM using the GCD result. Given two integers, A and B, your task is to:

Implement a function to find the GCD of A and B.
Implement a function to find the LCM of A and B using the relationship between GCD and LCM.
Your program should read two positive integers and output their GCD and LCM. This exercise will help you solidify your understanding of the Euclidean Algorithm and how GCD and LCM are related. These concepts are often used in technical interviews at top companies, and mastering them is a core step in building strong DSA foundations.

"""

def gcd(a,b):

    while b != 0:
        a, b = b, a%b
    return a 

def lcm(a,b):
    ## lcm = a*b/gcd
    lcm = a*b/gcd(a,b)

    return lcm

print(gcd(12,36))
print(lcm(12,18))



