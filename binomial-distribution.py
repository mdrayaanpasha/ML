import random
import math

n=int(input("how many visites did u get? "))
clicks = random.choice(range(0,n))


pr=clicks/n

#now that we know what the thing is? lets create this binomial dist
expectedV = int(input("How many visits do you expect? "))
expectedC = int(input("How many Clicks do you wanna compute on? "))


binomialCoeff = math.factorial(expectedV) / (math.factorial(expectedC) * math.factorial(expectedV - expectedC))

probability = binomialCoeff * (pr ** expectedC) * ((1 - pr) ** (expectedV - expectedC))

print(probability)

