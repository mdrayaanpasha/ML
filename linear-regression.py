"""========================================================================================================================================================================================

                                                    🫠 Mohammed Rayaan Pasha  | 📅 30-10-2024  | ☀️ Wednesday.

                                👋 Yello! This is a fun project designed to find the unique combinations of habitability in exo-planets.


                                # METHADOLOGY:

                                  + I've chosen "Gradient Descent" as the regression model, which consists of the following functions:

========================================================================================================================================================================================

                              # Program MAP:

                                                            -----------------------------------
                                                            | Division          | Search Code |
                                                            | ----------------  |-------------|
                                                            |                   |             |
                                                            |  Library Imports  |     #0      |
                                                            |                   |             |
                                                            |-------------------|-------------|
                                                            |                   |             |
                                                            |  Function         |             |
                                                            |  Declartion       |     #1      |
                                                            |                   |             |
                                                            |-------------------|-------------|
                                                            |                   |             |
                                                            |   Execuion        |             |
                                                            |                   |     #2      |
                                                            |-------------------|-------------|

========================================================================================================================================================================================

                              # FUNCTION MAP:

                                                            -----------------------------------
                                                            | Function Name     | Search Code |
                                                            | ----------------  |-------------|
                                                            |                   |             |
                                                            |  h(X, Theta)      |     #11     |
                                                            |                   |             |
                                                            |-------------------|-------------|
                                                            |                   |             |
                                                            |  J(X, Y, Theta)   |     #12     |
                                                            |                   |             |
                                                            |-------------------|-------------|
                                                            |                   |             |
                                                            |  Gradient(X, Y,   |             |
                                                            |   Theta)          |     #13     |
                                                            |                   |             |
                                                            |-------------------|-------------|
                                                            |                   |             |
                                                            |  gradient_descent |             |
                                                            |  (X, Y, Theta,    |             |
                                                            |  Learn_Rate,      |    #14      |
                                                            |  max_iterations)  |             |
                                                            |                   |             |
                                                            -----------------------------------

========================================================================================================================================================================================

                              # Usage:

                                + Data set has following features: { Distance, Oxygen, Water, Radiation } to know habitability scores.
                                + Call the gradient descent function to optimize the co-efs.
                                + Generate a range of feature combinations and predict their habitability scores.
                                + Filter results based on the predicted score to find viable exo-planets.

========================================================================================================================================================================================

                            #NOTE: NO LIBS USED!! EXCEPT FOR TABLES.

========================================================================================================================================================================================"""

                                                                        #0 => LIBRARIES

from tabulate import tabulate
# this was just for tables!! 😳

#========================================================================================================================================================================================"""

                                                                        #1 => FUNCTION DECLARATION


#11 => Hypotesis Function.
"""

  This function takes co-ef and var and multiplies and returns the value!
  eg: m0+m1*x1+.....mn*xn

"""

def h(X, Theta):
    predictions = []
    for i in range(len(X)):
        prediction = 0
        for j in range(len(Theta)):
            prediction += X[i][j] * Theta[j]
        predictions.append(prediction)
    return predictions


#11 => Cost Function.
"""

  This function takes Takes Prediction & Actual Value of data set and return Mean Squared Error.
  formula: 1/ (2*m) * sum from 0-m of (Y^(i)-h^(i))^2

"""

def J(X, Y, Theta):
    prediction = h(X, Theta)
    Error = 0
    for i in range(len(Y)):
        Error += (Y[i] - prediction[i]) ** 2
    MSE = Error / (len(Y))
    return MSE


#12 => Derivite Function / Gradient Function.
"""

  - This function takes Takes Prediction, Actual Value & Co-eff (Thetha) of data set returns d/dθ of J(θ).
  - The idea is to reduce this derivitive from theta to reduce errors.
  - formula: d/dθ of J(θ) = - (2 / M ) * sum of err[i] - X[i][j] for all i,j in Y.

"""


def Gradient(X, Y, Theta):
    prediction = h(X, Theta)
    errors = [Y[i] - prediction[i] for i in range(len(Y))]
    gradient = [0] * len(Theta)
    for j in range(len(Theta)):
        total_gradient = 0
        for i in range(len(Y)):
            total_gradient += errors[i] * X[i][j]
        gradient[j] = -(2 / len(Y)) * total_gradient
    return gradient

#12 => Gradient Descent Function.
"""

  - This function takes  we Initialize min_cost to infinity and track best_theta.
  - we iterate and in each iteration:
    + calculate current MSE.
    + update min_cost and best_theta if the new cost is lower.
    + compute the gradient
    + update Theta using the learning rate and gradient.

"""
def gradient_descent(X, Y, Theta, Learn_Rate=0.01, max_iterations=100):
    min_cost = float('inf')
    best_theta = Theta.copy()
    cost_history = []

    for iteration in range(max_iterations):
        mse_value = J(X, Y, Theta)
        cost_history.append(mse_value)

        if mse_value < min_cost:
            min_cost = mse_value
            best_theta = Theta.copy()

        gradient = Gradient(X, Y, Theta)

        for j in range(len(Theta)):
            Theta[j] -= Learn_Rate * gradient[j]

    return best_theta

#========================================================================================================================================================================================"""

                                                                        #1 => Execution


# Ex dataset.

X = [
    [0.5, 20, 60, 1.2],
    [1.0, 18, 70, 1.5],
    [1.5, 15, 50, 1.8],
    [2.0, 12, 40, 2.1],
    [1.2, 20, 65, 1.4]
]

#eg outputs.
Y = [50, 55, 45, 30, 52]

# initial co-efs.
Theta = [0.5, 0.5, 0.5, 0.5]
Learn_Rate = 0.001




# run gradient descent
optimized_theta = gradient_descent(X, Y, Theta, Learn_Rate)



# ================== making up all combinations of possible features. ===========================

# Distance in AU: 0.5, 1.0, 1.5, 2.0
distance_range = [i * 0.5 for i in range(1, 5)]

# Oxygen level: 10, 15, 20, 25, 30 (%)
oxygen_range = [i for i in range(10, 31, 5)]

# Water content: 0, 10, ..., 100 (%)
water_range = [i for i in range(0, 101, 10)]

# Radiation: 1.0, 1.5, 2.0, 2.5, 3.0
radiation_range = [1.0 + i * 0.5 for i in range(5)]



results = []

# generating that combinations via loops
for distance in distance_range:
    for oxygen in oxygen_range:
        for water in water_range:
            for radiation in radiation_range:
                combination = [distance, oxygen, water, radiation]
                predicted_score = h([combination], optimized_theta)[0]
                if predicted_score > 60:
                    results.append(combination + [predicted_score])



#printing in a table.

print("\n","="*200, "\n")
print("\n\t\t\t\t\t\t👽✨ Unique Exo-Planet Habitability Predictions ✨👽")
print("\n\t\t\t\t\t\t🚀 After optimizing coefficients using gradient descent:")
print("\n\t\t\t\t\t\t📊 The following feature combinations yield habitability scores above 60:\n")
print("\n","="*200, "\n")

print("\n\t\t\t\t\t\t","-"*24, " QUICK STATS ","-"*24,)
print("\n\t\t\t\t\t\t🪐 Total Combinations Found:", len(results))
print("\n\t\t\t\t\t\t🧪 Optimized Coefficients (Theta):", optimized_theta)
print("\n","="*200, "\n")

headers = ["Distance (AU)", "Oxygen (%)", "Water (%)", "Radiation", "Predicted Score"]
print(tabulate(results, headers=headers, tablefmt="pretty", floatfmt=".2f"))


print("\n\n Good luck finding us humans! 👽")
