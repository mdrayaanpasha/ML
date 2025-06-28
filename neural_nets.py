import numpy as np

# activation functions
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# loss function
def mean_squared_error(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mean_squared_error_derivative(y_pred, y_true):
    return (y_pred - y_true)

np.random.seed(42)

# Dummy input and output
X = np.array([[1.0], [2.0]]) 
Y = np.array([[2.0], [4.0]])  


learning_rate = 0.01
epochs = 1000

W1 = np.random.randn(1, 2) 
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)  
b2 = np.zeros((1, 1))

for epoch in range(epochs):

    z1 = np.dot(X, W1) + b1       
    a1 = relu(z1)                 

    z2 = np.dot(a1, W2) + b2      
    y_pred = z2                   

    
    loss = mean_squared_error(y_pred, Y)

    dL_dy_pred = mean_squared_error_derivative(y_pred, Y)  

    dL_dW2 = np.dot(a1.T, dL_dy_pred)                   
    dL_db2 = np.sum(dL_dy_pred, axis=0, keepdims=True)  

    dL_da1 = np.dot(dL_dy_pred, W2.T)                   
    dL_dz1 = dL_da1 * relu_derivative(z1)               
    dL_dW1 = np.dot(X.T, dL_dz1)                        
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)      

    
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2

    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("\ntrained W and B:")

print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)
