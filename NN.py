import numpy as np

# ReLU function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Convert integers to binary arrays for input
def int_to_binary_array(n, length):
    return np.array(list(np.binary_repr(n, length))).astype(np.int8)

# Create training data for binary addition
n_bits = 8
max_number = 2 ** n_bits

input_data = []
target_output = []

for _ in range(1000):  # Generate 1000 training examples
    num1 = np.random.randint(0, max_number)
    num2 = np.random.randint(0, max_number)
    
    # Convert numbers to binary arrays
    binary_num1 = int_to_binary_array(num1, n_bits)
    binary_num2 = int_to_binary_array(num2, n_bits)
    
    # Binary addition
    sum_result = np.binary_repr(num1 + num2, n_bits + 1)  # +1 for possible carry
    binary_sum = np.array(list(sum_result)).astype(np.int8)
    
    # Append to training data
    input_data.append(np.concatenate((binary_num1, binary_num2)))  # Inputs: two binary numbers
    target_output.append(binary_sum)  # Output: binary sum

input_data = np.array(input_data)
target_output = np.array(target_output)

# Initialize network parameters
input_size = n_bits * 2
hidden_size = 64  # Increased the number of neurons in the hidden layer
output_size = n_bits + 1  # Output has an extra bit for possible carry

weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1

# Forward propagation using ReLU for hidden layer
def forward_propagation(input_data):
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = relu(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

# Backpropagation using ReLU
def backpropagation(input_data, target, learning_rate):
    global weights_input_hidden, weights_hidden_output

    hidden_layer_output, output_layer_output = forward_propagation(input_data)

    output_error = target - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * relu_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += input_data.T.dot(hidden_delta) * learning_rate

# Training the model
epochs = 50000
learning_rate = 0.01  # Decreased learning rate for more stable training

for epoch in range(epochs):
    backpropagation(input_data, target_output, learning_rate)
    if epoch % 1000 == 0:
        _, predicted_output = forward_propagation(input_data)
        error = np.mean(np.abs(target_output - predicted_output))
        print(f'Epoch {epoch}, Error: {error}')

# Test the model
def test_model():
    num1 = np.random.randint(0, max_number)
    num2 = np.random.randint(0, max_number)

    binary_num1 = int_to_binary_array(num1, n_bits)
    binary_num2 = int_to_binary_array(num2, n_bits)

    test_input = np.concatenate((binary_num1, binary_num2)).reshape(1, -1)
    _, predicted_output = forward_propagation(test_input)
    
    predicted_binary = (predicted_output > 0.5).astype(np.int8)[0]
    actual_sum = np.binary_repr(num1 + num2, n_bits + 1)
    
    if ''.join(map(str, predicted_binary)) == actual_sum:
        output = "Correct"
    else:
        output = "Incorrect"

    print(f"Test case: {num1} + {num2}")
    print(f"Predicted: {''.join(map(str, predicted_binary))}, Actual: {actual_sum}, {output}")

# Run the test
for _ in range(10):  
    test_model()
