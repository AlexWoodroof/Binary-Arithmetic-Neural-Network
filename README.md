# Binary Addition Neural Network
#### [Video Demo](https://www.youtube.com/watch?v=orgLG-LPoy4&t=9s)

This repository contains a simple neural network implementation designed to perform binary addition. The neural network takes two n-bit binary numbers as input, adds them, and outputs the binary sum. We are using 8 bit for the case of simplicity and functionality (not looked at higher numbers).

## Code Overview

### Functions

- **`relu(x)`**: The Rectified Linear Unit (ReLU) activation function.
  - Returns the element-wise maximum of `0` and `x`.

- **`relu_derivative(x)`**: The derivative of the ReLU function, used during backpropagation.
  - Returns `1` for positive values, `0` otherwise.

- **`sigmoid(x)`**: The sigmoid activation function.
  - Returns a value between `0` and `1` for each element of `x`.

- **`sigmoid_derivative(x)`**: The derivative of the sigmoid function for backpropagation.
  - Returns `x * (1 - x)` where `x` is the sigmoid output.

- **`test_model()`**: Tests the network on randomly generated binary addition problems.
  - Prints the predicted binary sum compared to the actual binary sum, along with whether the prediction was correct.

### Neural Network Architecture/Structure

- **Input layer**: 16 neurons (two 8-bit numbers concatenated).
- **Hidden layer**: 64 neurons, with ReLU activation.
- **Output layer**: 9 neurons (sum of two 8-bit numbers, with one extra bit for carry), using sigmoid activation.

### Hyperparameters

- **`n_bits`**: Number of bits in the binary numbers (8).
- **`input_size`**: 16 (two 8-bit numbers).
- **`hidden_size`**: 64 neurons in the hidden layer.
- **`output_size`**: 9 (to account for carry bit).
- **`epochs`**: Number of iterations for training (50,000).
- **`learning_rate`**: 0.01.

### Training Output

During training, the error between the predicted and actual binary sums is printed to show the learning progress. As the network trains, the error decreases, showing the network is improving at predicting.

Example output:
```bash
Epoch 0, Error: 0.4852
Epoch 1000, Error: 0.2053
Epoch 2000, Error: 0.1147
...
Epoch 49000, Error: 0.0008
```

### Testing

After training, the `test_model()` function generates random test cases, feeds them through the NN, and compares the predicted sum to the actual sum. It also prints whether the prediction was correct.

Example test output:
```bash
Test case: 57 + 109
Predicted: 000110010, Actual: 000110010, Correct
Test case: 3 + 130
Predicted: 100001101, Actual: 100001101, Correct
```

### Running the Code

1. Clone the repository.

   ```bash
   git clone https://github.com/AlexWoodroof/Binary-Arithmetic-Neural-Network.git
   cd Binary-Arithmetic-Neural-Network
   ```
   
3. Ensure you have `numpy` installed (`pip install numpy`).
4. Run the script to train the neural network and test it with random cases:

    ```bash
    python BinaryArithmeticNN.py
    ```

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

