# Binary Addition Neural Network
#### [Video Demo](https://www.youtube.com/watch?v=orgLG-LPoy4&t=9s)

This repository contains a simple neural network implementation designed to perform binary addition. The neural network takes two n-bit binary numbers as input, adds them, and outputs the binary sum. We are using 8 bit for the case of simplicity and functionality (not looked at higher numbers).

So what is a neural network? A neural network is a series of algorithms intended to recognise patterns designed to mimic how the brain works. Just like children learn to identify numbers - they don't get it straight away but through iteration and correction they begin to learn. As is the same with a neural network - it takes inputs, makes it's guess and then told when it's correct or incorrect. 

<div align=center>
  
  ![image](https://github.com/user-attachments/assets/edc04160-600c-4222-9137-d67c30ea74d2)
  
  </div>

When a neural network processes information, it does so in a step-by-step manner (forward propagation). Here's how it works:
- The input Layer: This is where the data enters the network. In this case, the two binary numbers.
- The Hidden Layer: The processing layer if you will, responsible for making weighted calculations, helping the NN understand the relationship between the numbers.
  - ReLU is a function that helps a neural network decide which neurons to activate. It works like a filter that only lets through positive values and blocks out any negative ones. In simple terms, ReLU helps the network focus on useful information and ignore the rest.
  - The Sigmoid function takes any number and squashes it into a value between 0 and 1. Think of it as a way to convert raw data into something more manageable, like a probability. It’s like a switch that helps the network decide whether a neuron should be on (close to 1) or off (close to 0).
- The Output Layer: Finally, the networks spits out a prediction or decision based on what it has inferred.

Once the network has made its prediction, it checks to see if it got it right. If it’s wrong, the network learns from its mistake using backward propagation. Here's how it works:
- The network calculates how far off its prediction was from the correct answer (called the error - seen in the results further down the page).
- It then adjusts the weights (the numbers it uses to process the data) to make a better guess next time. This process is repeated multiple times, making the network more accurate, essentially learning.

In short, forward propagation is about making a prediction, and backward propagation is how the network learns and improves. This cycle allows neural networks to solve complex problems like recognizing images, understanding speech, etc...

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
- **Output layer**: 9 neurons (sum of two 8-bit numbers, with one extra bit for carry).

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

