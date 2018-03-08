# Toy Neural Network Examples
Data transfer between main system memory and GPU memory seems to be the largest bottleneck.

*   [MNIST](./mnist/)
    This is the classic "Hello World!" of machine learning. This example uses multiple calls for training and prediction. All processing is done through the `draw()` loop.

*   [Prettier MNIST](./prettier-mnist/)
    This is a WIP interpretation of the above. It currently has a similar interface and outward functionality. It provides a lot of useful diagnostic information. Vast speedups are achieved through batch processing of training and testing data. Testing data is processed inside of the `draw()` loop, while training is handled between frames. I doubt this accomplishes a lot, but I like the logical separation.

*   [XOR Visualization](./xor/)
    A simple and elegant mapping of logical XOR to two-dimensional space. The small training data set and intense demand for prediction results makes this especially sensitive to buffer creation and transfer overhead. Batch processing makes this feasible.