# Toy Neural Network Examples
Data transfer between main system memory and GPU memory seems to be the largest bottleneck. After an unfortunate experience wasting too much time on a silly bug, I moved to using non-minified versions of libraries for verbose error messages during development.

*   [MNIST](./mnist/)
    This is the classic "Hello World!" of machine learning. This implementation uses multiple calls for training and prediction. All processing is done through the `draw()` loop. The user experience on slow connections would be greatly improved by some sort of user interface elements to indicate loading.

*   [Prettier MNIST](./prettier-mnist/)
    This is a WIP interpretation of the above. It currently has a similar interface and outward functionality. It provides a lot of useful diagnostic information. Vast speedups are achieved through batch processing of training and testing data. Testing data is processed inside of the `draw()` loop, while training is handled between frames. I doubt this accomplishes a lot, but I like the logical separation.

*   [XOR Visualization](./xor/)
    A simple and elegant mapping of logical XOR to two-dimensional space. The small training data set and intense demand for prediction results makes this especially sensitive to buffer creation and transfer overhead. Batch processing makes this feasible.

*   *TODO*Add more examples.