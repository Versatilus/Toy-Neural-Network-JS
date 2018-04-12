// Other techniques for learning

// ESLint comment code to quiet my linter. It also provides a form of documentation.
/* global dl */
/* exported ActivationFunction, tanh, sigmoid, NeuralNetwork */

// This is dead code until deeplearn.js functions are investigated fully.
class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

let tanh = new ActivationFunction(x => Math.tanh(x), y => 1 - y * y);

class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    let ihVariance = Math.sqrt((input_nodes + hidden_nodes) * 0.5);
    let hoVariance = Math.sqrt((output_nodes + hidden_nodes) * 0.5);
    tf.tidy(() => {
      this.weights_ih = tf.variable(
        tf.randomNormal(
          [this.hidden_nodes, this.input_nodes],
          0,
          1 / ihVariance
        ),
        true,
        'weights1'
      );
      this.weights_ho = tf.variable(
        tf.randomNormal(
          [this.output_nodes, this.hidden_nodes],
          0,
          1 / hoVariance
        ),
        true,
        'weights2'
      );
      this.bias_h = tf.variable(
        tf.randomNormal(
          [this.hidden_nodes, 1],
          0,
          Math.sqrt(1 / this.hidden_nodes)
        ),
        true,
        'bias1'
      );
      this.bias_o = tf.variable(
        tf.randomNormal(
          [this.output_nodes, 1],
          0,
          Math.sqrt(1 / this.output_nodes)
        ),
        true,
        'bias2'
      );
      this.hidden_activations = tf.variable(
        tf.zeros([this.hidden_nodes, 1]),
        true,
        'activation1'
      );
      this.output_activations = tf.variable(
        tf.zeros([this.output_nodes, 1]),
        true,
        'activation2'
      );
    });
    this.setLearningRate();

    // this.setActivationFunction(); // currently a no-op
  }

  predict(input_array) {
    /* TODO: It should be fairly straightforward to refactor this to handle both individual and
       batch processing by reorienting the matrices and letting broadcasting work it's miracles. */

    return tf.tidy(() => {
      // Generating the Hidden Outputs
      let inputs;
      if (input_array instanceof tf.Tensor) inputs = input_array;
      else inputs = tf.tensor2d(input_array, [input_array.length, 1]);

      this.hidden_activations.assign(tf.matMul(this.weights_ih, inputs));

      this.hidden_activations.assign(
        tf.add(this.hidden_activations, this.bias_h)
      );

      // activation function!
      this.hidden_activations.assign(tf.sigmoid(this.hidden_activations));

      // Generating the output's output!
      this.output_activations.assign(
        tf.matMul(this.weights_ho, this.hidden_activations)
      );

      this.output_activations.assign(
        tf.add(this.output_activations, this.bias_o)
      );
      this.output_activations.assign(tf.sigmoid(this.output_activations));

      // Sending back to the caller!
      return this.output_activations.dataSync();
    });
  }

  setLearningRate(learning_rate = 0.1) {
    if (this.learning_rate instanceof tf.Tensor) this.learning_rate.dispose();

    this.learning_rate = tf.scalar(learning_rate);
  }

  setActivationFunction(func = sigmoid) {
    // currently a no-op
    this.activation_function = func;
  }

  train(input_array, target_array) {
    // Generating the Hidden Outputs
    tf.tidy(() => {
      let inputs;
      if (input_array instanceof tf.Tensor) inputs = input_array;
      else inputs = tf.tensor2d(input_array, [input_array.length, 1]);
      this.predict(input_array);

      let targets = tf.tensor2d(target_array, [target_array.length, 1]);

      // Calculate the error
      // ERROR = TARGETS - OUTPUTS
      let output_errors = tf.sub(targets, this.output_activations);

      // Calculate gradient
      // TODO: use deeplearn.js gradient functions
      let gradients = tf.mul(
        this.output_activations,
        tf.sub(tf.scalar(1), this.output_activations)
      );
      gradients = tf.mul(gradients, output_errors).mul(this.learning_rate);

      // Calculate deltas
      let hidden_T = tf.transpose(this.hidden_activations);
      let weight_ho_deltas = tf.matMul(gradients, hidden_T);

      // Calculate the hidden layer errors
      let who_t = tf.transpose(this.weights_ho);
      let hidden_errors = tf.matMul(who_t, output_errors);

      // Calculate hidden gradient
      let hidden_gradient = tf.mul(
        this.hidden_activations,
        tf.sub(tf.scalar(1), this.hidden_activations)
      );
      hidden_gradient = tf
        .mul(hidden_gradient, hidden_errors)
        .mul(this.learning_rate);

      // Calcuate input->hidden deltas
      let inputs_T = tf.transpose(inputs);
      let weight_ih_deltas = tf.matMul(hidden_gradient, inputs_T);

      // Adjust the weights by deltas
      this.weights_ho.assign(tf.add(this.weights_ho, weight_ho_deltas));
      this.weights_ih.assign(tf.add(this.weights_ih, weight_ih_deltas));

      // Adjust the bias by its deltas (which is just the gradients)
      this.bias_o.assign(tf.add(this.bias_o, gradients));
      this.bias_h.assign(tf.add(this.bias_h, hidden_gradient));
    });
  }

  predictBatch(input_arrays) {
    // The trick to batch processing is correct orientation of the matrices.
    return tf.tidy(() => {
      let inputs;
      if (input_arrays instanceof tf.Tensor) inputs = input_arrays;
      else inputs = tf.tensor2d(input_arrays);
      let hidden = tf.matMul(inputs, tf.transpose(this.weights_ih));
      hidden = tf.add(hidden, tf.transpose(this.bias_h));
      hidden = tf.sigmoid(hidden);
      let outputs = tf.matMul(hidden, tf.transpose(this.weights_ho));
      outputs = tf.add(outputs, tf.transpose(this.bias_o));
      return tf.sigmoid(outputs); // Return a tensor and rely on the user to know what to do with it
    });
  }

  trainBatch(input_arrays, target_arrays) {
    // Batch operation requires correct orientation of the matrices
    // Generating the Hidden Outputs
    tf.tidy(() => {
      let inputs;
      if (input_arrays instanceof tf.Tensor) inputs = input_arrays;
      else inputs = tf.tensor2d(input_arrays);
      let hidden = tf.matMul(inputs, tf.transpose(this.weights_ih));
      hidden = tf.add(hidden, tf.transpose(this.bias_h));
      hidden = tf.sigmoid(hidden);
      let outputs = tf.matMul(hidden, tf.transpose(this.weights_ho));
      outputs = tf.add(outputs, tf.transpose(this.bias_o));
      outputs = tf.sigmoid(outputs);

      // Back propagation
      let targets;
      if (target_arrays instanceof tf.Tensor) targets = target_arrays;
      else targets = tf.tensor2d(target_arrays);

      // Calculate the error
      // ERROR = TARGETS - OUTPUTS
      let output_errors = tf.sub(targets, outputs).mul(this.learning_rate);

      // Calculate gradient
      // TODO: use deeplearn.js gradient functions
      let gradients = tf.mul(outputs, tf.sub(tf.scalar(1), outputs));
      gradients = tf.mul(gradients, output_errors);

      // Calculate the hidden layer errors
      let hidden_errors = tf.matMul(gradients, this.weights_ho);

      // Calculate hidden gradient
      let hidden_gradient = tf.mul(hidden, tf.sub(tf.scalar(1), hidden));
      hidden_gradient = tf.mul(hidden_gradient, hidden_errors);

      // Calcuate deltas
      let weight_ih_deltas = tf.matMul(hidden_gradient.transpose(), inputs);
      let weight_ho_deltas = tf.matMul(gradients.transpose(), hidden);

      // Adjust the weights by deltas
      this.weights_ho.assign(tf.add(this.weights_ho, weight_ho_deltas));
      this.weights_ih.assign(tf.add(this.weights_ih, weight_ih_deltas));

      // Adjust the bias by its deltas (which is just the gradients)
      this.bias_o.assign(
        tf.add(
          this.bias_o,
          tf.matMul(gradients.transpose(), tf.ones([gradients.shape[0], 1]))
        )
      );
      this.bias_h.assign(
        tf.add(
          this.bias_h,
          tf.matMul(
            hidden_gradient.transpose(),
            tf.ones([hidden_gradient.shape[0], 1])
          )
        )
      );
    });
  }
}
