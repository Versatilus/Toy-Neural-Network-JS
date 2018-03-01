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

let tanh = new ActivationFunction(
  x => Math.tanh(x),
  y => 1 - (y * y)
);


class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    let ihVariance = Math.sqrt((input_nodes + hidden_nodes) * 0.5);
    let hoVariance = Math.sqrt((output_nodes + hidden_nodes) * 0.5);
    dl.tidy(() => {
      this.weights_ih = dl.variable(dl.randomNormal([this.hidden_nodes, this.input_nodes],
        0, 1 / ihVariance), true, "weights1");
      this.weights_ho = dl.variable(dl.randomNormal([this.output_nodes, this.hidden_nodes],
        0, 1 / hoVariance), true, "weights2");
      this.bias_h = dl.variable(dl.randomNormal([this.hidden_nodes, 1], 0, Math.sqrt(1 /
        this.hidden_nodes)), true, "bias1");
      this.bias_o = dl.variable(dl.randomNormal([this.output_nodes, 1], 0, Math.sqrt(1 /
        this.output_nodes)), true, "bias2");
      this.hidden_activations = dl.variable(dl.zeros([this.hidden_nodes, 1]), true,
        "activation1");
      this.output_activations = dl.variable(dl.zeros([this.output_nodes, 1]), true,
        "activation2");
    });
    this.setLearningRate();

    // this.setActivationFunction(); // currently a no-op
  }

  predict(input_array) {
    /* TODO: It should be fairly straightforward to refactor this to handle both individual and
       batch processing by reorienting the matrices and letting broadcasting work it's miracles. */

    return dl.tidy(() => {
      // Generating the Hidden Outputs
      let inputs;
      if (input_array instanceof dl.Tensor)
        inputs = input_array;
      else
        inputs = dl.tensor2d(input_array, [input_array.length, 1]);

      this.hidden_activations.assign(dl.matMul(this.weights_ih, inputs));

      this.hidden_activations.assign(dl.add(this.hidden_activations, this.bias_h));

      // activation function!
      this.hidden_activations.assign(dl.sigmoid(this.hidden_activations));

      // Generating the output's output!
      this.output_activations.assign(dl.matMul(this.weights_ho, this.hidden_activations));

      this.output_activations.assign(dl.add(this.output_activations, this.bias_o));
      this.output_activations.assign(dl.sigmoid(this.output_activations));

      // Sending back to the caller!
      return this.output_activations.getValues();
    });
  }

  setLearningRate(learning_rate = 0.1) {
    if (this.learning_rate instanceof dl.Tensor)
      this.learning_rate.dispose();

    this.learning_rate = dl.scalar(learning_rate);
  }

  setActivationFunction(func = sigmoid) {
    // currently a no-op
    this.activation_function = func;
  }

  train(input_array, target_array) {
    // Generating the Hidden Outputs
    dl.tidy(() => {
      let inputs;
      if (input_array instanceof dl.Tensor)
        inputs = input_array;
      else
        inputs = dl.tensor2d(input_array, [input_array.length, 1]);
      this.predict(input_array);

      let targets = dl.tensor2d(target_array, [target_array.length, 1]);

      // Calculate the error
      // ERROR = TARGETS - OUTPUTS
      let output_errors = dl.sub(targets, this.output_activations);

      // Calculate gradient
      // TODO: use deeplearn.js gradient functions
      let gradients = dl.mul(this.output_activations, dl.sub(dl.scalar(1), this.output_activations));
      gradients = dl.mul(gradients, output_errors)
        .mul(this.learning_rate);

      // Calculate deltas
      let hidden_T = dl.transpose(this.hidden_activations);
      let weight_ho_deltas = dl.matMul(gradients, hidden_T);

      // Calculate the hidden layer errors
      let who_t = dl.transpose(this.weights_ho);
      let hidden_errors = dl.matMul(who_t, output_errors);

      // Calculate hidden gradient
      let hidden_gradient = dl.mul(this.hidden_activations, dl.sub(dl.scalar(1), this.hidden_activations));
      hidden_gradient = dl.mul(hidden_gradient, hidden_errors)
        .mul(this.learning_rate);

      // Calcuate input->hidden deltas
      let inputs_T = dl.transpose(inputs);
      let weight_ih_deltas = dl.matMul(hidden_gradient, inputs_T);

      // Adjust the weights by deltas
      this.weights_ho.assign(dl.add(this.weights_ho, weight_ho_deltas));
      this.weights_ih.assign(dl.add(this.weights_ih, weight_ih_deltas));

      // Adjust the bias by its deltas (which is just the gradients)
      this.bias_o.assign(dl.add(this.bias_o, gradients));
      this.bias_h.assign(dl.add(this.bias_h, hidden_gradient));
    });
  }

  predictBatch(input_arrays) {
    // The trick to batch processing is correct orientation of the matrices.
    return dl.tidy(() => {
      let inputs;
      if (input_arrays instanceof dl.Tensor)
        inputs = input_arrays;
      else
        inputs = dl.tensor2d(input_arrays);
      let hidden = dl.matMul(inputs, dl.transpose(this.weights_ih));
      hidden = dl.add(hidden, dl.transpose(this.bias_h));
      hidden = dl.sigmoid(hidden);
      let outputs = dl.matMul(hidden, dl.transpose(this.weights_ho));
      outputs = dl.add(outputs, dl.transpose(this.bias_o));
      return dl.sigmoid(outputs); // Return a tensor and rely on the user to know what to do with it
    });
  }

  trainBatch(input_arrays, target_arrays) {
    // Batch operation requires correct orientation of the matrices
    // Generating the Hidden Outputs
    dl.tidy(() => {
      let inputs;
      if (input_arrays instanceof dl.Tensor)
        inputs = input_arrays;
      else
        inputs = dl.tensor2d(input_arrays);
      let hidden = dl.matMul(inputs, dl.transpose(this.weights_ih));
      hidden = dl.add(hidden, dl.transpose(this.bias_h));
      hidden = dl.sigmoid(hidden);
      let outputs = dl.matMul(hidden, dl.transpose(this.weights_ho));
      outputs = dl.add(outputs, dl.transpose(this.bias_o));
      outputs = dl.sigmoid(outputs);

      // Back propagation
      let targets;
      if (target_arrays instanceof dl.Tensor)
        targets = target_arrays;
      else
        targets = dl.tensor2d(target_arrays);

      // Calculate the error
      // ERROR = TARGETS - OUTPUTS
      let output_errors = dl.sub(targets, outputs)
        .mul(this.learning_rate);

      // Calculate gradient
      // TODO: use deeplearn.js gradient functions
      let gradients = dl.mul(outputs, dl.sub(dl.scalar(1), outputs));
      gradients = dl.mul(gradients, output_errors);

      // Calculate the hidden layer errors
      let hidden_errors = dl.matMul(gradients, this.weights_ho);

      // Calculate hidden gradient
      let hidden_gradient = dl.mul(hidden, dl.sub(dl.scalar(1), hidden));
      hidden_gradient = dl.mul(hidden_gradient, hidden_errors);

      // Calcuate deltas
      let weight_ih_deltas = dl.matMul(hidden_gradient.transpose(), inputs);
      let weight_ho_deltas = dl.matMul(gradients.transpose(), hidden);

      // Adjust the weights by deltas
      this.weights_ho.assign(dl.add(this.weights_ho, weight_ho_deltas));
      this.weights_ih.assign(dl.add(this.weights_ih, weight_ih_deltas));

      // Adjust the bias by its deltas (which is just the gradients)
      this.bias_o.assign(dl.add(this.bias_o, dl.matMul(gradients.transpose(), dl.ones(
        [gradients.shape[0], 1]))));
      this.bias_h.assign(dl.add(this.bias_h, dl.matMul(hidden_gradient.transpose(), dl.ones(
        [hidden_gradient.shape[0], 1]))));
    });
  }
}
