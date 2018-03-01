// ESLint comment code to quiet my linter. It also provides a form of documentation.
/* global dl, NeuralNetwork, createCanvas, createSlider, rect, fill, noStroke, random, background,
   canvas */
/* exported setup, draw */

let nn;
let lr_slider;
let test_inputs = [];
let resolution = 10;

/* eslint-disable indent*/
let training_data = [{
    inputs: [0, 0],
    outputs: [0]
  },
  {
    inputs: [0, 1],
    outputs: [1]
  },
  {
    inputs: [1, 0],
    outputs: [1]
  },
  {
    inputs: [1, 1],
    outputs: [0]
  }
];
/* eslint-enable indent*/

function setup() {
  createCanvas(400, 400);
  //dl.setBackend('cpu');
  nn = new NeuralNetwork(2, 4, 1);
  lr_slider = createSlider(0.01, 0.5, 0.1, 0.01);

  let cols = canvas.width / resolution;
  let rows = canvas.height / resolution;
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      let x1 = i / cols;
      let x2 = j / rows;
      test_inputs.push([x1, x2]);
    }
  }
}

function draw() {
  background(0);
  let batch_inputs = [],
    batch_outputs = [];
  for (let i = 0; i < 15; i++) {
    let data = random(training_data);
    batch_inputs[i] = data.inputs;
    batch_outputs[i] = data.outputs;
  }
  nn.trainBatch(batch_inputs, batch_outputs);
  nn.setLearningRate(lr_slider.value());
  let cols = canvas.width / resolution;
  let rows = canvas.height / resolution;
  let scratch = nn.predictBatch(test_inputs);
  let outputs = dl.tidy(() => (scratch.flatten()
    .getValues()));
  scratch.dispose();
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      noStroke();
      fill(outputs[i + cols * j] * 255);
      rect(i * resolution, j * resolution, resolution, resolution);
    }
  }
}
