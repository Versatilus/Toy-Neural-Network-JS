let nn;
let lr_slider;
let test_inputs = [];
let resolution = 10;

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

function setup() {
  createCanvas(400, 400);
  //dl.setBackend('cpu');
  nn = new NeuralNetwork(2, 4, 1);
  lr_slider = createSlider(0.01, 0.5, 0.1, 0.01);

  let cols = width / resolution;
  let rows = height / resolution;
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      let x1 = i / cols;
      let x2 = j / rows;
      test_inputs.push([x1, x2]);
    }
  }
}

let testing = 0;

function draw() {
  background(0);
  // noLoop();

  // if (frameCount > 100) {
  //   console.log('done');
  //   noLoop();
  // }
  let batch_inputs = [],
    batch_outputs = [];
  for (let i = 0; i < 10; i++) {
    let data = random(training_data);
    batch_inputs[i] = data.inputs;
    batch_outputs[i] = data.outputs;
    // nn.train(data.inputs, data.outputs);
  }
  nn.trainBatch(batch_inputs, batch_outputs);
  //nn.setLearningRate(lr_slider.value());
  let cols = width / resolution;
  let rows = height / resolution;
  batch_outputs = nn.predictBatch(test_inputs);
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {

      //let y = nn.predict(inputs);
      noStroke();
      fill(batch_outputs.pop() * 255);
      rect(i * resolution, j * resolution, resolution, resolution);
    }
  }



}
