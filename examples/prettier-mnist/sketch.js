// ESLint comment code to quiet my linter. It also provides a form of documentation.
/* global dl, NeuralNetwork, createCanvas, random, background, key, frameRate, millis, line,
   canvas, stroke, createGraphics, nf, frameCount, loadMNIST, select, max, sqrt, mouseIsPressed,
   image, pmouseX, pmouseY, mouseX, mouseY */
/* exported setup, draw, keyPressed */

let mnist;
let bubbleSize;

let train_total = 0;
let training_batch_size = 20;
let test_batch_size = 20;
let desired_frame_rate = 60;

// testing variables
let test_index = 0;
let total_tests = 0;
let total_correct = 0;
let average_training_time = 1;
let average_frame_time = 1;
let average_frame_rate = 60;
let twti = performance.now();
let time_training_started = twti;
let twto = 0;
let twa = 250;
let nn;
let test_image;
let current_test_image = Array(784)
  .fill(0);
let user_digit;
let user_has_drawing = false;

let user_guess_ele;
let percent_ele;
let total_trained_ele;

let canvasSize = () => [800, 600];

function setup() {
  createCanvas(...canvasSize())
    .parent('container');
  // noStroke();

  stroke(51, 127);
  background(0);
  nn = new NeuralNetwork(784, 64, 10);

  user_digit = createGraphics(canvas.width / 2, canvas.width / 2);
  user_digit.pixelDensity(1);

  test_image = createGraphics(canvas.width / 4, canvas.width / 4);

  user_guess_ele = select('#user_guess');
  percent_ele = select('#percent');
  total_trained_ele = select('#total_trained');

  loadMNIST(function (data) {
    mnist = data;
    train();
  });
}

function train() {
  // Train random samples continuously in the time between frames

  if (mnist) {
    Math.max(average_frame_time, 16.666);
    let iterations = max(~~((1000.0 / desired_frame_rate - (average_frame_time)) /
      (training_batch_size * average_training_time)), 1);
    for (let k = 0; k < iterations; k++) {
      let ts = millis();
      let batch_images = Array(training_batch_size);
      let batch_labels = Array(training_batch_size);
      for (var i = 0; i < training_batch_size; i++) {
        let train_index = ~~(Math.random() * mnist.train_images.length);
        batch_images[i] = [];
        for (let j = 0; j < mnist.train_images[train_index].length; j++) {
          batch_images[i][j] = mnist.train_images[train_index][j] / 255;
        }
        // Do the neural network stuff;
        let label = mnist.train_labels[train_index];
        batch_labels[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        batch_labels[i][label] = 1;
      }
      nn.trainBatch(batch_images, batch_labels);
      train_total += training_batch_size;
      average_training_time = (average_training_time * 4 + ((millis() - ts) / training_batch_size)) /
        5;
    }
  } else {
    time_training_started = millis();
  }
  setTimeout(train, 0);
}

function testing() {
  let
    current, labels = [],
    images = Array.from({
      'length': test_batch_size
    }, () => []);

  for (let j = 0; j < images.length; j++) {
    current = ~~random(mnist.test_images.length);
    for (let i = 0; i < mnist.test_images[current].length; i++) {
      images[j][i] = mnist.test_images[current][i] / 255;
    }
    labels[j] = mnist.test_labels[current];
  }
  let scratch = nn.predictBatch(images);
  let predictions = dl.tidy(() => (scratch.flatten()
    .getValues()));
  scratch.dispose();
  for (let i = 0; i < labels.length; i++) {
    let guess = findMax(predictions.slice(i * nn.output_nodes, (i + 1) * nn.output_nodes));
    total_tests++;
    if (guess == labels[i]) {
      total_correct++;
    }
  }

  let percent = nf(100 * (total_correct / total_tests), 2, 2) + '%';
  percent_ele.html(percent);

  test_index = current;
  if (total_tests >= mnist.test_labels.length) {
    console.log('finished test set');
    console.log(percent);
    total_tests = 0;
    total_correct = 0;
  }

}

function guessUserDigit() {
  let img = user_digit.get();
  if (!user_has_drawing) {
    user_guess_ele.html('_');
    return img;
  }
  let inputs = [];
  img.resize(28, 28);
  img.loadPixels();
  for (let i = 0; i < 784; i++) {
    inputs[i] = img.pixels[i * 4] / 255;
  }
  let prediction = nn.predict(inputs);
  let guess = findMax(prediction);
  user_guess_ele.html(guess);
  return img;
}


function draw() {
  // setTimeout(train, 0);
  let ts = millis();


  if (frameCount % 15 == 0)
    guessUserDigit();

  if (mnist) {
    testing();

    twa = (twa * 9 + (train_total - twto) / ((millis() - twti) * 0.001)) / 10;
    /* eslint-disable indent*/
    total_trained_ele.html(train_total + " (" + nf(twa, 0, 2) + " samples/s windowed, " + nf(
        train_total / ((millis() - time_training_started) * 0.001), 0, 2) +
      " samples/s overall)");
    /* eslint-enable indent*/
    if (frameCount % 61 == 0) {
      current_test_image = mnist.test_images[test_index];
      let ymax = ~~sqrt(current_test_image.length);
      let xmax = ymax;
      bubbleSize = (test_image.height / xmax);
      let offset = bubbleSize / 2;
      for (let i = 0; i < ymax; i++)
        for (let j = 0; j < xmax; j++) {
          test_image.fill(current_test_image[i * ymax + j]);
          test_image.ellipse(offset + j * bubbleSize, offset + i * bubbleSize, bubbleSize);
        }
      twti = millis();
      twto = train_total;
    }
    image(test_image, canvas.width / 2, 0, canvas.width / 2, canvas.width / 2);
  }

  image(user_digit, 0, 0, canvas.width / 2, canvas.width / 2);

  if (mouseIsPressed) {
    user_has_drawing = true;
    user_digit.stroke(255);
    user_digit.strokeWeight(user_digit.width / 15);
    user_digit.line(mouseX, mouseY, pmouseX, pmouseY);
  }
  stroke(255);
  line(canvas.width / 2, 0, canvas.width / 2, canvas.width / 2);
  line(0, canvas.width / 2, canvas.width, canvas.width / 2);
  average_frame_time = (average_frame_time * 4 + (millis() - ts)) / 5;
  average_frame_rate = (average_frame_rate * 3 + frameRate()) / 4;
}

function keyPressed() {
  if (key == ' ') {
    user_has_drawing = false;
    user_digit.background(0);
  }
}

function findMax(arr) {
  let record = 0;
  let index = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > record) {
      record = arr[i];
      index = i;
    }
  }
  return index;

}
