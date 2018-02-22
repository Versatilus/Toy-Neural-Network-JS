let mnist;
let bubbleSize;

let train_total = 0;


// testing variables
let test_index = 0;
let total_tests = 0;
let total_correct = 0;
let average_training_time = 1;
let average_frame_time = 1;
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
let desired_frame_rate = 30;
//let training_rate_ele;

let canvasSize = _ => [800, 600];

function setup() {
  createCanvas(...canvasSize())
    .parent('container');
  // noStroke();

  stroke(51, 127);
  background(0);
  nn = new NeuralNetwork(784, 64, 10);
  // nn.setActivationFunction({
  //   func: x => x > 0 ? x : 0.01 * x,
  //   dfunc: y => y > 0 ? 1 : 0.01
  // });
  //  nn.setLearningRate(0.01);
  // let ihAverage = (nn.weights_ih.rows + nn.weights_ih.cols) / 2;
  // let hoAverage = (nn.weights_ho.rows + nn.weights_ho.cols) / 2;
  // nn.weights_ih.map(zigGauss(0, sqrt(1 / ihAverage)));
  // nn.weights_ho.map(zigGauss(0, sqrt(1 / hoAverage)));

  user_digit = createGraphics(width / 2, width / 2);
  user_digit.pixelDensity(1);

  test_image = createGraphics(width / 4, width / 4);

  user_guess_ele = select('#user_guess');
  percent_ele = select('#percent');
  total_trained_ele = select('#total_trained');

  loadMNIST(function (data) {
    mnist = data;
    //console.log(mnist);
  });
  setTimeout(train, 0);
}

function train() {
  // Train random samples continuously in the time between frames

  if (mnist) {
    // Math.max(average_frame_time, 16.666)
    let iterations = max(~~((1000.0 / desired_frame_rate - (average_frame_time)) /
      average_training_time), 1);
    let ts = millis();
    let batch_images = Array(iterations);
    let batch_labels = Array(iterations);
    for (var i = 0; i < iterations; i++) {
      let train_index = ~~(Math.random() * mnist.train_images.length);
      batch_images[i] = [];
      for (let j = 0; j < mnist.train_images[train_index].length; j++) {
        batch_images[i][j] = mnist.train_images[train_index][j]/ 255;
      }
      // Do the neural network stuff;
      let label = mnist.train_labels[train_index];
      batch_labels[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
      batch_labels[i][label] = 1;
      // nn.train(inputs, targets);
      // train_total++;
      // average_training_time = (average_training_time * 4 + (millis() - ts)) / 5;
    }
    nn.trainBatch(batch_images, batch_labels);
    train_total += iterations;
    average_training_time = (average_training_time * 4 + (millis() - ts)/iterations) / 5;
  } else {
    time_training_started = millis();
  }
  setTimeout(train, average_frame_time);
}

function testing() {
  let inputs = [];
  for (let i = 0; i < mnist.test_images[test_index].length; i++) {
    let bright = mnist.test_images[test_index][i];
    inputs[i] = bright / 255;
  }
  let label = mnist.test_labels[test_index];

  let prediction = nn.predict(inputs);
  let guess = findMax(prediction);
  total_tests++;
  if (guess == label) {
    total_correct++;
  }

  let percent = nf(100 * (total_correct / total_tests), 2, 2) + '%';
  percent_ele.html(percent);

  test_index++;
  if (test_index == mnist.test_labels.length) {
    test_index = 0;
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
    let test_batch_size = 5;
    for (let i = 0; i < test_batch_size; i++) {
      testing();
    }
    twa = (twa * 179 + (train_total - twto) / ((millis() - twti) * 0.001)) / 180;
    total_trained_ele.html(train_total + " (" + nf(twa, 0, 2) + " samples/s windowed, " + nf(
        train_total / ((millis() - time_training_started) * 0.001), 0, 2) +
      " samples/s overall)");
    if (frameCount % 61 == 0) {
      //test_image.background(0, 240);
      // if (frameCount % 66 == 0)
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
    image(test_image, width / 2, 0, width / 2, width / 2)


  }

  image(user_digit, 0, 0, width / 2, width / 2);

  if (mouseIsPressed) {
    user_has_drawing = true;
    user_digit.stroke(255);
    user_digit.strokeWeight(user_digit.width / 15);
    user_digit.line(mouseX, mouseY, pmouseX, pmouseY);
  }
  stroke(255);
  line(width / 2, 0, width / 2, width / 2);
  line(0, width / 2, width, width / 2);
  average_frame_time = (average_frame_time * 4 + (millis() - ts)) / 5;
}

function activationBox() {
  let tl = [0, width / 2];
  let tr = [width / 2, width / 2];
  let bl = [0, height];
  let br = [width / 2, height];
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
