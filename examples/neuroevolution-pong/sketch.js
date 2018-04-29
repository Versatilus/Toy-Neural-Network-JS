// port of Daniel Shiffman's Pong coding challenge
// by madacoo

let bots = [];
let winners = [];
let losers = [];
let bestScore = -Infinity;
let bestBot;
let leftBot;
let rightBot;
let puck;
let left;
let right;


function setup() {
  createCanvas(600, 400);
  // ding = loadSound("data/ding.mp3");
  puck = new Puck();
  left = new Paddle(true);
  right = new Paddle(false);
  bots = Array.from({length:4},()=>new Bot());
  leftBot = bots[0];
  rightBot = bots[0];
  puck.reset();
}

function draw() {
  background(0);
  for (var i = 0; i < 3; i++) {
      
    
    leftBot.paddle = left;
    rightBot.think();
    rightBot.paddle = right;
    rightBot.think();
    puck.checkPaddleRight(right);
    puck.checkPaddleLeft(left);
    
    left.update();
    right.update();
    
    puck.update();
    puck.edges();
  }
  left.show();
  right.show();
  puck.show();
    
  fill(255);
  textSize(32);
  textAlign(LEFT);
  text(left.score, 32, 40);
  textAlign(RIGHT);
  text(right.score, width-32, 40);
}


function keyReleased() {
  left.move(0);
  right.move(0);
}

function keyPressed() {
  console.log(key);
  if (key == 'A') {
    left.move(-10);
  } else if (key == 'Z') {
    left.move(10);
  }

  if (key == 'J') {
    right.move(-10);
  } else if (key == 'M') {
    right.move(10);
  }
}

function finalizeSet() {
  leftBot.calculateFitness();
  rightBot.calculateFitness();
  if (leftBot.fitness>rightBot.fitness){
    winners.push(leftBot);
  }else if (leftBot.fitness<rightBot.fitness){
    winners.push(rightBot);}
  else{
    let scratch = leftBot;
    leftBot = rightBot;
    rightBot = scratch;
    // return rematch();
  }
}
