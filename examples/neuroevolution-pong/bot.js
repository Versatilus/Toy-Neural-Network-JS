class Bot{
  constructor(isLeft=true, brain){
    this.brain = brain?brain.copy():new NeuralNetwork(10,10,1);
    this.brain.setActivationFunction(tanh);
    this.paddle = isLeft?left:right;
    this.isLeft = isLeft;
    this.fitness = 0;
  }
  think(){
    let action;
    if(this.isLeft)action = this.brain.predict([
      this.paddle.x,
      this.paddle.y,
      this.paddle.ychange,
      right.x,
      right.y,
      right.ychange,
      puck.x,
      puck.xspeed,
      puck.y,
      puck.yspeed
    ])[0];
    else action = -this.brain.predict([
      width-this.paddle.x,
      height-this.paddle.y,
      -this.paddle.ychange,
      width-left.x,
      height-left.y,
      -left.ychange,
      width-puck.x,
      -puck.xspeed,
      height-puck.y,
      -puck.yspeed
    ])[0];
    this.paddle.move(round(action*10));
  }
  calculateFitness(){
    let diff = this.isLeft?left.score-right.score:right.score -left.score;
    this.fitness = diff+this.paddle.volley*2;
  }
}