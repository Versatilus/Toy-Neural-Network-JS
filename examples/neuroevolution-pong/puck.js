class Puck {
    constructor() {
        this.x = width/2;
        this.y = height/2;
        this.xspeed = 0;
        this.yspeed = 0;
        this.r = 12;
    }
    
    checkPaddleLeft(p) {
        if (this.y - this.r < p.y + p.h/2 &&
            this.y + this.r > p.y - p.h/2 &&
            this.x - this.r < p.x + p.w/2) {
                
            if (this.x > p.x) {
                let diff = this.y - (p.y - p.h/2);
                let rad = radians(45);
                let angle = map(diff, 0, p.h, -rad, rad);
                this.xspeed = 5 * cos(angle);
                this.yspeed = 5 * sin(angle);
                this.x = p.x + p.w/2 + this.r;
                left.volley++;
            }
            
        }
    }
    checkPaddleRight(p) {
        if (this.y - this.r < p.y + p.h/2 &&
            this.y + this.r > p.y - p.h/2 &&
            this.x + this.r > p.x - p.w/2) {
                
            if (this.x < p.x) {
                let diff = this.y - (p.y - p.h/2);
                let angle = map(diff, 0, p.h, radians(225), radians(135));
                this.xspeed = 5 * cos(angle);
                this.yspeed = 5 * sin(angle);
                this.x = p.x - p.w/2 - this.r;
                right.volley++;
            }
        }
    }
    
    update() {
        this.x += this.xspeed;
        this.y += this.yspeed;
    }
    
    reset() {
        this.x = width/2;
        this.y = height/2;
        let angle = random(-PI/4, PI/4);
        this.xspeed = 5 * Math.cos(angle);
        this.yspeed = 5 * Math.sin(angle);
        
        if (random(1) < 0.5) {
            this.xspeed *= -1;
        }
        if (left.score===10||right.score===10) {
          finalizeSet();
        }
    }
    
    edges() {
        if (this.y < this.r || this.y > height-this.r) {
            this.yspeed *= -1;
        }
        
        if (this.x - this.r > width) {
            // ding.play();
            left.score++;
            this.reset();
        }
        
        if (this.x + this.r < 0) {
            // ding.play();
            right.score++;
            this.reset();
        }
    }
    
    show() {
        fill(255);
        ellipse(this.x, this.y, this.r*2);
    }
}
