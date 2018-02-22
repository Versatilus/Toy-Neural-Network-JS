function gaussianGenerator(mu = 0, sigma = 1, nsamples = 5) {
  return function () {
    let sum = 0;
    for (let i = 0; i < nsamples; i++) {
      sum += 2 * Math.random() - 1;
    }
    let halfSamples = nsamples * 0.5;
    return mu + sigma * (sum - halfSamples) / halfSamples;
  }
}

// returns a gaussian random function with the given mean and stdev.
function gaussianBell(mean, stdev) {
  var y2;
  var use_last = false;
  return function () {
    var y1;
    if (use_last) {
      y1 = y2;
      use_last = false;
    } else {
      var x1, x2, w;
      do {
        x1 = 2.0 * Math.random() - 1.0;
        x2 = 2.0 * Math.random() - 1.0;
        w = x1 * x1 + x2 * x2;
      } while (w >= 1.0);
      w = Math.sqrt((-2.0 * Math.log(w)) / w);
      y1 = x1 * w;
      y2 = x2 * w;
      use_last = true;
    }

    return mean + stdev * y1;
  };
}

function randn_bm(mu, sigma) {
  return function () {
    var u = 0;
    while (u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    return mu + sigma * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * Math.random());
  };
}

function Ziggurat() {

  var jsr = 123456789;

  var wn = Array(128);
  var fn = Array(128);
  var kn = Array(128);

  function RNOR() {
    var hz = SHR3();
    var iz = hz & 127;
    return (Math.abs(hz) < kn[iz]) ? hz * wn[iz] : nfix(hz, iz);
  }

  this.nextGaussian = function () {
    return RNOR();
  }

  function nfix(hz, iz) {
    var r = 3.442619855899;
    var r1 = 1.0 / r;
    var x;
    var y;
    while (true) {
      x = hz * wn[iz];
      if (iz == 0) {
        x = (-Math.log(UNI()) * r1);
        y = -Math.log(UNI());
        while (y + y < x * x) {
          x = (-Math.log(UNI()) * r1);
          y = -Math.log(UNI());
        }
        return (hz > 0) ? r + x : -r - x;
      }

      if (fn[iz] + UNI() * (fn[iz - 1] - fn[iz]) < Math.exp(-0.5 * x * x)) {
        return x;
      }
      hz = SHR3();
      iz = hz & 127;

      if (Math.abs(hz) < kn[iz]) {
        return (hz * wn[iz]);
      }
    }
  }

  function SHR3() {
    var jz = jsr;
    var jzr = jsr;
    jzr ^= (jzr << 13);
    jzr ^= (jzr >>> 17);
    jzr ^= (jzr << 5);
    jsr = jzr;
    return (jz + jzr) | 0;
  }

  function UNI() {
    return 0.5 * (1 + SHR3() / -Math.pow(2, 31));
  }

  function zigset() {
    // seed generator based on current time
    jsr ^= new Date()
      .getTime();

    var m1 = 2147483648.0;
    var dn = 3.442619855899;
    var tn = dn;
    var vn = 9.91256303526217e-3;

    var q = vn / Math.exp(-0.5 * dn * dn);
    kn[0] = Math.floor((dn / q) * m1);
    kn[1] = 0;

    wn[0] = q / m1;
    wn[127] = dn / m1;

    fn[0] = 1.0;
    fn[127] = Math.exp(-0.5 * dn * dn);

    for (var i = 126; i >= 1; i--) {
      dn = Math.sqrt(-2.0 * Math.log(vn / dn + Math.exp(-0.5 * dn * dn)));
      kn[i + 1] = Math.floor((dn / tn) * m1);
      tn = dn;
      fn[i] = Math.exp(-0.5 * dn * dn);
      wn[i] = dn / m1;
    }
  }
  zigset();
}

function zigGauss(mu = 0, sigma = 1) {
  let z = new Ziggurat();
  return _ => mu + sigma * z.nextGaussian();
}
