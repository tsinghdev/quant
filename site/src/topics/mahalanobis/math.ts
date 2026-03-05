export function mulberry32(a: number) {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function boxMullerSeeded(rng: () => number) {
  const u1 = rng();
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
}

export function generateCorrelatedData(
  n: number, sx: number, sy: number, corr: number, seed = 42
): [number, number][] {
  const rng = mulberry32(seed);
  const points: [number, number][] = [];
  for (let i = 0; i < n; i++) {
    const z1 = boxMullerSeeded(rng);
    const z2 = boxMullerSeeded(rng);
    const x = sx * z1;
    const y = sy * (corr * z1 + Math.sqrt(1 - corr * corr) * z2);
    points.push([x, y]);
  }
  return points;
}

type Mat2 = [[number, number], [number, number]];

export function matInv2x2(m: Mat2): Mat2 {
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
  return [
    [m[1][1] / det, -m[0][1] / det],
    [-m[1][0] / det, m[0][0] / det],
  ];
}

export function mahalanobis(x: number, y: number, sigmaInv: Mat2): number {
  const t0 = sigmaInv[0][0] * x + sigmaInv[0][1] * y;
  const t1 = sigmaInv[1][0] * x + sigmaInv[1][1] * y;
  return Math.sqrt(x * t0 + y * t1);
}

export function euclidean(x: number, y: number): number {
  return Math.sqrt(x * x + y * y);
}

export function ellipsePoints(sigmaInv: Mat2, r: number, nPts = 80): [number, number][] {
  const pts: [number, number][] = [];
  for (let i = 0; i <= nPts; i++) {
    const theta = (2 * Math.PI * i) / nPts;
    const cx = Math.cos(theta);
    const cy = Math.sin(theta);
    const a =
      sigmaInv[0][0] * cx * cx +
      (sigmaInv[0][1] + sigmaInv[1][0]) * cx * cy +
      sigmaInv[1][1] * cy * cy;
    const scale = r / Math.sqrt(a);
    pts.push([scale * cx, scale * cy]);
  }
  return pts;
}
