import { useState, useEffect, useCallback, useRef } from 'react'
import { Link } from 'react-router-dom'
import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'
import {
  generateCorrelatedData, matInv2x2, mahalanobis, euclidean, ellipsePoints,
} from './math'
import './mahalanobis.css'

type Mat2 = [[number, number], [number, number]]

const W = 480, H = 420
const CENTER_X = W / 2, CENTER_Y = H / 2
const SCALE = 28
const GRID = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]

function toSvg(p: [number, number]): [number, number] {
  return [CENTER_X + p[0] * SCALE, CENTER_Y - p[1] * SCALE]
}
function fromSvg(sx: number, sy: number): [number, number] {
  return [(sx - CENTER_X) / SCALE, -(sy - CENTER_Y) / SCALE]
}

const AXIS_TICKS = [-6, -4, -2, 2, 4, 6]

function Grid() {
  return (
    <>
      {GRID.map(v => (
        <g key={v}>
          <line x1={CENTER_X + v * SCALE} y1={0} x2={CENTER_X + v * SCALE} y2={H}
            stroke="#3d3d37" strokeWidth={v === 0 ? 1.5 : 0.5} />
          <line x1={0} y1={CENTER_Y - v * SCALE} x2={W} y2={CENTER_Y - v * SCALE}
            stroke="#3d3d37" strokeWidth={v === 0 ? 1.5 : 0.5} />
        </g>
      ))}
      {AXIS_TICKS.map(v => (
        <g key={`tick-${v}`}>
          <text x={CENTER_X + v * SCALE} y={CENTER_Y + 16}
            fill="#6b6052" fontSize={9} textAnchor="middle"
            fontFamily="'JetBrains Mono', monospace">{v}</text>
          <text x={CENTER_X - 14} y={CENTER_Y - v * SCALE + 3}
            fill="#6b6052" fontSize={9} textAnchor="end"
            fontFamily="'JetBrains Mono', monospace">{v}</text>
        </g>
      ))}
      <text x={W - 8} y={CENTER_Y - 8} fill="#7a756a" fontSize={11}
        textAnchor="end" fontFamily="'JetBrains Mono', monospace"
        fontStyle="italic">x</text>
      <text x={CENTER_X + 10} y={14} fill="#7a756a" fontSize={11}
        textAnchor="start" fontFamily="'JetBrains Mono', monospace"
        fontStyle="italic">y</text>
    </>
  )
}

function SceneCompare() {
  const [corr, setCorr] = useState(0.75)
  const [probe, setProbe] = useState<[number, number]>([2.5, -1.0])
  const [dragging, setDragging] = useState(false)
  const svgRef = useRef<SVGSVGElement>(null)

  const sx = 2.5, sy = 1.2
  const sigma: Mat2 = [
    [sx * sx, corr * sx * sy],
    [corr * sx * sy, sy * sy],
  ]
  const sigmaInv = matInv2x2(sigma)
  const data = generateCorrelatedData(200, sx, sy, corr, 42)

  const dM = mahalanobis(probe[0], probe[1], sigmaInv)
  const dE = euclidean(probe[0], probe[1])

  const ellM = ellipsePoints(sigmaInv, dM, 100)
  const ellMPath = ellM.map((p, i) => {
    const [ex, ey] = toSvg(p)
    return (i === 0 ? 'M' : 'L') + ex + ',' + ey
  }).join(' ') + 'Z'

  const [px, py] = toSvg(probe)

  const handleMove = useCallback((clientX: number, clientY: number) => {
    if (!dragging || !svgRef.current) return
    const rect = svgRef.current.getBoundingClientRect()
    const [wx, wy] = fromSvg(clientX - rect.left, clientY - rect.top)
    setProbe([Math.max(-7, Math.min(7, wx)), Math.max(-6, Math.min(6, wy))])
  }, [dragging])

  useEffect(() => {
    const up = () => setDragging(false)
    window.addEventListener('mouseup', up)
    window.addEventListener('touchend', up)
    return () => { window.removeEventListener('mouseup', up); window.removeEventListener('touchend', up) }
  }, [])

  return (
    <div>
      <div className="data-note">
        <div className="data-note-title">Data generation</div>
        <p>
          We sample <InlineMath math="n = 200" /> points from a bivariate normal
          {' '}<InlineMath math="(x, y) \sim \mathcal{N}(\mathbf{0}, \Sigma)" /> using
          the Cholesky decomposition. Let <InlineMath math="z_1, z_2 \sim \mathcal{N}(0,1)" /> be
          independent standard normals. Then:
        </p>
        <BlockMath math={`x = \\sigma_x \\, z_1, \\qquad y = \\sigma_y \\left( \\rho \\, z_1 + \\sqrt{1 - \\rho^2} \\, z_2 \\right)`} />
        <p>
          This yields the covariance structure:
        </p>
        <BlockMath math={`\\Sigma = \\begin{pmatrix} \\sigma_x^2 & \\rho \\, \\sigma_x \\sigma_y \\\\ \\rho \\, \\sigma_x \\sigma_y & \\sigma_y^2 \\end{pmatrix} = \\begin{pmatrix} ${(sx * sx).toFixed(2)} & ${(corr * sx * sy).toFixed(2)} \\\\ ${(corr * sx * sy).toFixed(2)} & ${(sy * sy).toFixed(2)} \\end{pmatrix}`} />
        <p>
          with <InlineMath math={`\\sigma_x = ${sx}`} />, <InlineMath math={`\\sigma_y = ${sy}`} />,
          and <InlineMath math={`\\rho = ${corr.toFixed(2)}`} />.
          The seed is fixed, so the underlying <InlineMath math="z_1, z_2" /> draws
          are identical across all values of <InlineMath math="\rho" /> — only
          the linear transformation changes.
        </p>
        <p>
          The Mahalanobis distance from a point <InlineMath math="\mathbf{x}" /> to
          the origin is then <InlineMath math="d_M = \sqrt{\mathbf{x}^\top \Sigma^{-1} \mathbf{x}}" />,
          while the Euclidean distance is simply <InlineMath math="d_E = \|\mathbf{x}\|_2" />.
          Note that <InlineMath math="d_M = d_E" /> when <InlineMath math="\Sigma = I" />.
        </p>
      </div>

      <div className="scene-row">
        <svg ref={svgRef} width={W} height={H} className="scene-svg"
          style={{ cursor: dragging ? 'grabbing' : 'default', touchAction: 'none' }}
          onMouseMove={e => handleMove(e.clientX, e.clientY)}
          onTouchMove={e => { e.preventDefault(); handleMove(e.touches[0].clientX, e.touches[0].clientY) }}
        >
          <Grid />
          {data.map((p, i) => {
            const [dx, dy] = toSvg(p)
            return <circle key={i} cx={dx} cy={dy} r={2} fill="#c49a6c" opacity={0.25} />
          })}
          <circle cx={CENTER_X} cy={CENTER_Y} r={dE * SCALE}
            fill="none" stroke="#d98a7a" strokeWidth={1.5} strokeDasharray="6,4" opacity={0.7} />
          <path d={ellMPath} fill="none" stroke="#7aaf6e" strokeWidth={2} opacity={0.85} />
          <line x1={CENTER_X} y1={CENTER_Y} x2={px} y2={py}
            stroke="#e8e4d9" strokeWidth={1} strokeDasharray="3,3" opacity={0.4} />
          <circle cx={px} cy={py} r={8} fill="#e8e4d9" stroke="#c49a6c" strokeWidth={2}
            style={{ cursor: 'grab' }}
            onMouseDown={() => setDragging(true)}
            onTouchStart={() => setDragging(true)} />
          <circle cx={CENTER_X} cy={CENTER_Y} r={3} fill="#e8e4d9" opacity={0.6} />
        </svg>

        <div className="scene-panel">
          <div style={{ marginBottom: 20 }}>
            <label className="slider-label">CORRELATION: {corr.toFixed(2)}</label>
            <input type="range" min={-0.95} max={0.95} step={0.05}
              value={corr} onChange={e => setCorr(+e.target.value)} style={{ width: '100%' }} />
          </div>
          <div className="info-card">
            <div className="metric-row">
              <div className="metric-swatch" style={{ background: '#d98a7a' }} />
              <span style={{ color: '#d98a7a', fontSize: 12 }}>Euclidean</span>
            </div>
            <div className="metric-value">{dE.toFixed(2)}</div>
            <div className="metric-row">
              <div className="metric-swatch" style={{ background: '#7aaf6e' }} />
              <span style={{ color: '#7aaf6e', fontSize: 12 }}>Mahalanobis</span>
            </div>
            <div className="metric-value">{dM.toFixed(2)}</div>
            <div style={{ borderTop: '1px solid #4a4940', paddingTop: 10 }}>
              <span className="slider-label">RATIO (E/M)</span>
              <div className="ratio-value">{(dE / (dM + 1e-8)).toFixed(2)}×</div>
            </div>
          </div>
          <div className="coord-label">
            Point: ({probe[0].toFixed(1)}, {probe[1].toFixed(1)})
          </div>
        </div>
      </div>

      <div className="calc-breakdown">
        <div className="calc-side">
          <div className="calc-label" style={{ color: '#d98a7a' }}>Euclidean distance</div>
          <div className="calc-step">
            <span className="calc-step-label">Inverse covariance</span>
            <BlockMath math={`\\Sigma_E^{-1} = I = \\begin{pmatrix} 1 & 0 \\\\ 0 & 1 \\end{pmatrix}`} />
          </div>
          <div className="calc-step">
            <span className="calc-step-label">Quadratic form</span>
            <BlockMath math={`\\mathbf{x}^\\top I \\, \\mathbf{x} = x^2 + y^2`} />
            <BlockMath math={`= ${probe[0].toFixed(2)}^2 + ${probe[1].toFixed(2)}^2 = ${(probe[0]*probe[0] + probe[1]*probe[1]).toFixed(3)}`} />
          </div>
          <div className="calc-step">
            <span className="calc-step-label">Distance</span>
            <BlockMath math={`d_E = \\sqrt{${(probe[0]*probe[0] + probe[1]*probe[1]).toFixed(3)}} = ${dE.toFixed(3)}`} />
          </div>
        </div>

        <div className="calc-side">
          <div className="calc-label" style={{ color: '#7aaf6e' }}>Mahalanobis distance</div>
          <div className="calc-step">
            <span className="calc-step-label">Inverse covariance</span>
            <BlockMath math={`\\Sigma^{-1} = \\begin{pmatrix} ${sigmaInv[0][0].toFixed(3)} & ${sigmaInv[0][1].toFixed(3)} \\\\ ${sigmaInv[1][0].toFixed(3)} & ${sigmaInv[1][1].toFixed(3)} \\end{pmatrix}`} />
          </div>
          <div className="calc-step">
            <span className="calc-step-label">Matrix-vector product</span>
            <BlockMath math={`\\Sigma^{-1} \\mathbf{x} = \\begin{pmatrix} ${sigmaInv[0][0].toFixed(3)} & ${sigmaInv[0][1].toFixed(3)} \\\\ ${sigmaInv[1][0].toFixed(3)} & ${sigmaInv[1][1].toFixed(3)} \\end{pmatrix} \\begin{pmatrix} ${probe[0].toFixed(2)} \\\\ ${probe[1].toFixed(2)} \\end{pmatrix} = \\begin{pmatrix} ${(sigmaInv[0][0]*probe[0] + sigmaInv[0][1]*probe[1]).toFixed(3)} \\\\ ${(sigmaInv[1][0]*probe[0] + sigmaInv[1][1]*probe[1]).toFixed(3)} \\end{pmatrix}`} />
          </div>
          <div className="calc-step">
            <span className="calc-step-label">Quadratic form</span>
            {(() => {
              const t0 = sigmaInv[0][0]*probe[0] + sigmaInv[0][1]*probe[1]
              const t1 = sigmaInv[1][0]*probe[0] + sigmaInv[1][1]*probe[1]
              const qf = probe[0]*t0 + probe[1]*t1
              return (
                <>
                  <BlockMath math={`\\mathbf{x}^\\top (\\Sigma^{-1} \\mathbf{x}) = (${probe[0].toFixed(2)})(${t0.toFixed(3)}) + (${probe[1].toFixed(2)})(${t1.toFixed(3)}) = ${qf.toFixed(3)}`} />
                </>
              )
            })()}
          </div>
          <div className="calc-step">
            <span className="calc-step-label">Distance</span>
            <BlockMath math={`d_M = \\sqrt{${(dM*dM).toFixed(3)}} = ${dM.toFixed(3)}`} />
          </div>
        </div>
      </div>

      <div className="reading-guide">
        <div className="reading-guide-title">Reading the plot</div>
        <div className="guide-items">
          <div className="guide-item">
            <svg width="24" height="24" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="6" fill="#e8e4d9" stroke="#c49a6c" strokeWidth="2" />
            </svg>
            <span>
              <strong>Draggable test point.</strong> Place it anywhere to compare
              distances from the origin. Drag along the correlation axis vs.
              perpendicular to it and watch the ratio shift.
            </span>
          </div>
          <div className="guide-item">
            <svg width="24" height="14" viewBox="0 0 24 14">
              <circle cx="12" cy="7" r="11" fill="none" stroke="#d98a7a" strokeWidth="1.5" strokeDasharray="4,3" />
            </svg>
            <span>
              <strong style={{ color: '#d98a7a' }}>Dashed circle</strong> — all
              points at the same Euclidean distance as the test point. Always
              round because Euclidean distance is direction-agnostic.
            </span>
          </div>
          <div className="guide-item">
            <svg width="24" height="14" viewBox="0 0 24 14">
              <ellipse cx="12" cy="7" rx="11" ry="6" fill="none" stroke="#7aaf6e" strokeWidth="2" />
            </svg>
            <span>
              <strong style={{ color: '#7aaf6e' }}>Solid ellipse</strong> — all
              points at the same Mahalanobis distance as the test point. It
              stretches along the data's natural spread, so moving along the
              correlation axis registers as "closer" than Euclidean suggests.
            </span>
          </div>
        </div>
      </div>

      <div className="data-note">
        <div className="data-note-title">When do the two distances agree?</div>
        <p>
          The <strong style={{ color: '#d98a7a' }}>Euclidean</strong> locus
          at distance <InlineMath math="r" /> from the origin is a circle:
        </p>
        <BlockMath math="x^2 + y^2 = r^2" />
        <p>
          The <strong style={{ color: '#7aaf6e' }}>Mahalanobis</strong> locus
          at distance <InlineMath math="r" /> is an ellipse defined by the
          quadratic form:
        </p>
        <BlockMath math="\mathbf{x}^\top \Sigma^{-1} \mathbf{x} = r^2" />
        <p>
          Expanding with the inverse covariance for our 2D case:
        </p>
        <BlockMath math={`\\frac{1}{1-\\rho^2} \\left( \\frac{x^2}{\\sigma_x^2} - \\frac{2\\rho \\, x y}{\\sigma_x \\sigma_y} + \\frac{y^2}{\\sigma_y^2} \\right) = r^2`} />
        <p>
          The two distances are <strong style={{ color: '#d4a56a' }}>equal</strong> along
          directions where the circle and ellipse intersect — i.e., directions <InlineMath math="\mathbf{u} = (\cos\theta, \sin\theta)" /> satisfying:
        </p>
        <BlockMath math="\mathbf{u}^\top \Sigma^{-1} \mathbf{u} = 1" />
        <p>
          These are the directions where the covariance-weighted norm happens to
          equal the ordinary norm. For <InlineMath math="\Sigma = I" /> this
          holds everywhere — the ellipse collapses to a circle.
        </p>
        <p>
          The <strong style={{ color: '#d4a56a' }}>largest disagreement</strong> occurs
          along the eigenvectors of <InlineMath math="\Sigma" />. Decompose <InlineMath math="\Sigma = V \Lambda V^\top" /> where <InlineMath math="\Lambda = \text{diag}(\lambda_1, \lambda_2)" /> with <InlineMath math="\lambda_1 \geq \lambda_2" />:
        </p>
        <ul className="eigen-list">
          <li>
            Along the <strong>major eigenvector</strong> (direction
            of <InlineMath math="\lambda_1" />, largest variance): the
            Mahalanobis ellipse extends furthest. Here <InlineMath math="d_M = d_E / \sqrt{\lambda_1}" />,
            so <InlineMath math="d_M < d_E" /> — the point is "closer" than
            Euclidean says because the data naturally spreads in this direction.
          </li>
          <li>
            Along the <strong>minor eigenvector</strong> (direction
            of <InlineMath math="\lambda_2" />, smallest variance): the ellipse
            is narrowest. Here <InlineMath math="d_M = d_E / \sqrt{\lambda_2}" />,
            so <InlineMath math="d_M > d_E" /> — the point is "further" because
            deviations in this direction are unusual for the data.
          </li>
        </ul>
        <p>
          The ratio <InlineMath math="d_E / d_M" /> ranges
          from <InlineMath math="\sqrt{\lambda_2}" /> to <InlineMath math="\sqrt{\lambda_1}" /> depending
          on direction. The condition number <InlineMath math="\sqrt{\lambda_1 / \lambda_2}" /> measures
          how anisotropic the data is — and how much the two distances can disagree.
        </p>
      </div>
    </div>
  )
}

const APP_ITEMS = [
  {
    title: 'Multivariate hypothesis testing',
    content: (
      <p>
        Hotelling's <InlineMath math="T^2" /> test
        is <InlineMath math="n \cdot d_M^2(\bar{\mathbf{x}}, \boldsymbol{\mu}_0)" />.
        It generalizes the univariate <InlineMath math="t" />-test to
        multiple dimensions by measuring how far the sample mean is from
        a hypothesized mean in covariance-adjusted units.
      </p>
    ),
  },
  {
    title: 'Classification',
    content: (
      <p>
        In quadratic discriminant analysis (QDA), a new
        point <InlineMath math="\mathbf{x}" /> is assigned to the class
        with the smallest Mahalanobis distance to the class
        mean: <InlineMath math="\arg\min_k \, d_M(\mathbf{x}, \boldsymbol{\mu}_k; \Sigma_k)" />.
        Each class gets its own covariance, so the decision boundaries are
        quadratic curves rather than lines.
      </p>
    ),
  },
  {
    title: 'Shrinkage estimation of correlated means',
    content: (
      <>
        <p>
          The James-Stein estimator shrinks the sample mean toward a
          target <InlineMath math="\boldsymbol{\mu}_0" />. For correlated
          data with known covariance <InlineMath math="\Sigma" />, the
          estimator is:
        </p>
        <BlockMath math="\hat{\boldsymbol{\mu}}_{JS} = \boldsymbol{\mu}_0 + \left(1 - \frac{p - 2}{n \cdot d_M^2(\bar{\mathbf{x}}, \boldsymbol{\mu}_0)}\right)(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)" />
        <p>
          where <InlineMath math="d_M^2 = n(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^\top \Sigma^{-1}(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)" />.
          The shrinkage
          factor <InlineMath math="(p-2) / (n \cdot d_M^2)" /> is
          inversely proportional to the squared Mahalanobis distance:
          when the sample mean is far from the target in
          covariance-adjusted units, shrinkage is mild; when it is close,
          shrinkage is aggressive. Using Euclidean distance here would
          ignore correlations — over-shrinking in high-variance directions
          and under-shrinking in low-variance ones. This matters in
          portfolio optimization where expected return estimates are notoriously
          noisy and assets are correlated.
        </p>
      </>
    ),
  },
  {
    title: 'Portfolio risk',
    content: (
      <p>
        Given a vector of asset returns with covariance <InlineMath math="\Sigma" />,
        the Mahalanobis distance of a return observation from its mean is
        a measure of how unusual that market day was — accounting for
        cross-asset correlations. Spikes
        in <InlineMath math="d_M" /> flag regime changes or stress events
        that a per-asset z-score would miss.
      </p>
    ),
  },
]

function AppCarousel() {
  const [open, setOpen] = useState<number | null>(null)

  return (
    <div className="app-section">
      <h3 className="app-heading">Other uses</h3>
      <div className="app-list">
        {APP_ITEMS.map((item, i) => (
          <div key={i} className={`app-item ${open === i ? 'app-item-open' : ''}`}>
            <button
              className="app-item-header"
              onClick={() => setOpen(open === i ? null : i)}
            >
              <span className="app-item-title">{item.title}</span>
              <span className="app-item-chevron">{open === i ? '−' : '+'}</span>
            </button>
            {open === i && (
              <div className="app-item-body">
                {item.content}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function SceneApplications() {
  const sx = 2.2, sy = 1.0, corr = 0.7
  const sigma: Mat2 = [[sx * sx, corr * sx * sy], [corr * sx * sy, sy * sy]]
  const sigmaInv = matInv2x2(sigma)
  const data = generateCorrelatedData(150, sx, sy, corr, 77)
  const threshold = 2.5

  const ellBound = ellipsePoints(sigmaInv, threshold, 100)
  const ellPath = ellBound.map((p, i) => {
    const [ex, ey] = toSvg(p)
    return (i === 0 ? 'M' : 'L') + ex + ',' + ey
  }).join(' ') + 'Z'

  const classified = data.map(p => ({
    p, dM: mahalanobis(p[0], p[1], sigmaInv),
  }))
  const mOutliers = classified.filter(c => c.dM > threshold).length
  const eRadius = threshold * Math.sqrt((sx * sx + sy * sy) / 2)

  return (
    <div>
      <div className="app-section">
        <h3 className="app-heading">Outlier detection</h3>
        <p className="scene-desc">
          Under a multivariate normal, the squared Mahalanobis
          distance <InlineMath math="d_M^2" /> follows
          a <InlineMath math="\chi^2" /> distribution
          with <InlineMath math="p" /> degrees of freedom (where <InlineMath math="p" /> is
          the dimensionality). This gives a principled threshold: flag a
          point as an outlier if
        </p>
        <BlockMath math="d_M^2(\mathbf{x}) > \chi^2_{p, \, 1-\alpha}" />
        <p className="scene-desc">
          For <InlineMath math="p = 2" /> and <InlineMath math="\alpha = 0.01" />,
          the critical value is <InlineMath math="\chi^2_{2,\,0.99} \approx 9.21" />,
          i.e. <InlineMath math="d_M \approx 3.03" />. A threshold
          of <InlineMath math="2.5" /> corresponds roughly
          to <InlineMath math="\alpha \approx 0.04" />.
        </p>
        <p className="scene-desc">
          The key advantage over Euclidean: the boundary is an <strong>ellipse
          aligned with the data</strong>, not a circle. A circular boundary
          either misses outliers along the narrow axis or falsely flags
          inliers along the spread.
        </p>
        <div className="scene-row">
          <svg width={W} height={H} className="scene-svg">
            <Grid />
            <circle cx={CENTER_X} cy={CENTER_Y} r={eRadius * SCALE}
              fill="none" stroke="#d98a7a" strokeWidth={1} strokeDasharray="4,4" opacity={0.35} />
            <path d={ellPath} fill="#7aaf6e" fillOpacity={0.04}
              stroke="#7aaf6e" strokeWidth={2} opacity={0.7} />
            {classified.map((c, i) => {
              const [dx, dy] = toSvg(c.p)
              const isOut = c.dM > threshold
              return <circle key={i} cx={dx} cy={dy}
                r={isOut ? 4 : 2.5} fill={isOut ? '#d98a7a' : '#c49a6c'}
                opacity={isOut ? 0.9 : 0.35}
                stroke={isOut ? '#d98a7a' : 'none'} strokeWidth={1} />
            })}
            <circle cx={CENTER_X} cy={CENTER_Y} r={3} fill="#e8e4d9" opacity={0.5} />
          </svg>

          <div className="info-card" style={{ minWidth: 150 }}>
            <span className="slider-label">THRESHOLD</span>
            <div style={{ color: '#7aaf6e', fontSize: 20, fontWeight: 700, marginBottom: 16 }}>{threshold.toFixed(1)}σ</div>
            <span className="slider-label">OUTLIERS DETECTED</span>
            <div style={{ color: '#d98a7a', fontSize: 20, fontWeight: 700, marginBottom: 16 }}>
              {mOutliers} / {data.length}
            </div>
            <div style={{ borderTop: '1px solid #4a4940', paddingTop: 12, color: '#9b9688', fontSize: 11, lineHeight: 1.6 }}>
              <span style={{ color: '#7aaf6e' }}>━</span> Mahalanobis boundary<br />
              <span style={{ color: '#d98a7a' }}>╌</span> Euclidean boundary<br />
              <span style={{ color: '#d98a7a' }}>●</span> Flagged outliers
            </div>
          </div>
        </div>
      </div>

      <AppCarousel />
    </div>
  )
}

function SceneDefinition() {
  const steps = [
    { step: '1', title: 'Center the data', formula: '(x − μ)', desc: 'Shift so the mean is at the origin. Now we\'re measuring deviation from center.', color: '#c49a6c' },
    { step: '2', title: 'Invert the covariance', formula: 'Σ⁻¹', desc: 'The inverse covariance \'whitens\' the space — it undoes correlations and normalizes variances. Directions with high variance get compressed; low variance gets amplified.', color: '#7aaf6e' },
    { step: '3', title: 'Quadratic form', formula: '(x−μ)ᵀ Σ⁻¹ (x−μ)', desc: 'This is a weighted inner product. It\'s like computing the squared Euclidean distance, but in the whitened space where all features are uncorrelated with unit variance.', color: '#d4a56a' },
    { step: '4', title: 'Square root', formula: '√(·)', desc: 'Take the root to get an actual distance (not squared distance). The result is unitless and scale-invariant.', color: '#d98a7a' },
  ]

  return (
    <div>
      <div className="data-note" style={{ marginBottom: 24 }}>
        <div className="data-note-title">Formal definition</div>
        <p>
          The Mahalanobis distance, introduced by P. C. Mahalanobis in 1936,
          measures the distance between a point <InlineMath math="\mathbf{x} \in \mathbb{R}^p" /> and
          a distribution with mean <InlineMath math="\boldsymbol{\mu}" /> and
          covariance <InlineMath math="\Sigma" />:
        </p>
        <BlockMath math="d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}" />
        <p>
          More generally, the Mahalanobis distance between two
          points <InlineMath math="\mathbf{x}, \mathbf{y} \in \mathbb{R}^p" /> with
          respect to a positive-definite matrix <InlineMath math="\Sigma" /> is:
        </p>
        <BlockMath math="d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^\top \Sigma^{-1} (\mathbf{x} - \mathbf{y})}" />
        <p>where:</p>
        <ul className="eigen-list">
          <li>
            <InlineMath math="\mathbf{x}, \mathbf{y} \in \mathbb{R}^p" /> are
            observation vectors
          </li>
          <li>
            <InlineMath math="\boldsymbol{\mu} \in \mathbb{R}^p" /> is the
            mean of the distribution
          </li>
          <li>
            <InlineMath math="\Sigma \in \mathbb{R}^{p \times p}" /> is the
            covariance matrix (positive definite)
          </li>
          <li>
            <InlineMath math="\Sigma^{-1}" /> is the precision matrix (inverse covariance)
          </li>
        </ul>
      </div>

      <div className="data-note" style={{ marginBottom: 24 }}>
        <div className="data-note-title">Key properties</div>
        <ul className="eigen-list">
          <li>
            <strong>Multivariate z-score.</strong> In one
            dimension, <InlineMath math="d_M = |x - \mu| / \sigma" />, which
            is the absolute z-score. The Mahalanobis distance generalizes this
            to <InlineMath math="p" /> dimensions, accounting for all
            pairwise correlations.
          </li>
          <li>
            <strong>Unitless and scale-invariant.</strong> Because <InlineMath math="\Sigma^{-1}" /> normalizes
            by variance, the result does not depend on the units of measurement.
            Scaling any variable leaves <InlineMath math="d_M" /> unchanged.
          </li>
          <li>
            <strong>Euclidean distance after whitening.</strong> Let <InlineMath math="\Sigma = L L^\top" /> (Cholesky).
            Then <InlineMath math="d_M(\mathbf{x}, \boldsymbol{\mu}) = \| L^{-1}(\mathbf{x} - \boldsymbol{\mu}) \|_2" />.
            The Mahalanobis distance is the Euclidean distance in the
            whitened coordinate system where all variables are uncorrelated
            with unit variance.
          </li>
          <li>
            <strong>Affine invariant.</strong> For any invertible
            matrix <InlineMath math="A" /> and
            vector <InlineMath math="\mathbf{b}" />,
            the Mahalanobis distance of <InlineMath math="A\mathbf{x} + \mathbf{b}" /> under
            the transformed covariance <InlineMath math="A \Sigma A^\top" /> equals
            the original <InlineMath math="d_M(\mathbf{x}, \boldsymbol{\mu})" />.
          </li>
          <li>
            <strong>Chi-squared connection.</strong> If <InlineMath math="\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)" />,
            then <InlineMath math="d_M^2 \sim \chi^2_p" />. This gives
            principled probability thresholds for outlier detection.
          </li>
        </ul>
      </div>

      <div className="data-note-title" style={{ marginBottom: 14 }}>Step-by-step construction</div>
      <p className="scene-desc">
        How the formula transforms a naive distance into a covariance-aware one.
      </p>
      <div className="formula-steps">
        {steps.map(s => (
          <div key={s.step} className="formula-step" style={{ borderLeftColor: s.color }}>
            <div className="step-number" style={{ background: s.color + '22', color: s.color }}>
              {s.step}
            </div>
            <div>
              <div className="step-title">{s.title}</div>
              <div className="step-formula" style={{ color: s.color }}>{s.formula}</div>
              <div className="step-desc">{s.desc}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="special-case">
        <div className="slider-label" style={{ marginBottom: 8, letterSpacing: 1 }}>SPECIAL CASE</div>
        <div style={{ color: '#e8e4d9', fontSize: 14, lineHeight: 1.6 }}>
          When Σ = I (identity matrix), Mahalanobis distance{' '}
          <strong style={{ color: '#d4a56a' }}>equals</strong> Euclidean distance.
          <br />
          <span style={{ color: '#9b9688', fontSize: 12 }}>
            Euclidean is just Mahalanobis with the assumption that all features
            are independent with equal variance.
          </span>
        </div>
      </div>
    </div>
  )
}

const TABS = ['Definition', 'Euclidean vs Mahalanobis', 'Applications']

export default function MahalanobisDistance() {
  const [tab, setTab] = useState(0)

  return (
    <div className="topic-page mahal">
      <Link to="/" className="back-link">← back</Link>
      <h1 className="mahal-title">Mahalanobis Distance</h1>
      <p className="mahal-subtitle">Distance that knows your data's shape</p>
      <div className="tab-bar">
        {TABS.map((t, i) => (
          <button key={i} onClick={() => setTab(i)}
            className={`tab-btn ${tab === i ? 'active' : ''}`}>
            {t}
          </button>
        ))}
      </div>
      {tab === 0 && <SceneDefinition />}
      {tab === 1 && <SceneCompare />}
      {tab === 2 && <SceneApplications />}
    </div>
  )
}
