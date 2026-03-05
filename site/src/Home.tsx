import { Link } from 'react-router-dom'

const topics = [
  { title: 'Mahalanobis Distance', path: '/mahalanobis-distance', desc: 'Distance that knows your data\'s shape' },
]

function Home() {
  return (
    <>
      <h1 className="site-title">Topics in Quant Research</h1>
      <p className="site-subtitle">
        A collection of useful ideas, not exhaustive but worth knowing.
      </p>
      <ul className="topic-list">
        {topics.map(t => (
          <li key={t.path}>
            <Link to={t.path}>{t.title}</Link>
            <span className="topic-desc"> — {t.desc}</span>
          </li>
        ))}
      </ul>
    </>
  )
}

export default Home
