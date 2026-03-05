import { Routes, Route } from 'react-router-dom'
import './App.css'
import Home from './Home'
import MahalanobisDistance from './topics/mahalanobis/MahalanobisDistance'

function App() {
  return (
    <div className="layout">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/mahalanobis-distance" element={<MahalanobisDistance />} />
      </Routes>
    </div>
  )
}

export default App
