import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './theme.css'
import App from './App'

const root = document.getElementById('root')
if (!root) throw new Error('index.html is missing <div id="root">. Restore it and reload.')

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
