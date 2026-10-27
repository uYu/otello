import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.tsx";
import "./style.css";
import { initEngine } from './game/engine.ts';
const root = document.getElementById('root')!;
root.innerHTML = '<div class="loading">黑白之间 <span>正在打开棋盒…</span></div>';
initEngine().then(() => ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)).catch(() => { root.textContent = '引擎加载失败，请刷新页面重试。'; });
