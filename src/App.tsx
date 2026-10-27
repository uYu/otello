import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, BookOpen, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Download, Home, Lightbulb, Pause, Play, Plus, RotateCcw, Upload, X, Cpu, Users, Eye, Check, ArrowUpRight, Leaf, Undo2 } from 'lucide-react';
import { Board } from './components/Board.tsx';
import { classicPosition, INAUGURAL_FINAL } from './game/classic-games.ts';
import { Dialog } from './components/Dialog.tsx';
import { initial, legal, score } from './game/engine.ts';
import { downloadSave, history, MAX_SAVE_BYTES, parseSave, SAVE_KEY } from './game/storage.ts';
import { nativeEngineReady, requestNativeAnalysis } from './game/remote.ts';
import { colorName, coordinate, LEVELS, MODES } from './game/types.ts';
import type { Analysis, Color, Difficulty, Engine, Mode, Save } from './game/types.ts';

function load() {
  try { const text = localStorage.getItem(SAVE_KEY); return { save: text ? parseSave(text) : null, error: '' }; }
  catch { return { save: null, error: '本地存档无法读取。可以开始新局，也可以导入棋谱。' }; }
}
function BrandMark() { return <span className="brand-mark" aria-hidden="true"><i /><i /><i /><i /></span>; }
function DemoBoard() {
  const position = useMemo(() => classicPosition(), []);
  return <figure className="hero-scene" aria-label={`${INAUGURAL_FINAL.title}，${INAUGURAL_FINAL.black}对${INAUGURAL_FINAL.white}，第${INAUGURAL_FINAL.displayPly}手后的真实局面`}>
    <div className="hero-orbit" aria-hidden="true" /><div className="hero-board" aria-hidden="true"><Board position={position} decorative showLegal={false} /></div>
    <figcaption className="hero-caption"><span className="tiny-discs" aria-hidden="true"><i /><i /></span><span><a className="hero-game-source" href={INAUGURAL_FINAL.source} target="_blank" rel="noreferrer" title="查看世界黑白棋联合会的比赛记载">{INAUGURAL_FINAL.title}<ArrowUpRight size={13} /></a><small>{INAUGURAL_FINAL.black} × {INAUGURAL_FINAL.white} · 第 {INAUGURAL_FINAL.displayPly} 手</small></span></figcaption>
    <span className="hero-note" aria-hidden="true">东京 · 1977</span>
  </figure>;
}
export default function App() {
  const [loaded] = useState(load);
  const [save, setSave] = useState<Save | null>(loaded.save);
  const [screen, setScreen] = useState<'menu' | 'game'>('menu');
  const [modal, setModal] = useState<'new' | 'rules' | 'pause' | 'result' | null>(null);
  const [mode, setMode] = useState<Mode>('ai');
  const [difficulty, setDifficulty] = useState<Difficulty>('normal');
  const [engine, setEngine] = useState<Engine>('builtin');
  const [nativeReady, setNativeReady] = useState(false);
  const [nativeCheck, setNativeCheck] = useState(0);
  const choosingEngine = modal === 'new' || modal === 'pause';
  const [human, setHuman] = useState<Color>(1);
  const [showLegal, setShowLegal] = useState(true);
  const [error, setError] = useState(loaded.error);
  const [aiError, setAIError] = useState('');
  const [thinking, setThinking] = useState(false);
  const [task, setTask] = useState<'move' | 'hint' | null>(null);
  const [hint, setHint] = useState<number | null>(null);
  const [stats, setStats] = useState<Analysis | null>(null);
  const [replay, setReplay] = useState<number | null>(null);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [watchPaused, setWatchPaused] = useState(false);
  const [retry, setRetry] = useState(0);
  const file = useRef<HTMLInputElement>(null);
  const entries = useMemo(() => save ? history(save) : [], [save]);
  const finalPosition = entries.at(-1)?.after ?? initial();
  const shown = replay ?? entries.length;
  const position = shown ? entries[shown - 1].after : initial();
  const counts = score(position), moves = legal(position);
  const isReplay = replay !== null;
  const humanTurn = !!save && (save.mode === 'local' || save.mode === 'ai' && position.turn === save.human);
  const active = screen === 'game' && !modal && !isReplay && !position.over;
  const interactive = active && humanTurn && !thinking;
  const automatic = !!save && active && !humanTurn && !(save.mode === 'watch' && watchPaused);
  const commit = (move: number) => {
    if (!save || !interactive || !moves.includes(move)) return;
    setSave({ ...save, actions: [...save.actions, move] }); setHint(null); setStats(null); setAIError('');
  };
  useEffect(() => {
    const controller = new AbortController();
    nativeEngineReady(controller.signal).then(ready => {
      if (!controller.signal.aborted) { setNativeReady(ready); if (ready) setEngine('egaroucid'); }
    });
    return () => controller.abort();
  }, [nativeCheck, choosingEngine]);
  useEffect(() => {
    if (!save) return;
    try { localStorage.setItem(SAVE_KEY, JSON.stringify(save)); }
    catch { setError('自动保存失败，请导出棋谱备份。'); }
  }, [save]);
  useEffect(() => { setHint(null); setTask(null); setAIError(''); }, [save, screen, replay]);
  useEffect(() => { setStats(null); }, [screen, replay]);
  useEffect(() => { setStats(null); }, [save?.engine]);
  useEffect(() => {
    const hintTask = task === 'hint' && active && humanTurn;
    if ((!automatic && !hintTask) || !save) { setThinking(false); return; }
    setThinking(true); setAIError('');
    const useNative = save.engine === 'egaroucid';
    const worker = useNative ? null : new Worker(new URL('./game/ai.worker.ts', import.meta.url), { type: 'module' });
    const controller = new AbortController();
    let cancelled = false;
    const finishError = (message: string) => { if (cancelled) return; setThinking(false); setAIError(message); setTask(null); worker?.terminate(); };
    const acceptResult = (result?: Analysis, message?: string) => {
      if (cancelled) return;
      if (!result || !moves.includes(result.move)) { finishError(message ?? 'AI 返回了非法落点'); return; }
      setStats(result); setThinking(false); setTask(null);
      if (hintTask) setHint(result.move);
      else setSave(previous => previous === save ? { ...save, actions: [...save.actions, result.move] } : previous);
      worker?.terminate();
    };
    if (worker) {
      worker.onmessage = (event: MessageEvent<{ result?: Analysis; error?: string }>) => acceptResult(event.data.result, event.data.error);
      worker.onerror = () => finishError('AI 引擎运行失败，请重试。');
    }
    const timer = setTimeout(() => {
      if (useNative) requestNativeAnalysis(position, hintTask ? 'hard' : save.difficulty, controller.signal).then(result => acceptResult(result)).catch(e => finishError(e instanceof Error ? e.message : '本机引擎调用失败'));
      else worker?.postMessage({ position, difficulty: hintTask ? 'hard' : save.difficulty, seed: (save.actions.length + 1) * 7919 });
    }, save.mode === 'watch' ? 700 : 280);
    return () => { cancelled = true; clearTimeout(timer); controller.abort(); worker?.terminate(); };
    // Position and moves are derived solely from save/replay; keep the worker stable during UI renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [save, automatic, task, active, humanTurn, retry]);
  useEffect(() => {
    if (screen === 'game' && finalPosition.over && replay === null) setModal('result');
  }, [save, finalPosition.over, screen, replay]);
  useEffect(() => {
    if (!replayPlaying || replay === null || modal || screen !== 'game') return;
    if (replay >= entries.length) { setReplayPlaying(false); return; }
    const timer = setTimeout(() => setReplay(i => (i ?? 0) + 1), 850);
    return () => clearTimeout(timer);
  }, [replayPlaying, replay, entries.length, modal, screen]);
  const start = () => {
    setSave({ version: 1, mode, difficulty, human, engine: mode === 'local' ? 'builtin' : engine, actions: [], createdAt: new Date().toISOString() });
    setReplay(null); setReplayPlaying(false); setWatchPaused(false); setModal(null); setScreen('game'); setError('');
  };
  const undo = () => {
    if (!save || isReplay || !save.actions.length) return;
    let index = save.actions.length - 1;
    if (save.mode === 'ai') { while (index >= 0 && entries[index].color !== save.human) index--; }
    if (index < 0) return;
    setWatchPaused(true); setSave({ ...save, actions: save.actions.slice(0, index) });
  };
  const enterReplay = (index: number) => { setTask(null); setHint(null); setReplay(Math.max(0, Math.min(entries.length, index))); setReplayPlaying(false); };
  const aiName = save?.engine === 'egaroucid' ? 'Egaroucid' : `${LEVELS[save?.difficulty ?? 'normal']} AI`;
  const who = (color: Color) => !save ? '' : save.mode === 'local' ? (color === 1 ? '玩家一' : '玩家二') : save.mode === 'watch' ? aiName : color === save.human ? '你' : aiName;
  const canUndo = !!save && !!entries.length && (save.mode !== 'ai' || entries.some(e => e.color === save.human));
  const title = position.over ? counts.black === counts.white ? '棋逢对手，平分秋色' : `${counts.black > counts.white ? '黑棋' : '白棋'}获胜` : isReplay ? '重温每一个转折' : thinking ? task === 'hint' ? '正在寻找好棋…' : 'AI 正在思考…' : `${colorName(position.turn)}行棋`;
  return <div className="app-shell">
    <header className="topbar"><button className="brand" onClick={() => { setScreen('menu'); setModal(null); }} aria-label="黑白之间，返回首页"><BrandMark /><span>黑白之间<small>OTHELLO ATELIER</small></span></button>
      <nav>{screen === 'game' && <button className="text-btn home-link" onClick={() => setScreen('menu')}><Home size={16} />首页</button>}<button className="text-btn" onClick={() => setModal('rules')}><BookOpen size={16} />游戏规则</button><span className="nav-divider" /><span className="edition">经典黑白棋</span></nav>
    </header>
    {error && <div className="notice" role="alert"><span>{error}</span><button aria-label="关闭提示" onClick={() => setError('')}><X size={16} /></button></div>}
    <input ref={file} type="file" accept=".json,application/json" hidden onChange={async event => {
      const selected = event.target.files?.[0]; event.target.value = ''; if (!selected) return;
      try { if (selected.size > MAX_SAVE_BYTES) throw new Error('棋谱文件过大'); const imported = parseSave(await selected.text()); setSave(imported); setReplay(null); setReplayPlaying(false); setWatchPaused(true); setScreen('game'); setModal(null); setError(''); }
      catch (e) { setError(e instanceof Error ? e.message : '导入失败'); }
    }} />
    {screen === 'menu' ? <>
      <main className="menu"><section className="intro"><div className="eyebrow"><span /> LESS IS MORE. EVERY MOVE MATTERS.</div>
        <h1>黑白之间，<br /><em>自有天地。</em></h1><p className="intro-copy">一枚棋子，八个方向。<br />在翻转与取舍之间，发现简单规则里的深邃。</p>
        <div className="menu-actions"><button className="primary large" onClick={() => setModal('new')}>开始新对局<ArrowRight size={18} /></button>{save && <button className="secondary large" onClick={() => { setScreen('game'); setReplay(null); }}>继续对局 <span className="count-badge">{save.actions.length}</span></button>}</div>
        <div className="menu-links"><button onClick={() => file.current?.click()}><Upload size={14} />导入棋谱</button>{save && <button onClick={() => downloadSave(save)}><Download size={14} />导出棋谱</button>}</div>
        <div className="intro-facts"><div><b>8 × 8</b><span>经典棋盘</span></div><div><b>3 档</b><span>离线 AI</span></div><div><b>∞</b><span>对弈的乐趣</span></div></div>
      </section><DemoBoard /></main>
      <div className="mode-preview"><div className="mode-preview-intro"><span className="eyebrow">YOUR TABLE, YOUR PACE</span><h2>选一种方式，坐下来。</h2></div>{(['ai', 'local', 'watch'] as Mode[]).map(m => { const Icon = m === 'ai' ? Cpu : m === 'local' ? Users : Eye; return <button className="mode-preview-card" key={m} onClick={() => { setMode(m); setModal('new'); }}><Icon size={21} strokeWidth={1.4} /><div><h3>{MODES[m]}</h3><p>{m === 'ai' ? '与 AI 切磋，慢慢精进' : m === 'local' ? '与身边的人，面对面对弈' : '暂歇片刻，看好棋发生'}</p></div><ArrowUpRight size={16} /></button>; })}</div>
    </> : <main className="game">
      <div className="game-heading"><div><div className="eyebrow">A QUIET MOMENT. A THOUGHTFUL MOVE.</div><h1>{isReplay ? '棋谱回放' : '棋局进行中'}<span>{save && MODES[save.mode]}</span></h1></div><div className="heading-actions"><button className="text-btn" onClick={() => setModal('new')}><Plus size={16} />新对局</button><button className="icon-btn" aria-label="暂停与棋局菜单" onClick={() => setModal('pause')}><Pause size={18} /></button></div></div>
      <div className="game-layout"><section className="table"><div className="table-status"><span className={`status-disc ${position.turn === 1 ? 'black' : 'white'}`} /><div role="status" aria-live="polite"><strong>{title}</strong><small>{position.over ? '终局以盘面实际棋子数计分' : isReplay ? `第 ${shown} / ${entries.length} 手` : `${who(position.turn)} · ${moves.length} 个合法落点`}</small></div><span className="move-label">MOVE <b>{String(shown).padStart(2, '0')}</b></span></div>
        <Board position={position} interactive={interactive} onMove={commit} showLegal={showLegal && !position.over && (!isReplay || showLegal)} hint={hint} />
        <div className="board-legend"><span><i className="legend-dot" />可落点</span><span><i className="legend-last" />最后落子</span><span className="keyboard-note">方向键移动 · Enter 落子</span></div>
        {position.passed && !position.over && <div className="pass-notice" role="status">{colorName(position.passed)}无合法落点，自动停着。{colorName(position.turn)}继续行棋。</div>}
        {aiError && <div className="ai-error" role="alert">{aiError}<button onClick={() => { setTask(null); setRetry(i => i + 1); }}>重试</button></div>}
        <div className="board-tools"><button className="tool" disabled={!canUndo || isReplay} onClick={undo}><Undo2 size={17} /><span>悔棋</span></button>{save?.mode !== 'watch' && <button className={`tool ${hint !== null ? 'selected' : ''}`} disabled={!interactive || thinking} onClick={() => { if (hint !== null) setHint(null); else setTask('hint'); }}><Lightbulb size={17} /><span>{thinking && task === 'hint' ? '分析中' : '提示'}</span></button>}<label className="legal-toggle"><input type="checkbox" checked={showLegal} onChange={e => setShowLegal(e.target.checked)} /><span>显示可落点</span></label>{save?.mode === 'watch' && !isReplay && <button className="tool" disabled={position.over} onClick={() => setWatchPaused(p => !p)}>{watchPaused ? <Play size={17} /> : <Pause size={17} />}<span>{watchPaused ? '继续' : '暂停'}</span></button>}</div>
      </section><aside className="sidebar"><section className="panel score-panel"><div className="panel-title"><h2>棋局概览</h2><span className="pill">{isReplay ? '回放中' : position.over ? '已结束' : '进行中'}</span></div>
        <div className="players">{([1, -1] as Color[]).map(c => <div key={c} className={`player ${!position.over && position.turn === c ? 'current' : ''}`}><span className={`score-disc ${c === 1 ? 'black' : 'white'}`} /><span className="player-name">{colorName(c)}<small>{who(c)}</small></span><strong>{c === 1 ? counts.black : counts.white}</strong></div>)}</div>
        <div className="score-bar"><i style={{ width: `${counts.black / (counts.black + counts.white) * 100}%` }} /></div><div className="score-caption"><span>黑 {counts.black}</span><span>空位 {64 - counts.black - counts.white}</span><span>白 {counts.white}</span></div>
      </section>
      <section className="panel insight-panel"><div className="panel-title"><h2><Leaf size={15} />一手之间</h2>{thinking && <span className="thinking-dots"><i /><i /><i /></span>}</div><p>{hint !== null ? `试试 ${coordinate(hint)}。提示会综合行动力、角位和前沿子，不只看眼前的翻子数。` : position.over ? counts.black === counts.white ? '这盘棋平分秋色。回放棋谱，找找那些改变走向的瞬间。' : `最终比分 ${counts.black} : ${counts.white}。回到棋谱里，看看局势是从哪一步开始改变的。` : '角位无法被翻转。比起一时的子数，保留选择的余地往往更重要。'}</p>
        {stats && !stats.bookHit && !isReplay && <div className="engine-stats"><span>{stats.exact ? '精确终局' : `${stats.depth} 层搜索`}</span><span>{stats.nodes.toLocaleString()} 节点</span>{stats.ttHits !== null && <span>缓存命中 {stats.ttHits.toLocaleString()}</span>}{stats.engine === 'egaroucid' && <span>Egaroucid</span>}<span>{(stats.elapsed / 1000).toFixed(2)}s</span></div>}
      </section>
      <section className="panel history-panel"><div className="panel-title"><h2>落子记录 <span>{entries.length}</span></h2><button className="icon-btn" aria-label="导出棋谱" onClick={() => save && downloadSave(save)}><Download size={16} /></button></div>
        <div className="history-list">{entries.length === 0 ? <div className="history-empty"><span className="tiny-discs"><i /><i /></span><p>好棋，从第一手开始。</p><small>黑棋先行，点击棋盘上的可落点。</small></div> : entries.map((entry, i) => <button key={i} className={`history-item ${shown === i + 1 ? 'active' : ''}`} onClick={() => enterReplay(i + 1)} aria-label={`回放第${i + 1}手 ${colorName(entry.color)} ${coordinate(entry.move)}`}><span className="history-number">{String(i + 1).padStart(2, '0')}</span><i className={`mini-disc ${entry.color === 1 ? 'black' : 'white'}`} /><b>{coordinate(entry.move)}</b><small>翻 {entry.flipped.length} 子</small>{entry.after.passed && <span className="pass-mark" title={`${colorName(entry.after.passed)}停着`}>停</span>}</button>)}</div>
        <div className="replay-controls"><div className="replay-buttons"><button className="icon-btn" disabled={!entries.length} aria-label="回到开局" onClick={() => enterReplay(0)}><ChevronsLeft size={17} /></button><button className="icon-btn" disabled={!entries.length || shown === 0} aria-label="上一步" onClick={() => enterReplay(shown - 1)}><ChevronLeft size={17} /></button><button className="icon-btn playback" disabled={!entries.length} aria-label={replayPlaying ? '暂停回放' : '播放棋谱'} onClick={() => { if (replay === null || shown === entries.length) setReplay(0); setReplayPlaying(p => !p); }}>{replayPlaying ? <Pause size={16} /> : <Play size={16} />}</button><button className="icon-btn" disabled={!entries.length || shown >= entries.length} aria-label="下一步" onClick={() => enterReplay(shown + 1)}><ChevronRight size={17} /></button><button className="icon-btn" disabled={!entries.length} aria-label="最后一步" onClick={() => enterReplay(entries.length)}><ChevronsRight size={17} /></button></div><input aria-label="棋谱进度" type="range" min="0" max={entries.length} value={shown} disabled={!entries.length} onChange={e => enterReplay(Number(e.target.value))} />{isReplay && <button className="return-live" onClick={() => { setReplay(null); setReplayPlaying(false); }}>退出回放，返回棋局 <ArrowRight size={13} /></button>}</div>
      </section></aside></div>
    </main>}
    <footer><span>落子有声，思考无界。</span><span><i className="online-dot" /> {screen === 'game' && save?.engine === 'egaroucid' ? 'Egaroucid 原生引擎' : '本地 C++ 引擎'}{screen === 'game' && !isReplay && stats?.bookHit && <span className="book-hit-indicator" role="img" title="开局库命中" aria-label="开局库命中">·</span>} <span className="footer-separator">/</span> 无需联网</span><span>OTHELLO ATELIER · 2026</span></footer>
    {modal === 'new' && <Dialog title="开启一局好棋" onClose={() => setModal(null)}><p className="dialog-copy">不必匆忙，给下一步留一点时间。</p><label className="field-label">对局方式</label><div className="mode-options">{(['ai', 'local', 'watch'] as Mode[]).map(m => { const Icon = m === 'ai' ? Cpu : m === 'local' ? Users : Eye; return <button key={m} className={`choice mode-choice ${mode === m ? 'chosen' : ''}`} onClick={() => setMode(m)}><Icon size={21} /><b>{MODES[m]}</b><small>{m === 'ai' ? '挑战电脑' : m === 'local' ? '同屏轮流' : '欣赏对弈'}</small>{mode === m && <Check className="choice-check" size={13} />}</button>; })}</div>
      {mode !== 'local' && <><label className="field-label">对战引擎 <small>{nativeReady ? '本机引擎已连接' : '本机引擎未连接'}</small></label><div className="engine-options"><button className={`choice ${engine === 'egaroucid' ? 'chosen' : ''}`} disabled={!nativeReady} onClick={() => setEngine('egaroucid')}><b>Egaroucid</b><small>原生引擎 · 开局库已启用</small></button><button className={`choice ${engine === 'builtin' ? 'chosen' : ''}`} onClick={() => setEngine('builtin')}><b>内置 AI</b><small>浏览器离线运行</small></button></div>{!nativeReady && <button className="native-reconnect" onClick={() => setNativeCheck(n => n + 1)}>重新连接本机引擎</button>}<label className="field-label">AI 难度</label><div className="difficulty-options">{(['easy', 'normal', 'hard'] as Difficulty[]).map((d, i) => <button key={d} className={`choice ${difficulty === d ? 'chosen' : ''}`} onClick={() => setDifficulty(d)}><span className="level-bars">{[0, 1, 2].map(n => <i className={n <= i ? 'filled' : ''} key={n} />)}</span><b>{LEVELS[d]}</b><small>{i === 0 ? '轻松入门' : i === 1 ? '有来有回' : '深思熟虑'}</small></button>)}</div><p className="field-help">{engine === 'egaroucid' ? `Egaroucid 等级 ${{ easy: 1, normal: 5, hard: 10 }[difficulty]} · 优先查询开局库，出库后原生搜索。` : difficulty === 'hard' ? 'C++ 迭代加深搜索，每手约 1.5 秒，残局可搜索至终局。' : difficulty === 'normal' ? '最多前瞻 5 层，每手约 0.45 秒。' : '带探索性的局面评估，适合熟悉规则。'}</p></>}
      {mode === 'ai' && <><label className="field-label">选择执子 <small>黑棋先行</small></label><div className="color-options">{([1, -1] as Color[]).map(c => <button key={c} className={`choice ${human === c ? 'chosen' : ''}`} onClick={() => setHuman(c)}><i className={`mini-disc ${c === 1 ? 'black' : 'white'}`} />{colorName(c)}{human === c && <Check size={14} />}</button>)}</div></>}
      {save && <p className="replace-note">开始新局会替换当前存档。<button onClick={() => downloadSave(save)}>先导出棋谱</button></p>}<button className="primary dialog-start" onClick={start}>坐下来，开始吧<ArrowRight size={17} /></button>
    </Dialog>}
    {modal === 'rules' && <Dialog title="一分钟，学会黑白棋" onClose={() => setModal(null)}><div className="rules-lead"><span className="tiny-discs"><i /><i /></span><p>规则很简单，好棋值得慢慢想。</p></div><ol className="rules-list"><li><b>黑棋先行，轮流落子</b><p>棋盘为 8 × 8 格。开局中央四子交错摆放，双方轮流下一枚棋子。</p></li><li><b>夹住对方，全部翻转</b><p>新棋子与己方棋子之间，若在横、竖或斜线方向连续夹住对方棋子，必须将这些棋子全部翻成己方颜色。一手可同时翻转多个方向。</p></li><li><b>每一手，必须有所翻转</b><p>不能下在已有棋子的格子，也不能下在无法翻转任何对手棋子的位置。小圆点标出了所有合法落点，悬停可预览将要翻转的棋子。</p></li><li><b>无处可下，自动停着</b><p>一方无合法落点时，自动跳过该方，由对手继续。双方都无合法落点即终局，即使棋盘还没填满。</p></li><li><b>终局子多者胜</b><p>以盘面上实际棋子数量计分，子数相同则平局。中盘领先不代表胜利，角位和行动力尤其重要。</p></li></ol><div className="rules-tip"><Lightbulb size={18} /><p>人机模式的悔棋会退回到你上一次落子之前；点击落子记录即可回放。手机也可以直接点击棋盘操作。</p></div><button className="primary dialog-start" onClick={() => setModal(null)}>明白了，落子吧<ArrowRight size={16} /></button></Dialog>}
    {modal === 'pause' && <Dialog title="让思考歇一会儿" onClose={() => setModal(null)}><p className="dialog-copy">棋局已暂停，进度会自动保存在这个浏览器。</p>{save && save.mode !== 'local' && <><label className="field-label">当前对战引擎</label><div className="engine-options"><button disabled={!nativeReady} className={`choice ${save.engine === 'egaroucid' ? 'chosen' : ''}`} onClick={() => setSave({ ...save, engine: 'egaroucid' })}><b>Egaroucid</b><small>原生引擎 · 开局库已启用</small></button><button className={`choice ${save.engine !== 'egaroucid' ? 'chosen' : ''}`} onClick={() => setSave({ ...save, engine: 'builtin' })}><b>内置 AI</b><small>浏览器离线运行</small></button></div><p className="field-help">切换引擎会保留当前棋谱。</p>{!nativeReady && <button className="native-reconnect" onClick={() => setNativeCheck(n => n + 1)}>重新连接本机引擎</button>}</>}<div className="pause-actions"><button className="primary" onClick={() => setModal(null)}><Play size={17} />继续对局</button><button className="secondary" onClick={() => save && downloadSave(save)}><Download size={17} />导出棋谱</button><button className="secondary" onClick={() => file.current?.click()}><Upload size={17} />导入棋谱</button><button className="secondary" onClick={() => setModal('new')}><RotateCcw size={17} />开启新局</button><button className="text-btn" onClick={() => { setModal(null); setScreen('menu'); }}><Home size={16} />返回首页</button></div></Dialog>}
    {modal === 'result' && <Dialog title="一局终了，余味悠长" onClose={() => setModal(null)}><div className="result"><span className="eyebrow">WELL PLAYED</span><h2>{score(finalPosition).black === score(finalPosition).white ? '平分秋色' : `${score(finalPosition).black > score(finalPosition).white ? '黑棋' : '白棋'}获胜`}</h2><div className="result-score"><span><i className="score-disc black" />{score(finalPosition).black}</span><small>:</small><span><i className="score-disc white" />{score(finalPosition).white}</span></div><p>共 {entries.length} 手。每一次翻转，都是新的可能。</p></div><div className="result-actions"><button className="secondary" onClick={() => { setModal(null); enterReplay(0); }}>回看棋谱</button><button className="primary" onClick={() => setModal('new')}>再来一局<ArrowRight size={16} /></button></div></Dialog>}
  </div>;
}
