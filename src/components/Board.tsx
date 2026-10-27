import { useRef, useState } from 'react';
import { flips, legal } from '../game/engine.ts';
import { colorName, coordinate } from '../game/types.ts';
import type { Position } from '../game/types.ts';
export function Board({ position, onMove, interactive = false, showLegal = true, hint = null, decorative = false }: {
  position: Position; onMove?: (move: number) => void; interactive?: boolean; showLegal?: boolean; hint?: number | null; decorative?: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [focus, setFocus] = useState(19);
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const moves = legal(position);
  const preview = interactive && hover !== null && moves.includes(hover) ? flips(position, hover) : [];
  const contents = (i: number) => <>
    {!!position.cells[i] && <span key={position.cells[i]} className={`disc ${position.cells[i] === 1 ? 'black' : 'white'} ${preview.includes(i) ? 'will-flip' : ''}`}>
      {position.lastMove === i && <span className="last-dot" />}
    </span>}
    {!position.cells[i] && interactive && hover === i && moves.includes(i) && <span className={`disc ghost ${position.turn === 1 ? 'black' : 'white'}`} />}
    {!position.cells[i] && showLegal && moves.includes(i) && <span className="legal-dot" />}
    {hint === i && <span className="hint-ring" />}
    {[18, 21, 42, 45].includes(i) && <span className="star-point" />}
  </>;
  return <div className={`board-frame ${decorative ? 'decorative' : ''}`}>
    <div className="board-coordinates top" aria-hidden="true">{'ABCDEFGH'.split('').map(c => <span key={c}>{c}</span>)}</div>
    <div className="board-coordinates left" aria-hidden="true">{Array.from({ length: 8 }, (_, i) => <span key={i}>{i + 1}</span>)}</div>
    <div className="board-grid" role={decorative ? undefined : 'grid'} aria-label={decorative ? undefined : '黑白棋棋盘，方向键移动，回车落子'}>
      {Array.from({ length: 8 }, (_, y) => <div className="board-row" role={decorative ? undefined : 'row'} key={y}>
        {Array.from({ length: 8 }, (_, x) => {
          const i = y * 8 + x;
          return decorative ? <div key={i} className="cell">{contents(i)}</div> :
            <button key={i} ref={el => { refs.current[i] = el; }} role="gridcell" tabIndex={focus === i ? 0 : -1}
              aria-label={`${coordinate(i)}，${position.cells[i] ? colorName(position.cells[i] as 1 | -1) : moves.includes(i) ? '可落子' : '空位'}${hint === i ? '，推荐落点' : ''}`}
              aria-disabled={!interactive || !moves.includes(i)} className={`cell ${interactive && moves.includes(i) ? 'playable' : ''} ${hint === i ? 'recommended' : ''}`}
              onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} onFocus={() => { setFocus(i); setHover(i); }} onBlur={() => setHover(null)}
              onClick={() => { if (interactive && moves.includes(i)) onMove?.(i); }}
              onKeyDown={event => {
                let next = i;
                if (event.key === 'ArrowLeft') next = y * 8 + (x + 7) % 8;
                else if (event.key === 'ArrowRight') next = y * 8 + (x + 1) % 8;
                else if (event.key === 'ArrowUp') next = ((y + 7) % 8) * 8 + x;
                else if (event.key === 'ArrowDown') next = ((y + 1) % 8) * 8 + x;
                else return;
                event.preventDefault(); setFocus(next); refs.current[next]?.focus();
              }}>{contents(i)}</button>;
        })}
      </div>)}
    </div>
    <div className="board-coordinates bottom" aria-hidden="true">{'ABCDEFGH'.split('').map(c => <span key={c}>{c}</span>)}</div>
  </div>;
}
