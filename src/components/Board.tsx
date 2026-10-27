import { useMemo, useRef, useState } from 'react';
import { RotateCw } from 'lucide-react';
import { flips, legal } from '../game/engine.ts';
import { boardTransition } from '../game/presentation.ts';
import { colorName, coordinate } from '../game/types.ts';
import type { Position } from '../game/types.ts';
export function Board({ position, onMove, onUnavailableClick, interactive = false, showLegal = true, hint = null, decorative = false }: {
  position: Position; onMove?: (move: number) => void; onUnavailableClick?: () => void; interactive?: boolean; showLegal?: boolean; hint?: number | null; decorative?: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const [focus, setFocus] = useState(19);
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const [animation, setAnimation] = useState({ position, placed: null as number | null, flipped: [] as number[] });
  if (animation.position !== position) setAnimation({ position, ...boardTransition(animation.position, position) });
  const moves = useMemo(() => legal(position), [position]);
  const preview = useMemo(() => interactive && hover !== null && moves.includes(hover) ? flips(position, hover) : [], [position, moves, interactive, hover]);
  const lastDot = (i: number) => position.lastMove === i && <span className="last-dot" />;
  const contents = (i: number) => <>
    {!!position.cells[i] && <span key={position.cells[i]} className={`disc ${position.cells[i] === 1 ? 'black' : 'white'} ${animation.flipped.includes(i) ? 'flipping' : animation.placed === i ? 'placed' : ''} ${preview.includes(i) ? 'will-flip' : ''}`}>
      {animation.flipped.includes(i) ? <>
        <span className={`disc-face back ${position.cells[i] === 1 ? 'white' : 'black'}`} />
        <span className={`disc-face front ${position.cells[i] === 1 ? 'black' : 'white'}`}>{lastDot(i)}</span>
      </> : lastDot(i)}
      {preview.includes(i) && <span className="flip-marker" aria-hidden="true"><RotateCw strokeWidth={2.5} /></span>}
    </span>}
    {!position.cells[i] && interactive && hover === i && moves.includes(i) && <span className={`disc ghost ${position.turn === 1 ? 'black' : 'white'}`} />}
    {!position.cells[i] && preview.length > 0 && hover === i && <span className="preview-count" aria-hidden="true">翻 {preview.length} 子</span>}
    {!position.cells[i] && showLegal && moves.includes(i) && <span className="legal-dot" />}
    {hint === i && <span className="hint-ring" />}
    {[18, 21, 42, 45].includes(i) && <span className="star-point" />}
  </>;
  return <div className={`board-frame ${decorative ? 'decorative' : ''}`}>
    <div className="board-coordinates top" aria-hidden="true">{'ABCDEFGH'.split('').map(c => <span key={c}>{c}</span>)}</div>
    <div className="board-coordinates left" aria-hidden="true">{Array.from({ length: 8 }, (_, i) => <span key={i}>{i + 1}</span>)}</div>
    <div className="board-grid" role={decorative ? undefined : 'grid'} aria-label={decorative ? undefined : onUnavailableClick ? '黑白棋棋盘，当前无合法落点，请确认跳过本回合' : '黑白棋棋盘，方向键移动，回车落子'}>
      {Array.from({ length: 8 }, (_, y) => <div className="board-row" role={decorative ? undefined : 'row'} key={y}>
        {Array.from({ length: 8 }, (_, x) => {
          const i = y * 8 + x;
          return decorative ? <div key={i} className="cell">{contents(i)}</div> :
            <button key={i} ref={el => { refs.current[i] = el; }} role="gridcell" tabIndex={focus === i ? 0 : -1}
              aria-label={`${coordinate(i)}，${position.cells[i] ? colorName(position.cells[i] as 1 | -1) : moves.includes(i) && (interactive || showLegal) ? '可落子' : '空位'}${hint === i ? '，推荐落点' : ''}${hover === i && preview.length ? `，可翻转 ${preview.length} 子` : ''}`}
              aria-disabled={!interactive || !moves.includes(i)} className={`cell ${interactive && moves.includes(i) ? 'playable' : ''} ${hint === i ? 'recommended' : ''}`}
              onPointerEnter={event => { if (event.pointerType === 'mouse') setHover(i); }} onPointerLeave={() => setHover(null)}
              onPointerDown={event => { if (event.pointerType !== 'mouse') setHover(null); }}
              onFocus={event => { setFocus(i); if (event.currentTarget.matches(':focus-visible')) setHover(i); }} onBlur={() => setHover(null)}
              onClick={() => { if (onUnavailableClick) onUnavailableClick(); else if (interactive && moves.includes(i)) onMove?.(i); }}
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
