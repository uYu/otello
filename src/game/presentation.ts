import type { Position } from './types.ts';

// Keep these in sync with the disc animations in style.css.
export const TURN_SETTLE_MS = 750;
export const PASS_PAUSE_MS = 2800;
export const AI_MIN_THINK_MS = 600;
export const REPLAY_STEP_MS = 1200;

// The engine has already advanced past a forced pass. Keep the player who
// cannot move on screen until the pass has been acknowledged.
export function presentedPosition(position: Position, awaitingPass: boolean): Position {
  return awaitingPass && position.passed && !position.over ? { ...position, turn: position.passed } : position;
}

export function boardTransition(before: Position, after: Position) {
  const placed = after.lastMove;
  const singleMove = placed !== null && before.cells[placed] === 0 && after.cells[placed] !== 0
    && after.cells.filter(Boolean).length === before.cells.filter(Boolean).length + 1;
  return {
    placed: singleMove ? placed : null,
    flipped: singleMove ? before.cells.flatMap((cell, i) => cell !== 0 && cell !== after.cells[i] ? [i] : []) : [],
  };
}
