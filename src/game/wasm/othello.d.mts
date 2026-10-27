interface Module {
  HEAP32: Int32Array;
  _ot_input(): number;
  _ot_output(): number;
  _ot_legal(input: number, turn: number, output: number): number;
  _ot_play(input: number, turn: number, move: number, output: number): number;
  _ot_search(input: number, turn: number, ms: number, depth: number, level: number, seed: number, output: number): number;
}
export default function createModule(): Promise<Module>;
