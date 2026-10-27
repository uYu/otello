interface Module {
  HEAP32: Int32Array;
  HEAPU8: Uint8Array;
  _ot_input(): number;
  _ot_output(): number;
  _ot_legal(input: number, turn: number, output: number): number;
  _ot_play(input: number, turn: number, move: number, output: number): number;
  _ot_search(input: number, turn: number, ms: number, depth: number, level: number, seed: number, output: number): number;
  _ot_search_clear_cache(): void;
  _ot_mcts_clear_cache(): void;
  _ot_default_search(input: number,turn: number,ms: number,seed: number,output: number): number;
  _ot_default_evaluate(input: number,turn: number,score: number): number;
  _ot_default_model_sha256(): number;
}
export default function createModule(): Promise<Module>;
