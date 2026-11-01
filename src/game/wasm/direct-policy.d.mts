interface Module {
  HEAP32: Int32Array;
  HEAPU8: Uint8Array;
  HEAPF64: Float64Array;
  _ot_policy_input(): number;
  _ot_policy_output(): number;
  _ot_policy_model_sha256(): number;
  _ot_policy_infer(turn: number): number;
}
export default function createModule(): Promise<Module>;
