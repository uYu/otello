export function beginnerGameSeed(createdAt: string): number {
  let hash = 2166136261;
  for (const char of createdAt) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
  return hash >>> 0;
}
