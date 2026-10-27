import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const api = spawn(process.execPath, ['scripts/native-api.mjs'], { cwd: root, stdio: 'inherit' });
const vite = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1'], { cwd: root, stdio: 'inherit' });
let closing = false;
function close() { if (closing) return; closing = true; api.kill('SIGTERM'); vite.kill('SIGTERM'); }
for (const child of [api, vite]) { child.on('exit', code => { if (!closing) { process.exitCode = code || 0; close(); } }); child.on('error', error => { console.error(error.message); process.exitCode = 1; close(); }); }
process.on('SIGINT', close); process.on('SIGTERM', close);
