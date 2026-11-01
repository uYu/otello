import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { resolve, dirname, extname, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const executable = resolve(process.env.OTELLO_NATIVE_BINARY ?? resolve(root, '.build/native-egaroucid/otello'));
const port = Number(process.env.OTELLO_API_PORT ?? 8207);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid OTELLO_API_PORT');
if (!existsSync(executable)) throw new Error('Native Egaroucid binary missing. Run npm run setup:egaroucid first.');

// A single persistent native process keeps the book loaded. Requests are serialized
// because Egaroucid owns global search state. Cancelling an active request kills
// that process; the next request starts a clean one.
class NativeClient {
  child = null;
  active = null;
  queue = [];
  closed = false;
  request(command, signal) {
    if (this.closed) return Promise.reject(new Error('Native service is closed'));
    if (this.queue.length >= 8) return Promise.reject(new Error('Native engine is busy'));
    return new Promise((resolve, reject) => {
      const item = { command, resolve, reject, signal, abort: null, timer: null };
      item.abort = () => {
        if (this.active === item) { this.reset(); this.finish(new Error('Request cancelled')); }
        else { this.queue = this.queue.filter(q => q !== item); item.reject(new Error('Request cancelled')); }
      };
      if (signal?.aborted) { reject(new Error('Request cancelled')); return; }
      signal?.addEventListener('abort', item.abort, { once: true });
      this.queue.push(item); this.pump();
    });
  }
  reset() { const child = this.child; this.child = null; child?.kill('SIGKILL'); }
  finish(error, result) {
    const item = this.active; if (!item) return;
    this.active = null; clearTimeout(item.timer);
    item.signal?.removeEventListener('abort', item.abort);
    if (error) item.reject(error); else item.resolve(result);
    this.pump();
  }
  pump() {
    if (this.closed || this.active || !this.queue.length) return;
    if (!this.child) {
      const child = spawn(executable, ['protocol'], { cwd: root, stdio: ['pipe', 'pipe', 'pipe'] });
      this.child = child;
      // Upstream diagnostics stay on stderr, never mixed into JSON protocol output.
      child.stderr.on('data', data => process.stderr.write(data));
      child.stdin.on('error', () => {});
      createInterface({ input: child.stdout }).on('line', line => {
        if (this.child !== child || !this.active) return;
        try {
          const data = JSON.parse(line);
          if (data.error) this.finish(new Error(String(data.error)));
          else if (data.engine !== 'egaroucid' || data.useBook !== !/(?:^|\s)--book off(?:\s|$)/.test(this.active.command) || (!data.useBook && data.bookHit !== false)) this.finish(new Error('Native engine book mode mismatch'));
          else this.finish(null, data);
        } catch { this.reset(); this.finish(new Error('Invalid native response')); }
      });
      const failed = () => { if (this.child === child) { this.reset(); this.finish(new Error('Native engine stopped')); } };
      child.on('error', failed); child.on('exit', failed);
    }
    const item = this.queue.shift(); this.active = item;
    item.timer = setTimeout(() => { this.reset(); this.finish(new Error('Native search exceeded 20 seconds')); }, 20000);
    this.child.stdin.write(item.command + '\n');
  }
  close() {
    this.closed = true; this.reset();
    if (this.active) this.finish(new Error('Native service is closed'));
    for (const item of this.queue) { item.signal?.removeEventListener('abort', item.abort); item.reject(new Error('Native service is closed')); }
    this.queue = [];
  }
}
const native = new NativeClient();
const send = (res, status, data) => {
  if (res.destroyed) return;
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff' });
  res.end(JSON.stringify(data));
};
const server = createServer(async (req, res) => {
  const controller = new AbortController();
  const disconnected = () => { if (!res.writableEnded) controller.abort(); };
  res.on('close', disconnected);
  try {
    const host = new URL(`http://${req.headers.host}`);
    if (!['localhost', '127.0.0.1'].includes(host.hostname)) { send(res, 403, { error: 'Invalid local host' }); return; }
    if (req.headers.origin) {
      const origin = new URL(req.headers.origin);
      if (origin.protocol !== 'http:' || !['localhost', '127.0.0.1'].includes(origin.hostname)) { send(res, 403, { error: 'Only local browser origins are accepted' }); return; }
    }
    const url = new URL(req.url, host);
    if (url.pathname === '/api/native/health' && req.method === 'GET') {
      const result = await native.request('analyze --engine egaroucid --eg-level 1 --book on', controller.signal);
      if (!result.bookHit) throw new Error('Opening book verification failed');
      send(res, 200, { available: true, engine: 'egaroucid', engineVersion: result.engineVersion, useBook: true, bookReady: true }); return;
    }
    if (url.pathname === '/api/native/move' && req.method === 'POST') {
      if (!req.headers['content-type']?.startsWith('application/json')) { send(res, 415, { error: 'Expected JSON' }); return; }
      let body = '', size = 0;
      for await (const chunk of req) { size += chunk.length; if (size > 4096) { send(res, 413, { error: 'Request too large' }); return; } body += chunk; }
      let input;
      try { input = JSON.parse(body); } catch { send(res, 400, { error: 'Invalid JSON' }); return; }
      if (!input || input.version !== 1 || typeof input.board !== 'string' || !/^[BW.]{64}$/.test(input.board) || ![1, -1].includes(input.turn) || !['easy', 'normal', 'hard'].includes(input.difficulty)) {
        send(res, 400, { error: 'Invalid position or difficulty' }); return;
      }
      const level = { easy: 1, normal: 5, hard: 10 }[input.difficulty];
      const useBook = input.difficulty !== 'easy';
      const variety = 1;
      const result = await native.request(`analyze --engine egaroucid --board ${input.board} --turn ${input.turn} --eg-level ${level} --book ${useBook ? 'on' : 'off'} --book-variety ${variety}`, controller.signal);
      send(res, 200, { move: result.moveIndex, score: result.score, depth: result.depth, elapsed: result.elapsedMs, nodes: result.nodes, ttHits: null, exact: result.exact, candidates: [], engine: 'egaroucid', engineVersion: result.engineVersion, bookHit: result.bookHit, bookVariety: result.bookVariety, useBook }); return;
    }
    if (url.pathname.startsWith('/api/')) { send(res, 404, { error: 'Unknown endpoint' }); return; }
    if (req.method !== 'GET' && req.method !== 'HEAD') { send(res, 405, { error: 'Method not allowed' }); return; }
    const staticRoot = resolve(root, 'dist');
    const path = resolve(staticRoot, '.' + decodeURIComponent(url.pathname === '/' ? '/index.html' : url.pathname));
    if (!path.startsWith(staticRoot + sep)) { send(res, 403, { error: 'Invalid path' }); return; }
    try {
      const content = await readFile(path);
      const mime = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml', '.png': 'image/png' }[extname(path)] ?? 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': mime, 'X-Content-Type-Options': 'nosniff' }); res.end(req.method === 'HEAD' ? undefined : content);
    } catch { send(res, 404, { error: 'Page not built. Run npm run build, or use npm run dev:native.' }); }
  } catch (error) { send(res, 503, { error: error instanceof Error ? error.message : 'Native engine unavailable' }); }
  finally { res.removeListener('close', disconnected); }
});
server.listen(port, '127.0.0.1', () => process.stdout.write(`Egaroucid native service: http://127.0.0.1:${port}\n`));
server.on('error', error => { console.error(error.message); native.close(); process.exitCode = 1; });
const close = () => { native.close(); server.close(); server.closeAllConnections(); };
process.on('SIGTERM', close); process.on('SIGINT', close);
