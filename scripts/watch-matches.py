#!/usr/bin/env python3
"""Live W/D/L monitor for the two v1 experiments (standard library only)."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def display(config):
    stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
    lines = [f"Otello {config.get('version', 'v1')} 对局监控  {stamp}", '']
    for item in config['experiments']:
        path = ROOT / item['report']
        live = read(path.with_suffix('.live.json'))
        result = live or read(path) or {}
        counts = result.get('results', {})
        games = result.get('games', 0)
        planned = result.get('plannedGames', item['games'])
        state = '已完成' if result.get('complete') else ('进行中' if result else '排队中')
        if item.get('status') == 'stopped':state = '已停止'
        if result.get('experiment') in ('search-matrix', 'nnue-search-100ms'):
            lines.extend([item['name'],f'  {state}  {games}/{planned} 局'])
            for c in result['configs']:
                r=counts[c['id']]
                lines.append(f"  {c['name']}: 胜 {r['win']} 平 {r['draw']} 负 {r['loss']}，得分率 {r['scoreRate']:.1%}")
            current=result.get('currentGame')
            if current:lines.append(f"  当前第 {current['number']} 局，第 {current['ply']} 步：{current['black']} / {current['white']}")
            lines.append('');continue
        lines.extend([item['name'], f"  统计视角：{item['focus']}",
                      f"  {state}  {games}/{planned} 局    胜 {counts.get('win', 0)}  平 {counts.get('draw', 0)}  负 {counts.get('loss', 0)}"])
        if games:
            score = (counts.get('win', 0) + counts.get('draw', 0) / 2) / games
            lines.append(f'  得分率 {score:.1%}')
        if result.get('experiment') == 'book':
            hits = sum(v.get('bookHits', 0) for v in result.get('engines', {}).values())
            lines.append(f'  开局库命中 {hits} 次')
        current = result.get('currentGame')
        if current:
            lines.append(f"  当前第 {current['number']} 局，第 {current.get('ply', 0)} 步；黑 {current.get('black', 2)} / 白 {current.get('white', 2)}")
        lines.append('')
    lines.append('Ctrl+C 退出监控；对局会继续运行。')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'models/v1-experiments.json')
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            config = read(args.config)
            view = display(config) if config else '实验配置准备中…'
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(view, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n监控已退出，对局不受影响。')
