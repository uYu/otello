#!/usr/bin/env python3
"""Live W/D/L monitor for versioned experiments (standard library only)."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time
import unicodedata
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def pad(text,width):
    size=sum(2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in text)
    return text+' '*max(0,width-size)


def display(config,details=False):
    stamp = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%H:%M:%S')
    lines = [f"Otello {config.get('version', 'v1')}  {stamp}", ''] if details else []
    for item in config['experiments']:
        paths = [ROOT / item['report']]
        if item.get('runtimeReport'):
            paths.insert(0, ROOT / item['runtimeReport'])
        result = {}
        for path in paths:
            result = read(path.with_suffix('.live.json')) or read(path) or {}
            if result:
                break
        if result.get('experiment') in ('pattern-incremental-benchmark', 'pattern-int8-benchmark', 'pattern-hotpath-benchmark', 'pattern-legal-benchmark', 'pattern-suite-sampling'):
            lines.extend([item['name'], ('已完成' if result.get('complete') else result.get('stage','测量中'))+
                f" · {result.get('completed',0)}/{result.get('total',0)} · {result.get('seconds',0):.0f} 秒"])
            if result.get('summary'):lines.append(result['summary'])
            lines.append('');continue
        if result.get('experiment') in ('pattern-width-training', 'checkpoint-average-training'):
            lines.extend([item['name'], ('已完成' if result['complete'] else result.get('stage','训练中'))+
                f" · {result.get('variant','')} · 第 {result.get('epoch',0)}/{result.get('epochs',15)} 轮 · {result.get('seconds',0):.0f} 秒"])
            if result.get('validation'):
                score=result['validation'];lines.append(f"验证胜负 MSE {score['mse']:.6f} · 策略命中率 {score['policyTop1']:.2%}")
            if result['complete']:
                for variant in result.get('variants',[]):
                    score=variant['metrics']['validation'];lines.append(f"{variant['name']}：验证胜负 MSE {score['mse']:.6f} · 策略命中率 {score['policyTop1']:.2%}")
            lines.append('');continue
        if result.get('experiment') == 'pattern-outcome-training':
            score=result['validation']
            lines.extend([item['name'], ('已完成' if result['complete'] else '训练中')+
                f" · 第 {result['epoch']}/{result['epochs']} 轮 · {result['seconds']:.0f} 秒",
                f"验证胜负 MSE {score['mse']:.6f} · 策略命中率 {score['policyTop1']:.2%}", ''])
            continue
        if result.get('experiment') == 'cnn-offline-training':
            lines.extend([item['name'],
                ('已完成' if result.get('complete') else result.get('stage', '训练中')) +
                f" · 模型 {result.get('completedModels', 0)}/{result.get('totalModels', 3)} · {result.get('seconds', 0):.0f} 秒",
                f"当前 {result.get('currentChannels', '?')} 通道 · 第 {result.get('epoch', 0)}/{result.get('maxEpochs', 30)} 轮"])
            score = result.get('validation')
            if result.get('currentName'):
                lines.append(f"参数组：{result['currentName']}")
            if score:
                lines.append(f"验证：胜负 MSE {score['value']['outcomeMse']:.6f} · 走法命中率 {score['policy']['top1']:.2%}")
            if result.get('complete'):
                for candidate in result['candidates']:
                    score = candidate['metrics']['test']
                    label = candidate.get('name', f"{candidate['channels']} 通道")
                    lines.append(f"{label} 测试：胜负 MSE {score['value']['outcomeMse']:.6f} · 非和局准确率 {score['value']['nonDrawSignAccuracy']:.2%} · 走法命中率 {score['policy']['top1']:.2%}")
            lines.append('')
            continue
        if result.get('experiment') == 'prediction-training':
            lines.extend([item['name'],
                ('已完成' if result.get('complete') else '训练/预测中') +
                f" · 第 {result.get('epoch', 0)}/{result.get('epochs', 1)} 轮 · 分片 {result.get('completedShards', 0)}/{result.get('totalShards', 64)}",
                f"新增局面 {result.get('newPositions', 0):,} · 原训练局面 {result.get('oldPositions', 0):,} · {result.get('seconds', 0):.0f} 秒"])
            score = result.get('validation')
            if score:
                lines.append(f"最近验证：胜负 MSE {score['value']['outcomeMse']:.6f} · 策略命中率 {score['policy']['top1']:.2%}")
            if result.get('complete'):
                for label, key in [('v1.9', 'baselineNative'), ('验证最佳检查点', 'candidateNative')]:
                    score = result['metrics']['test'][key]
                    lines.append(f"{label} 测试：胜负 MSE {score['value']['outcomeMse']:.6f} · 非和局准确率 {score['value']['nonDrawSignAccuracy']:.2%} · 策略命中率 {score['policy']['top1']:.2%}")
                if result.get('fullEpochMetrics'):
                    score = result['fullEpochMetrics']['test']['candidateNative']
                    lines.append(f"完整一轮测试：胜负 MSE {score['value']['outcomeMse']:.6f} · 非和局准确率 {score['value']['nonDrawSignAccuracy']:.2%} · 策略命中率 {score['policy']['top1']:.2%}")
            lines.append('')
            continue
        counts = result.get('results', {})
        games = result.get('games', 0)
        planned = result.get('plannedGames', item['games'])
        state = '已完成' if result.get('complete') else ('进行中' if result else '排队中')
        if item.get('status') == 'stopped':state = '已停止'
        if not result and item.get('trainingLog'):
            try:
                rows=(ROOT/item['trainingLog']).read_text().splitlines()
                progress=next((json.loads(row) for row in reversed(rows) if row.startswith('{') and '"epoch"' in row),{})
            except (OSError,json.JSONDecodeError):progress={}
            epoch=progress.get('epoch',0);epochs=item['epochs']
            stage=f'训练中  {epoch}/{epochs} 轮' if epoch<epochs else '训练结束，准备对局…'
            lines.extend([item.get('shortName',item['name']),stage,'']);continue
        if result.get('experiment') in ('search-matrix', 'nnue-search-100ms', 'candidate-vs-approved-baseline'):
            elapsed=int(result.get('seconds',0))
            lines.extend([item.get('shortName',item['name']),
                          f'{state}  {games:02d} / {planned:02d} 局    用时 {elapsed//60:02d}:{elapsed%60:02d}', ''])
            aliases=item.get('displayNames',{})
            names={key:aliases.get(c['id'],c['name']) for c in result['configs'] for key in (c['id'],c['name'])}
            order=item.get('displayOrder',[c['id'] for c in result['configs']])
            configs=sorted(result['configs'],key=lambda c:order.index(c['id']) if c['id'] in order else len(order))
            width=max(10,max(len(names[c['id']])*2 for c in configs))
            focus=configs[0]
            r=counts[focus['id']]
            lines.append(f"{names[focus['id']]}：{r['win']} 胜 {r['draw']} 平 {r['loss']} 负 · 得分率 {r['scoreRate']:.1%}")
            for c in configs:
                r=counts[c['id']]
                if details and 'meanWallMs' in r:
                    unit='模拟' if r.get('searchUnit')=='simulations' else '节点'
                    lines.append(f"  {names[c['id']]}：平均子差 {r['meanDiscDifference']:+.2f}；平均耗时 {r['meanWallMs']:.1f}ms；{unit}/步 {r['meanSearchUnits']:,.0f}")
            lines.append('')
            current=result.get('currentGame')
            showing_final=False
            if not current and games:
                completed=(read(path) or {}).get('records',[])
                if completed:
                    last=completed[-1];board=last['finalBoard']
                    current=dict(number=last['number'],ply=len(last['moves']),black=last['black'],white=last['white'],
                                 board=board,blackCount=board.count(1),whiteCount=board.count(-1),turn=0)
                    showing_final=True
            if current:
                label=f"第 {current['number']} 局终盘" if showing_final else f"当前第 {current['number']} 局"
                lines.append(f"{label} · 第 {current['ply']} 步")
                if 'blackCount' in current:
                    black=names.get(current['black'],current['black']);white=names.get(current['white'],current['white'])
                    lines.append('黑  '+pad(black,width)+f"{current['blackCount']:3d}"+('  ← 行棋' if current['turn']==1 else ''))
                    lines.append('白  '+pad(white,width)+f"{current['whiteCount']:3d}"+('  ← 行棋' if current['turn']==-1 else ''))
                    board=current.get('board')
                    if board is None:
                        captured=read(path.with_suffix('.live.board.json')) or {}
                        if captured.get('number')==current['number'] and captured.get('ply')==current['ply']:
                            board=captured.get('board')
                    if board is not None:
                        lines.extend(['', '    a b c d e f g h'])
                        for row in range(8):
                            cells=' '.join({1:'*',-1:'o',0:'.'}[cell] for cell in board[row*8:row*8+8])
                            lines.append(f' {row+1}  {cells}  {row+1}')
                        lines.extend(['    a b c d e f g h','    * 黑  o 白  . 空'])
                    else:
                        lines.append('棋盘从下一局开始显示。')
                else:
                    lines.append(f"  黑 {current['black']} / 白 {current['white']}")
            elif not result.get('complete') and games<planned:
                lines.append(f'  第 {games+1} 局进行中；统计在每局结束后更新。')
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
    lines.append('Ctrl+C 退出监控，不影响对局。')
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'models/v1-experiments.json')
    parser.add_argument('--interval', type=float, default=1)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--details', action='store_true',help='Show timing, disc difference and search statistics')
    parser.add_argument('--all', action='store_true',help='Include archived experiments in the monitor')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    try:
        while True:
            config = read(args.config)
            if config and args.all:
                config=dict(config,experiments=config['experiments']+config.get('archivedExperiments',[]))
            view = display(config,args.details) if config else '实验配置准备中…'
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(view, flush=True)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n监控已退出，对局不受影响。')
