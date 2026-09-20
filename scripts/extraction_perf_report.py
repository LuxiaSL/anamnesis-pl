"""Build timing ledger and export plots; keep historical/current scopes separate."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root
    rows = []
    for p in sorted((root/'reference').glob('run*/run_record_shard*.json')):
        record = json.loads(p.read_text())
        bank = p.with_name(p.name.replace('run_record', 'own_content').replace('.json', '.npz'))
        with np.load(bank, allow_pickle=False) as z:
            n = len(z['cell'])
        rows.append(dict(kind='historical', attempt=f'{p.parent.name}/{p.stem}',
                         utc=record['started_utc'], n=n,
                         seconds_per_row=record['timing']['replay_s']/n,
                         scope='historical_driver_replay',
                         shard_wall_s_per_row=record['wall_seconds']/n,
                         startup_s=record['feature_lane']['startup_s'],
                         fidelity='banked_cpu',
                         source=str(p.relative_to(root))))
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name == 'reference':
            continue
        records = [json.loads(p.read_text()) for p in sorted(folder.glob('row-*.json'))]
        if not records:
            continue
        scopes = {r['scope'] for r in records}
        if len(scopes) != 1:
            raise ValueError(f'mixed timing scopes in {folder}')
        startup = folder/'startup.json'
        complete = (folder/'complete.json').exists()
        failed = any(r.get('cpu_repeat_bit_exact') is False for r in records)
        rows.append(dict(kind='measured', attempt=folder.name, utc=records[0]['utc'],
                         n=len(records),
                         seconds_per_row=float(np.mean([r['seconds_per_row'] for r in records])),
                         scope=scopes.pop(), shard_wall_s_per_row=None,
                         startup_s=json.loads(startup.read_text())['pool_s'] if startup.exists() else None,
                         fidelity='failed' if failed else ('baseline_complete' if complete else 'partial'),
                         source=str(folder.relative_to(root))))
    (root/'timing-ledger.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    fields = ['kind','attempt','utc','n','seconds_per_row','scope','shard_wall_s_per_row',
              'startup_s','fidelity','source']
    with (root/'timing-ledger.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    historical = [r for r in rows if r['kind']=='historical']
    measured = [r for r in rows if r['kind']=='measured']
    scopes=sorted({r['scope'] for r in measured}) or ['prefill+cached_replay+D2H+pool_submit_drain']
    fig, axes = plt.subplots(1,1+len(scopes),figsize=(6.5*(1+len(scopes)),5),layout='constrained')
    ax=axes[0]
    x=np.arange(len(historical))
    ax.plot(x,[r['seconds_per_row'] for r in historical],'o-',label='Recorded replay / emitted row')
    ax.plot(x,[r['shard_wall_s_per_row'] for r in historical],'s-',label='Recorded full shard / emitted row')
    ax.set_xticks(x,[r['attempt'].replace('run_record_shard','s') for r in historical],rotation=70)
    ax.set_title('Historical artifacts: separate timing boundaries')
    ax.set_ylabel('seconds / emitted row')
    ax.legend(fontsize=8)
    for ax,scope in zip(axes[1:],scopes,strict=True):
        group=[r for r in measured if r['scope']==scope]
        if group:
            for i,r in enumerate(group):
                ax.scatter(i,r['seconds_per_row'],marker='x' if r['fidelity']=='failed' else 'o',
                           color='crimson' if r['fidelity']=='failed' else 'steelblue')
                ax.annotate(f"n={r['n']} · {r['fidelity']}",(i,r['seconds_per_row']),
                            xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
            ax.set_xticks(range(len(group)),[r['attempt'] for r in group],rotation=30)
        else:
            ax.text(.5,.5,'No new completed row measurements yet',ha='center',va='center',transform=ax.transAxes)
        ax.set_title(scope.replace('+',' +\n'),fontsize=9)
        ax.set_ylabel('seconds / replayed span (not full shard)')
    fig.suptitle('Extraction performance — all attempts retained; no GPU lane certified')
    for ext in ('png','svg'):
        fig.savefig(root/f'timing-history.{ext}',dpi=160)
    plt.close(fig)
    print(f'Wrote ledger ({len(rows)} records) and PNG/SVG under {root}')


if __name__ == '__main__':
    main()
