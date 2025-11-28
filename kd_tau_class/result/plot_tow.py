 plot_tau_sweep.py
# tau_sweep_results.csv를 읽어 히트맵/라인플롯 생성 (matplotlib만 사용)

import csv
from collections import defaultdict
import matplotlib.pyplot as plt

CSV_PATH = 'tau_sweep_results.csv'

def _read_rows(path=CSV_PATH):
    rows = []
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            # 숫자형 변환
            for k, v in list(row.items()):
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
            rows.append(row)
    return rows

def _filter_rows(rows, **eq):
    out = []
    for r in rows:
        ok = True
        for k, v in eq.items():
            if str(r[k]) != str(v):
                ok = False; break
        if ok: out.append(r)
    return out

def _pivot_max_ep(rows, metric):
    # (tau_g, tau_b)별로 ep가 가장 큰 결과를 사용
    gs = sorted({int(r['tau_g']) for r in rows})
    bs = sorted({int(r['tau_b']) for r in rows})
    best = {}
    for r in rows:
        key = (int(r['tau_g']), int(r['tau_b']))
        if key not in best or r['ep'] > best[key]['ep']:
            best[key] = r
    grid = [[float('nan') for _ in bs] for _ in gs]
    for (g, b), r in best.items():
        grid[gs.index(g)][bs.index(b)] = r[metric]
    return gs, bs, grid

def heatmap(metric, title, **filters):
    rows = _read_rows()
    rows = _filter_rows(rows, **filters) if filters else rows
    gs, bs, grid = _pivot_max_ep(rows, metric)
    plt.figure()
    plt.imshow(grid, aspect='auto')  # 색상/스타일 지정하지 않음
    plt.xticks(range(len(bs)), bs)
    plt.yticks(range(len(gs)), gs)
    plt.xlabel('tau_b'); plt.ylabel('tau_g')
    plt.title(title); plt.colorbar(); plt.tight_layout()
    plt.show()

def lineplot(metric, title, group_key='tau_g', **filters):
    rows = _read_rows()
    rows = _filter_rows(rows, **filters) if filters else rows
    grouped = defaultdict(list)
    for r in rows:
        grouped[int(r[group_key])].append((int(r['tau_b']), r[metric], r['ep']))
    plt.figure()
    for g, arr in sorted(grouped.items()):
        best = {}
        for b, val, ep in arr:
            if b not in best or ep > best[b][1]:
                best[b] = (val, ep)
        xs = sorted(best.keys())
        ys = [best[b][0] for b in xs]
        plt.plot(xs, ys, label=f'{group_key}={g}')
    plt.xlabel('tau_b'); plt.ylabel(metric)
    plt.title(title); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    # KD-only 결과 히트맵(둘 다 τ>1 케이스)
    heatmap('acc_forget_test', 'Forget Acc (test): KD-only, both> (scale=T2)',
            case='both>', scale='T2', ce_ret=0, ce_for_bad=0)
    heatmap('acc_retain_test', 'Retain Acc (test): KD-only, both> (scale=T2)',
            case='both>', scale='T2', ce_ret=0, ce_for_bad=0)

    # τ_b 스윕에 따른 MIA 곡선 (τ_g 별)
    lineplot('mia_forget', 'MIA on Forget vs tau_b (KD-only, scale=T2)',
             group_key='tau_g', scale='T2', ce_ret=0, ce_for_bad=0)
    
    
 

    # (원하면 train 지표도)
    heatmap('acc_forget_train', 'Forget Acc (train): KD-only, both>', case='both>', scale='T2', ce_ret=0, ce_for_bad=0)
    heatmap('acc_retain_train', 'Retain Acc (train): KD-only, both>', case='both>', scale='T2', ce_ret=0, ce_for_bad=0)

    # CE-both 케이스 비교 (retain에 CE, forget에 bad-hard CE)
    heatmap('acc_forget_test', 'Forget Acc (test): CE-both', case='CE-both', scale='T2', ce_ret=1, ce_for_bad=1)

