"""Generate every new README visual from committed CSVs, without caches or training."""
from pathlib import Path
import sys
import html
import tomllib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.report import headline_rows, MODELS, SCOREBOARD, GROUPS, COLORS, RESULTS

OUT = Path('docs/img')


def save(fig: plt.Figure, name: str) -> None:
    """Save an opaque white-background PNG with consistent styling."""
    fig.savefig(OUT/name, dpi=170, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """Render headline, scoreboard, costs, ablations, forecast weeks and badges."""
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'figure.facecolor':'white','axes.facecolor':'white','text.color':'#18212B',
                         'axes.labelcolor':'#18212B','xtick.color':'#18212B','ytick.color':'#18212B'})
    head = headline_rows()
    fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=True)
    horizons = sorted(head.horizon.unique())
    for ax, group in zip(axes, GROUPS):
        for i, model in enumerate(MODELS):
            data = head.loc[head.symbol_group.eq(group)&head.model.eq(model)].sort_values('horizon')
            x = np.arange(len(horizons)) + (i-(len(MODELS)-1)/2)*.13
            ax.bar(x,data.qlike_raw,.12,color=COLORS[i],label=model,alpha=1 if group==GROUPS[0] else .62,
                yerr=np.stack([data.qlike_raw-data.qlike_raw_lo,data.qlike_raw_hi-data.qlike_raw]),
                error_kw={'elinewidth':.8,'capsize':2})
        ax.set(ylabel='Raw QLIKE (lower is better)',title=group)
        ax.grid(axis='y',alpha=.15)
    axes[1].set_facecolor('#F5F7FA')
    axes[1].set_xticks(np.arange(len(horizons)),[f'h = {h} days' for h in horizons])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,0,1,.91])
    save(fig,'headline.png')
    mcs = pd.read_csv(RESULTS/'stage5_mcs.csv').set_index(['symbol_group','horizon','model','metric'])
    fig, axes = plt.subplots(2,1,figsize=(12,8))
    for ax, metric, title in zip(axes,['qlike_raw','qlike'],['Raw QLIKE','Calibrated QLIKE']):
        matrix, stars, labels = [], [], []
        for group in GROUPS:
            for h in horizons:
                rows = head.loc[head.symbol_group.eq(group)&head.horizon.eq(h)].set_index('model').loc[SCOREBOARD]
                matrix.append(rows[metric].to_numpy())
                stars.append([bool(mcs.loc[(group,h,r.family,metric),'mcs_95_included']) for r in rows.itertuples()])
                labels.append(f'{group} · h={h}')
        values = np.array(matrix)
        relative = (values-values.min(axis=1)[:,None])/(np.ptp(values,axis=1)[:,None]+1e-12)
        ax.imshow(relative,cmap='RdYlGn_r',vmin=-.2,vmax=1.5,aspect='auto',alpha=.35)
        for row in range(len(values)):
            for col in range(len(SCOREBOARD)):
                best = np.isclose(values[row,col], values[row].min(),rtol=0,atol=1e-12)
                if best:
                    ax.add_patch(Rectangle((col-.5,row-.5),1,1,fill=False,edgecolor='#007F5F',linewidth=2.5))
                ax.text(col,row,f'{values[row,col]:.4f}'+(' ★' if stars[row][col] else ''),ha='center',va='center',fontsize=9,
                    weight='bold' if best else 'normal')
        ax.set_xticks(range(len(SCOREBOARD)),[m.replace('HAR-log-mean-RV','HAR log\nmean RV').replace('Fine-tuned','Fine-tuned\n(frozen choice)') for m in SCOREBOARD],fontsize=9)
        ax.set_yticks(range(len(labels)),labels)
        ax.set_title(title+' — green border: row minimum; ★: in 95% MCS',fontsize=11)
    fig.tight_layout()
    save(fig,'scoreboard.png')
    costs = pd.read_csv(RESULTS/'stage4_finetune.csv')
    costs = costs.loc[costs.symbol_group.eq('fit-eligible') & costs.horizon.eq(1)]
    fig, ax = plt.subplots(figsize=(10,5))
    family_colors={'small-full':COLORS[5],'small-lora8':COLORS[1],'small-lora16':COLORS[2],
                   'tiny-full':COLORS[3],'tiny-lora8':COLORS[0],'tiny-lora16':COLORS[4]}
    for i,r in enumerate(costs.itertuples()):
        name=r.model.replace('Chronos-Bolt-','')
        ax.scatter(r.trainable_parameters,r.qlike_raw,s=r.fit_seconds_mean*3,color=family_colors[name],alpha=.75,edgecolors='white')
        right=r.trainable_parameters==costs.trainable_parameters.max()
        ax.annotate(name,(r.trainable_parameters,r.qlike_raw),xytext=(-8,10) if right else (5,7 if i%2 else -15),
                    ha='right' if right else 'left',textcoords='offset points',fontsize=8)
    for model,color in [('HAR-RV',COLORS[1]),('HAR-weekday',COLORS[2])]:
        value=head.loc[head.model.eq(model)&head.horizon.eq(1)&head.symbol_group.eq('fit-eligible'),'qlike_raw'].item()
        ax.axhline(value,color=color,linestyle='--',label=f'{model}: {value:.4f} (reference)')
    ax.set_xscale('log')
    ax.set(xlabel='Trainable neural parameters (log scale)',ylabel='Raw QLIKE · fit-eligible · h=1',
        title='Adapter or full model? Bubble area = mean training seconds per seed')
    reference_legend=ax.legend(loc='upper left',fontsize=9)
    ax.add_artist(reference_legend)
    times=costs.fit_seconds_mean.quantile([0,.5,1]).to_numpy()
    handles=[ax.scatter([],[],s=t*3,color='#697386',alpha=.5,label=f'{t:.1f} s') for t in times]
    ax.legend(handles=handles,title='Mean fit / seed',loc='center right',fontsize=8,labelspacing=1.5)
    ax.margins(x=.12,y=.12)
    ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig,'compute.png')
    ablation = pd.read_csv(RESULTS/'stage5_ablation_summary.csv')
    for axis in ['context','fraction','rank']:
        fig, ax = plt.subplots(figsize=(6,3))
        rows = ablation.loc[ablation.axis.eq(axis)&ablation.horizon.eq(1)&ablation.symbol_group.eq('fit-eligible')].copy()
        rows['line'] = 'LoRA' if axis=='rank' else rows['mode']
        for i,(mode,trial) in enumerate(rows.groupby('line')):
            trial=trial.sort_values('value')
            ax.plot(trial.value,trial.qlike_raw,'o-',color=COLORS[5-i],label=mode)
            ax.fill_between(trial.value,trial.qlike_raw_lo,trial.qlike_raw_hi,color=COLORS[5-i],alpha=.12)
        ax.set_xticks(sorted(rows.value.unique()))
        ax.set(xlabel={'context':'Context days','fraction':'Earliest training fraction','rank':'LoRA rank'}[axis],
               ylabel='Raw QLIKE',title='Fit-eligible · h=1 · reporting only')
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
        fig.tight_layout()
        save(fig,f'ablation_{axis}.png')
    fan = pd.read_csv(RESULTS/'stage6_forecast_weeks.csv',parse_dates=['target_end'])
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    for ax,(week,rows) in zip(axes,fan.groupby('week',sort=False)):
        ax.plot(rows.target_end,rows.realized_rv,'o-',color='#18212B',label='Realized RV')
        ax.plot(rows.target_end,rows.har_weekday,'--',color=COLORS[2],label='HAR-weekday')
        ax.plot(rows.target_end,rows.ft_median,color=COLORS[5],label='Fine-tuned seed median')
        ax.fill_between(rows.target_end,rows.ft_low,rows.ft_high,color=COLORS[5],alpha=.25,label='Seed range')
        ax.set_yscale('log')
        ax.set(ylabel='Daily RV (log axis)',title=f'BTCUSDT · {week} · {rows.target_end.iloc[0].year}')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.tick_params(axis='x',rotation=30)
        ax.legend(fontsize=8)
        ax.grid(alpha=.15)
    fig.tight_layout()
    save(fig,'forecast_weeks.png')
    testfile=RESULTS/'stage6_verification.csv'
    tests = str(pd.read_csv(testfile).passed.iloc[0])+' passed' if testfile.exists() else 'pending'
    python=tomllib.loads(Path('pyproject.toml').read_text())['project']['requires-python']
    gpu=pd.read_csv(RESULTS/'stage3_costs.csv').device_name.iloc[0].replace('NVIDIA GeForce ','').replace(' Laptop GPU','')
    for name,value in [('Python',python),('Tests',tests),('License','MIT'),('GPU',gpu)]:
        width=85+len(value)*7
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="24"><rect width="{width}" height="24" rx="4" fill="#18212B"/><text x="8" y="16" fill="white" font-family="sans-serif" font-size="12">{html.escape(name+": "+value)}</text></svg>'
        (OUT/f'badge_{name.lower()}.svg').write_text(svg,encoding='utf-8')


if __name__ == '__main__':
    main()
