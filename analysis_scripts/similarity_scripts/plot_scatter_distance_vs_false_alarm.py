"""
Speaker-pair embedding distance vs. perceptual legality effect.

Replaces plot_scatter.py / plot_scatter_exp1.py / plot_scatter_exp2.py /
plot_scatter_combined.py, which read the removed `output_results{,2}.csv`.
Same figures, same style, built from the current similarity CSVs:

  plots/scatter_distance_vs_false_alarm_exp1.png
  plots/scatter_distance_vs_false_alarm_exp2.png
  plots/scatter_distance_vs_false_alarm_combined.png

Distances come from `per_similarity_results/per{1,2}_speaker_pair_per_sim_full.csv`
— pair ("mutual") standardization, the Chernyak et al. (2024) procedure, which
is what all reported results use.

Two things differ from the superseded figures:

  * process_audio.py resampled to 16 kHz only inside its MP3-conversion branch.
    The per1 french_FI recordings are 44.1 kHz WAVs with no MP3 source, so they
    reached HuBERT at 44.1 kHz, inflating both french_FI pair distances to
    ~25-26 instead of ~16-18 and pulling Exp 1's r down to 0.710.
    process_audio_full.py resamples at load time, giving r = 0.835.
  * per1_speaker_legality_diffs.csv contains a duplicated EN_F1,FR_F1 row, which
    the old scripts plotted twice (N = 7 for 6 pairs). Here it is deduplicated.

Run: python3 analysis_scripts/similarity_scripts/plot_scatter_distance_vs_false_alarm.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns


# Times New Roman throughout, rather than matplotlib's default DejaVu Sans.
# The fallbacks are metric-compatible Times clones, so the figures still build
# on a machine without the Microsoft font (Linux/CI); order matters.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman',
                              'Liberation Serif', 'DejaVu Serif']
# Keep any math text in a Times-like face too, so it doesn't fall back to the
# sans-serif default mid-label.
plt.rcParams['mathtext.fontset'] = 'stix'
# Embed real TrueType rather than Type 3 subsets: most publishers reject Type 3,
# and it keeps the text selectable/searchable in a submitted PDF or EPS.
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PLOTS_DIR = os.path.join(REPO_ROOT, "plots")

# Audio folder names <- experiment speaker labels (Exp 1 only; Exp 2 already matches)
SPEAKER_MAP_PER1 = {
    'EN_F1': 'NS_F',
    'EN_M1': 'NS_M',
    'FR_F1': 'french_F',
    'FR_M1': 'french_M',
    'FR_F2': 'french_FI',
}

COLORS = {1: '#2b83ba', 2: '#d7191c'}

# Single-column journal figure spec, used by every figure here.
# 3.15 in is the target column width; fonts are sized for that FINAL width, so
# they look large relative to the axes compared with the old 10 in drafts.
FIG_W, FIG_H = 3.15, 3.0
FIG_DPI = 300
FS_AXIS_LABEL = 9
FS_TICK = 8
FS_POINT_LABEL = 6
FS_ANNOT = 8
# How far a label sits off its point, measured perpendicular to the trend line.
LABEL_OFFSET_PT = 13
# Pairs whose label goes on the opposite side from the one its residual picks,
# because the residual side is too crowded. Keyed by experiment, then Pair_Key.
FORCE_LABEL_SIDE = {
    2: {'EN_M__HI_F': 'up-left'},
}

XLABEL = "Perceptual similarity distance\nbetween talkers"
YLABEL = "Mean degree of adaptation"


# Exp 1 audio folder -> label shown on the plot. An explicit map rather than
# string substitution, since 'french_F' is a prefix of 'french_FI'. The two
# French females are numbered to tell them apart (FR_F1 / FR_F2).
DISPLAY_NAME_PER1 = {
    'NS_F': 'EN_F',
    'NS_M': 'EN_M',
    'french_F': 'FR_F1',
    'french_M': 'FR_M',
    'french_FI': 'FR_F2',
}


def display_pair(pair_key, exp):
    """Point label. Exp 1 audio folders use NS/french; show EN/FR.

    Wrapped after 'vs' so a label is ~half as wide: at 3.15 in the one-line
    form ran off the axes and adjacent labels collided.
    """
    names = pair_key.split('__')
    if exp == 1:
        names = [DISPLAY_NAME_PER1.get(n, n) for n in names]
    return f"{names[0]} vs\n{names[1]}"


def normalize_pair(s1, s2):
    return '__'.join(sorted([s1, s2]))


def pair_distances(exp):
    """Mean embedding distance per speaker pair."""
    path = os.path.join(REPO_ROOT, "per_similarity_results",
                        f"per{exp}_speaker_pair_per_sim_full.csv")
    df = pd.read_csv(path)
    df['Speaker1'] = df['S1'].str.replace(r"_[^_]+$", "", regex=True)
    df['Speaker2'] = df['S2'].str.replace(r"_[^_]+$", "", regex=True)
    avg = df.groupby(['Speaker1', 'Speaker2'])['distance'].mean().reset_index()
    avg['Pair_Key'] = avg.apply(lambda r: normalize_pair(r['Speaker1'], r['Speaker2']), axis=1)
    return avg.drop_duplicates(subset=['Pair_Key'])[['Pair_Key', 'distance']]


def legality_effects(exp):
    """Mean (legal - illegal) difference per speaker pair, keyed on audio names."""
    if exp == 1:
        df = pd.read_csv(os.path.join(REPO_ROOT, "experiment_data",
                                      "per1_speaker_legality_diffs.csv"))
        df = df[['Speaker_1', 'Speaker_2', 'avg_diff']].dropna()
        df['S1_audio'] = df['Speaker_1'].map(SPEAKER_MAP_PER1)
        df['S2_audio'] = df['Speaker_2'].map(SPEAKER_MAP_PER1)
        df = df.dropna(subset=['S1_audio', 'S2_audio'])
        df['Pair_Key'] = df.apply(lambda r: normalize_pair(r['S1_audio'], r['S2_audio']), axis=1)
        df = df.rename(columns={'avg_diff': 'diff'})
    else:
        df = pd.read_csv(os.path.join(REPO_ROOT, "experiment_data",
                                      "per2_speaker_legality_diffs.csv"))
        split = df['Speaker Combination'].str.split('/', expand=True)
        df['Pair_Key'] = [normalize_pair(a, b) for a, b in zip(split[0], split[1])]
        df = df.rename(columns={'Mean (Legal - Illegal) Difference': 'diff'})
    return df.drop_duplicates(subset=['Pair_Key'])[['Pair_Key', 'diff']]


def merged_for(exp):
    return pair_distances(exp).merge(legality_effects(exp), on='Pair_Key', how='inner')



def _offset_direction(ax, merged):
    """Up-left unit vector (display coords) perpendicular to the trend line.

    Labels used to be offset horizontally, which put them right on top of the
    fitted line, since the points themselves hug it. Offsetting perpendicular
    to the line instead keeps the two clear zones (up-left / down-right).
    Returns the fit too, so callers can pick a side from each residual.
    """
    slope, intercept = np.polyfit(merged['distance'], merged['diff'], 1)
    x0, x1 = ax.get_xlim()
    p0 = np.array(ax.transData.transform((x0, slope * x0 + intercept)))
    p1 = np.array(ax.transData.transform((x1, slope * x1 + intercept)))
    along = (p1 - p0) / np.linalg.norm(p1 - p0)
    perp = np.array([-along[1], along[0]])
    if perp[0] > 0:
        perp = -perp                               # the up-left side
    return perp, slope, intercept


def _place(text, ax, point_disp, direction, offset_px):
    """Anchor a label offset from its point along `direction`."""
    pos = point_disp + direction * offset_px
    text.set_ha('right' if direction[0] < 0 else 'left')
    text.set_va('bottom' if direction[1] > 0 else 'top')
    text.set_position(ax.transData.inverted().transform(pos))


def _flip_out_of_bounds(fig, ax, texts, points, directions, offset_px, tol_px=12.0):
    """Send labels that overhang the axes to the other side of the line.

    Side is normally chosen from the residual, but a point near an edge can
    have its own side run off the axes; that one takes the opposite zone,
    which is still clear of the trend line.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, (text, (x, y)) in enumerate(zip(texts, points)):
        bb = text.get_window_extent(renderer=renderer)
        # Only flip for an overhang too big to absorb; _clamp_to_axes can nudge
        # a small one back in, which keeps the label on its point's own side.
        overhang = max(ax.bbox.x0 - bb.x0, bb.x1 - ax.bbox.x1,
                       ax.bbox.y0 - bb.y0, bb.y1 - ax.bbox.y1)
        if overhang <= tol_px:
            continue
        directions[i] = -directions[i]
        _place(text, ax, np.array(ax.transData.transform((x, y))),
               directions[i], offset_px)


def _spread_labels(fig, ax, texts, max_iter=250, step_px=1.0, gap_px=1.2):
    """Nudge overlapping point labels apart vertically.

    At 3.15 in several pairs sit close enough that their labels collide
    (e.g. Exp 1 french_FI/french_M vs french_F/french_M). Uses real rendered
    extents rather than guessed offsets, so it adapts if the data change.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(max_iter):
        boxes = [t.get_window_extent(renderer=renderer).expanded(1.0, 1.0)
                 .padded(gap_px) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if not boxes[i].overlaps(boxes[j]):
                    continue
                moved = True
                up = step_px if boxes[i].y0 >= boxes[j].y0 else -step_px
                for t, delta in ((texts[i], up), (texts[j], -up)):
                    x, y = t.get_position()
                    px, py = ax.transData.transform((x, y))
                    _, new_y = ax.transData.inverted().transform((px, py + delta))
                    t.set_position((x, new_y))
        _clamp_to_axes(ax, texts, renderer)
        if not moved:
            return
        fig.canvas.draw()


def _clamp_to_axes(ax, texts, renderer, inset_px=2.0):
    """Keep nudged labels inside the axes.

    _spread_labels pushes purely vertically, so without this the outermost
    labels walk off the top and bottom of the figure.
    """
    for t in texts:
        bb = t.get_window_extent(renderer=renderer)
        dy = dx = 0.0
        if bb.y1 > ax.bbox.y1 - inset_px:
            dy = (ax.bbox.y1 - inset_px) - bb.y1
        elif bb.y0 < ax.bbox.y0 + inset_px:
            dy = (ax.bbox.y0 + inset_px) - bb.y0
        # x too: a label flipped to the down-right side can otherwise run past
        # the right spine and sit on the figure edge.
        if bb.x1 > ax.bbox.x1 - inset_px:
            dx = (ax.bbox.x1 - inset_px) - bb.x1
        elif bb.x0 < ax.bbox.x0 + inset_px:
            dx = (ax.bbox.x0 + inset_px) - bb.x0
        if not dx and not dy:
            continue
        x, y = t.get_position()
        px, py = ax.transData.transform((x, y))
        t.set_position(ax.transData.inverted().transform((px + dx, py + dy)))



def _draw_leaders(fig, ax, texts, points, color='0.35', lw=0.6,
                  pad_text_pt=1.0, pad_point_pt=1.5, min_len_pt=1.5):
    """Connect each settled label to its point with a short grey line.

    Drawn as real Line2D artists rather than annotate() arrows: the label's
    white-stroke path effect is applied to the whole Annotation and wipes out
    an attached arrow, and an arrow anchored before _spread_labels runs ends
    up measured from the pre-nudge position.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    ylim = ax.get_ylim()
    # get_window_extent works in FIGURE-dpi pixels (100), not the savefig dpi,
    # so the pads are specified in points and converted here.
    to_px = fig.dpi / 72.0
    pad_text_px = pad_text_pt * to_px
    pad_point_px = pad_point_pt * to_px
    min_len_px = min_len_pt * to_px
    for text, (x, y) in zip(texts, points):
        bb = text.get_window_extent(renderer=renderer)
        px, py = ax.transData.transform((x, y))
        # nearest point on the label's box, so the line meets the edge facing it
        nx = min(max(px, bb.x0), bb.x1)
        ny = min(max(py, bb.y0), bb.y1)
        dx, dy = nx - px, ny - py
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= pad_text_px + pad_point_px + min_len_px:
            continue  # label already touches its point; a stub would be noise
        ux, uy = dx / dist, dy / dist
        start = inv.transform((px + ux * pad_point_px, py + uy * pad_point_px))
        end = inv.transform((nx - ux * pad_text_px, ny - uy * pad_text_px))
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw,
                zorder=6, solid_capstyle='round')
    ax.set_ylim(ylim)   # ax.plot would otherwise re-autoscale y


def plot_experiment(exp):
    merged = merged_for(exp)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sns.regplot(
        data=merged, x='distance', y='diff', ax=ax,
        # 'linewidths', not 'linewidth': regplot forwards scatter_kws to
        # plt.scatter alongside its own 'linewidths' default, and matplotlib
        # rejects both aliases at once.
        scatter_kws={'s': 28, 'color': COLORS[exp], 'edgecolor': 'black',
                     'linewidths': 0.5, 'alpha': 0.8},
        line_kws={'color': COLORS[exp], 'linewidth': 1.2, 'linestyle': '--'},
        ci=None)

    lo, hi = merged['distance'].min(), merged['distance'].max()
    pad = 0.34 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.margins(y=0.18)

    # Leader lines, so labels can be pushed well clear of the cloud and still
    # be unambiguous. Without them the labels have to hug their points and
    # bunch up at 3.15 in.
    texts, points, keys = [], [], []
    for _, row in merged.iterrows():
        points.append((row['distance'], row['diff']))
        keys.append(row['Pair_Key'])
        texts.append(ax.text(
            row['distance'], row['diff'], display_pair(row['Pair_Key'], exp),
            fontsize=FS_POINT_LABEL, fontweight='bold', zorder=8,
            # White stroke around the glyphs instead of a bbox: keeps the text
            # readable where the trend line runs under it, with no visible box.
            path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]))

    r_val = merged['distance'].corr(merged['diff'])
    r2_val = r_val ** 2
    ax.set_xlabel(XLABEL, fontsize=FS_AXIS_LABEL, fontweight='bold')
    ax.set_ylabel(YLABEL, fontsize=FS_AXIS_LABEL, fontweight='bold')
    ax.tick_params(axis='both', labelsize=FS_TICK)
    # ".05" rather than "0.050": two decimals, no leading zero, so the y axis
    # eats less horizontal space. 0.05 steps keep every tick exact at 2 dp.
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _pos: f"{v:.2f}".replace("0.", ".", 1)))
    ax.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.04, 0.96, f"R\u00b2 = {r2_val:.3f}",
            transform=ax.transAxes, fontsize=FS_ANNOT, fontweight='bold', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                      edgecolor='gray', linewidth=0.5, alpha=0.9))
    fig.tight_layout(pad=0.3)
    up_left, slope, intercept = _offset_direction(ax, merged)
    offset_px = LABEL_OFFSET_PT * fig.dpi / 72.0
    # Put each label on the SAME side of the trend line as its own point, so a
    # point sitting below/right of the line gets its label below/right too.
    forced = FORCE_LABEL_SIDE.get(exp, {})
    directions = []
    for key, (x, y) in zip(keys, points):
        side = forced.get(key)
        if side is None:
            side = 'up-left' if y > slope * x + intercept else 'down-right'
        directions.append(up_left if side == 'up-left' else -up_left)
    for text, (x, y), direction in zip(texts, points, directions):
        _place(text, ax, np.array(ax.transData.transform((x, y))),
               direction, offset_px)
    _flip_out_of_bounds(fig, ax, texts, points, directions, offset_px)
    _spread_labels(fig, ax, texts)
    _draw_leaders(fig, ax, texts, points)
    out = os.path.join(PLOTS_DIR, f"scatter_distance_vs_false_alarm_exp{exp}.png")
    fig.savefig(out, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Exp {exp}: R\u00b2 = {r2_val:.3f} (r = {r_val:.3f}), N = {len(merged)} pairs "
          f"-> {os.path.relpath(out, REPO_ROOT)}")


def plot_combined():
    frames = []
    for exp in (1, 2):
        m = merged_for(exp)
        m['Experiment'] = f'Exp {exp}'
        frames.append(m)
    merged = pd.concat(frames, ignore_index=True)

    plt.figure(figsize=(FIG_W, FIG_H))
    palette = {'Exp 1': COLORS[1], 'Exp 2': COLORS[2]}
    # Marker/line/font sizes scaled down from the old 12 x 8 draft: at 3.15 in
    # the previous s=150 points and 12-15 pt text covered most of the axes.
    sns.scatterplot(data=merged, x='distance', y='diff', hue='Experiment',
                    style='Experiment', s=28, palette=palette,
                    edgecolor='black', linewidth=0.5, alpha=0.8, zorder=5)

    for exp_label in merged['Experiment'].unique():
        sns.regplot(data=merged[merged['Experiment'] == exp_label],
                    x='distance', y='diff', scatter=False, color=palette[exp_label],
                    line_kws={'linewidth': 1.2, 'label': f'{exp_label} Trend'}, ci=None)

    sns.regplot(data=merged, x='distance', y='diff', scatter=False,
                line_kws={'color': 'gray', 'linewidth': 1.2, 'linestyle': '--',
                          'label': 'Overall Trend', 'alpha': 0.6}, ci=None)

    for _, row in merged.iterrows():
        plt.text(row['distance'] + 0.05, row['diff'],
                 row['Pair_Key'].replace('__', ' vs '),
                 fontsize=FS_POINT_LABEL, ha='left', va='center',
                 bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.5))

    overall_r = merged['distance'].corr(merged['diff'])
    rs = {e: merged[merged['Experiment'] == e]['distance'].corr(
              merged[merged['Experiment'] == e]['diff'])
          for e in ('Exp 1', 'Exp 2')}

    # Title wrapped over three lines: at 3.15 in the one-line form ran well past
    # both figure edges.
    plt.title('Embedding Distance vs.\nPerceptual Legality Effect\n(Combined Experiments)',
              fontsize=FS_AXIS_LABEL, pad=6)
    plt.xlabel('Average Embedding Distance (HuBERT + DTW)', fontsize=FS_AXIS_LABEL)
    plt.ylabel('Mean (Legal - Illegal) Difference', fontsize=FS_AXIS_LABEL)
    plt.tick_params(axis='both', labelsize=FS_TICK)
    plt.legend(title='Experiment', title_fontsize=FS_TICK, fontsize=FS_POINT_LABEL,
               loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6, linewidth=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.text(0.05, 0.95,
             f"Overall r = {overall_r:.3f}\nExp 1 r = {rs['Exp 1']:.3f}\n"
             f"Exp 2 r = {rs['Exp 2']:.3f}\nN = {len(merged)}",
             transform=plt.gca().transAxes, fontsize=FS_ANNOT, fontweight='bold', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                       edgecolor='gray', linewidth=0.5, alpha=0.9))
    plt.tight_layout(pad=0.3)
    out = os.path.join(PLOTS_DIR, "scatter_distance_vs_false_alarm_combined.png")
    plt.savefig(out, dpi=FIG_DPI)
    plt.close()
    print(f"Combined: overall r = {overall_r:.3f}, Exp 1 r = {rs['Exp 1']:.3f}, "
          f"Exp 2 r = {rs['Exp 2']:.3f}, N = {len(merged)} "
          f"-> {os.path.relpath(out, REPO_ROOT)}")


if __name__ == "__main__":
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_experiment(1)
    plot_experiment(2)
    plot_combined()
