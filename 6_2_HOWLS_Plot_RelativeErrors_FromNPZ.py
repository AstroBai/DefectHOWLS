import glob
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

outdir = "outfiles"
candidates = glob.glob(os.path.join(outdir, "relative_errors_*.npz"))
if not candidates:
    raise FileNotFoundError(f"No relative_errors_*.npz found in {outdir}")

main_candidates = [p for p in candidates if not p.endswith("_nosub.npz")]
nosub_candidates = [p for p in candidates if p.endswith("_nosub.npz")]

main_path = max(main_candidates, key=os.path.getmtime) if main_candidates else max(candidates, key=os.path.getmtime)
nosub_path = max(nosub_candidates, key=os.path.getmtime) if nosub_candidates else None

with np.load(main_path, allow_pickle=True) as data:
    our_labels = []
    for x in data["labels"]:
        if isinstance(x, bytes):
            our_labels.append(x.decode("utf-8"))
        else:
            our_labels.append(str(x))
    vals_om_our = np.asarray(data["rel_err_om"], dtype=float)
    vals_s8_our = np.asarray(data["rel_err_s8"], dtype=float)

vals_om_nosub = None
vals_s8_nosub = None
if nosub_path is not None:
    with np.load(nosub_path, allow_pickle=True) as data:
        nosub_labels = []
        for x in data["labels"]:
            if isinstance(x, bytes):
                nosub_labels.append(x.decode("utf-8"))
            else:
                nosub_labels.append(str(x))
        vals_om_nosub = np.asarray(data["rel_err_om"], dtype=float)
        vals_s8_nosub = np.asarray(data["rel_err_s8"], dtype=float)
    if nosub_labels != our_labels:
        raise ValueError("labels in nosub NPZ do not match labels in main NPZ")

external_data = [
    (r"$\kappa$-2PCF", 1.88, 1.13),
    (r"$\kappa$-PDF", 0.70, 0.42),
    ("Peaks", 0.70, 0.45),
    ("MFs", 0.74, 0.35),
    ("BNs", 1.27, 0.72),
    ("Pers. BNs", 0.60, 0.32),
    ("Pers. heat.", 0.86, 0.56),
    ("ST", 0.63, 0.39),
]

our_data = []
for lbl, om, s8 in zip(our_labels, vals_om_our, vals_s8_our):
    our_data.append((lbl, om, s8))

all_data = our_data + external_data

labels = [d[0] for d in all_data]
main_vals_om = np.array([d[1] for d in all_data], dtype=float)
main_vals_s8 = np.array([d[2] for d in all_data], dtype=float)
n_our = len(our_data)
group_gap = 0.8
y_pos = np.arange(len(all_data), dtype=float)
y_pos[n_our:] += group_gap

vals_for_range_om = main_vals_om.copy()
vals_for_range_s8 = main_vals_s8.copy()
if vals_om_nosub is not None and vals_s8_nosub is not None:
    vals_for_range_om = np.concatenate([vals_for_range_om, vals_om_nosub])
    vals_for_range_s8 = np.concatenate([vals_for_range_s8, vals_s8_nosub])

min_om, max_om = vals_for_range_om.min(), vals_for_range_om.max()
min_s8, max_s8 = vals_for_range_s8.min(), vals_for_range_s8.max()
denom_om = max(max_om - min_om, 1e-12)
denom_s8 = max(max_s8 - min_s8, 1e-12)

fig, ax = plt.subplots(figsize=(8.5, 4))

defects_idx = 2 if len(our_data) > 2 else None
for i, (om, s8) in enumerate(zip(main_vals_om, main_vals_s8)):
    y = y_pos[i]
    alpha_om = 0.8 - 0.6 * (om - min_om) / denom_om
    alpha_s8 = 0.8 - 0.6 * (s8 - min_s8) / denom_s8
    color_om = cm.Blues(0.3 + 0.7 * alpha_om)
    color_s8 = cm.Oranges(0.3 + 0.7 * alpha_s8)

    marker = "*" if defects_idx is not None and i == defects_idx else "s"
    msize = 10 if marker == "*" else 6

    ax.plot([0, -om], [y, y], color=color_om, linewidth=2, marker=marker,
            markevery=[1], markersize=msize, markerfacecolor=color_om)
    ax.plot([0, s8], [y, y], color=color_s8, linewidth=2, marker=marker,
            markevery=[1], markersize=msize, markerfacecolor=color_s8)

    if vals_om_nosub is not None and vals_s8_nosub is not None and i < len(our_data):
        y_nosub = y + 0.3
        om_nosub = vals_om_nosub[i]
        s8_nosub = vals_s8_nosub[i]

        alpha_om_n = 0.8 - 0.6 * (om_nosub - min_om) / denom_om
        alpha_s8_n = 0.8 - 0.6 * (s8_nosub - min_s8) / denom_s8
        color_om_n = cm.Blues(0.3 + 0.7 * alpha_om_n)
        color_s8_n = cm.Oranges(0.3 + 0.7 * alpha_s8_n)

        ax.plot([0, -om_nosub], [y_nosub, y_nosub], color=color_om_n, marker=marker, markersize=msize*0.5, markerfacecolor=color_om_n, markevery=[1],
                linewidth=1, linestyle="--", alpha=0.95)
        ax.plot([0, s8_nosub], [y_nosub, y_nosub], color=color_s8_n, marker=marker, markersize=msize*0.5, markerfacecolor=color_s8_n, markevery=[1],
                linewidth=1, linestyle="--", alpha=0.95)

    if i == 0:
        ax.text(-om - 0.1, y, f"{om:.2f}%", va="center", ha="right", fontsize=9, color=color_om)
        ax.text(s8 + 0.1, y, f"{s8:.2f}%", va="center", ha="left", fontsize=9, color=color_s8)
    else:
        ratio_om = main_vals_om[0] / om if om != 0 else np.nan
        ratio_s8 = main_vals_s8[0] / s8 if s8 != 0 else np.nan
        ax.text(-om - 0.1, y, f"{ratio_om:.1f}x", va="center", ha="right", fontsize=9, color=color_om)
        ax.text(s8 + 0.1, y, f"{ratio_s8:.1f}x", va="center", ha="left", fontsize=9, color=color_s8)

ax.plot([], [], color=cm.Blues(0.8), linewidth=2, marker="s", markersize=6,
        label=r"$\delta \Omega_{\rm m} / \Omega_{\rm m} [\%]$")
ax.plot([], [], color=cm.Oranges(0.8), linewidth=2, marker="s", markersize=6,
        label=r"$\delta \sigma_8 / \sigma_8 [\%]$")
# if vals_om_nosub is not None and vals_s8_nosub is not None:
#     ax.plot([], [], color="gray", linewidth=1.8, linestyle="--", label="nosub")

this_work_center = np.mean(y_pos[:n_our])
euclid_center = np.mean(y_pos[n_our:])

ax.text(1.02, this_work_center, "This Work", transform=ax.get_yaxis_transform(),
    va="center", ha="left", rotation=90, fontsize=9, clip_on=False)
ax.text(1.02, euclid_center, "Euclid HOWLS", transform=ax.get_yaxis_transform(),
    va="center", ha="left", rotation=90, fontsize=9, clip_on=False)

sep_index = 0.5 * (y_pos[n_our - 1] + y_pos[n_our])
ax.axhline(y=sep_index, color="black", linestyle="-", linewidth=1)

if defects_idx is not None:
    ax.axvspan(-main_vals_om[defects_idx], main_vals_s8[defects_idx], color="gray", alpha=0.3)

ax.set_xlim(-max(vals_for_range_om) * 1.4, max(vals_for_range_s8) * 1.4)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()

ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{abs(x):.1f}"))

ax.set_xlabel(r"Relative Precision [%]")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
ax.axvline(x=0, color="black", linewidth=0.8)

plt.tight_layout(rect=[0, 0, 0.95, 1])
os.makedirs("outfiles", exist_ok=True)
out_png = "outfiles/1sigma_bars_from_npz.png"
plt.savefig(out_png, dpi=300)
plt.close()

print(f"Loaded NPZ: {main_path}")
if nosub_path is not None:
    print(f"Loaded nosub NPZ: {nosub_path}")
print(f"Saved figure: {out_png}")