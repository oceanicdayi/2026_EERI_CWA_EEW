#!/usr/bin/env python3
"""
Real records of the 2025 Dapu M6.4 earthquake at five CWA stations in the Taipei Basin.

Reads SAC files with ObsPy and picks P and S on the five-station beam (the stations lie within
8 km, so their envelopes are stacked after a distance moveout correction and picked with AIC).
Single-station STA/LTA + AIC picks are kept in picks.csv for comparison. Writes:
  record_section.png      record section with alert times (for slides)
  title_strip.png         transparent single-trace strip (for title/closing slides)
  picks.csv               station, distance, P, S, peak amplitude
  records_summary.json    numbers quoted on the slides and the website
  strip_meta.json         positions of P/S/alert on the title strip (13.333 in wide)
  dapu_records.js         compact min/max envelopes for the interactive website

Usage:
  python dapu_real_records_obspy.py ["/path/to/*.SAC"] [output_dir]
"""
import base64
import json
import os
import sys

import numpy as np
from obspy import UTCDateTime, read
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.filter import envelope
from obspy.signal.trigger import aic_simple, recursive_sta_lta, trigger_onset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

SAC_GLOB = sys.argv[1] if len(sys.argv) > 1 else "*.SAC"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)

ORIGIN = UTCDateTime("2025-01-20T16:17:26.9")  # CWA catalog: 2025-01-21 00:17:26.9 (UTC+8)
EPI_LAT, EPI_LON = 23.22, 120.55                # CWA catalog epicenter
ALERT1 = 7.9                                    # first alert, seconds after origin
PHONE = ALERT1 + 5.5                            # plus cell-broadcast delivery delay

if any(f.name == "Carlito" for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = "Carlito"     # metric-compatible with Calibri


def first_onset(cft, t, thr_on, thr_off, tmin, tmax):
    """Index of the first trigger-on between tmin and tmax (seconds after origin)."""
    for on, _off in trigger_onset(cft, thr_on, thr_off):
        if tmin < t[on] < tmax:
            return int(on)
    return None


def aic_refine(x, i0, before, after):
    """Refine a trigger with the AIC minimum in [i0 - before, i0 + after]."""
    a, b = max(i0 - before, 0), min(i0 + after, len(x))
    f = aic_simple(x[a:b])
    return a + 5 + int(np.argmin(f[5:-5]))      # skip unstable edge values


st = read(SAC_GLOB)
print(st)

rows = []
for tr in st:
    sac, df = tr.stats.sac, tr.stats.sampling_rate
    t = tr.times(reftime=ORIGIN)                # seconds after origin
    tr.data = tr.data.astype(np.float64)
    tr.data -= tr.data[t < -5].mean()           # remove the pre-event offset (raw counts)
    acc = tr.copy().filter("highpass", freq=0.05, corners=4, zerophase=True).data
    bp1 = tr.copy().filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True).data
    bp2 = tr.copy().filter("bandpass", freqmin=0.5, freqmax=5.0, corners=4, zerophase=True).data
    dist = gps2dist_azimuth(EPI_LAT, EPI_LON, float(sac.stla), float(sac.stlo))[0] / 1000.0

    # Single-station picks (diagnostic only): recursive STA/LTA refined with AIC. The emergent
    # regional S is ambiguous on one horizontal component, so the picks used are from the beam.
    iP0 = first_onset(recursive_sta_lta(bp1, int(0.5 * df), int(10 * df)), t, 4.0, 1.5, 20, 60)
    tP_sta = t[aic_refine(bp1, iP0, int(3 * df), int(1 * df))] if iP0 is not None else np.nan
    iS0 = None if iP0 is None else first_onset(recursive_sta_lta(bp2, int(1 * df), int(10 * df)), t, 2.5, 1.2, tP_sta + 20, tP_sta + 40)
    tS_sta = t[aic_refine(bp2, iS0, int(1.5 * df), int(0.5 * df))] if iS0 is not None else np.nan

    win = (t > 0) & (t < 300)
    k = int(np.argmax(np.abs(acc[win])))
    peak, tpk = float(np.abs(acc[win])[k]), float(t[win][k])
    noise = float(acc[(t > -60) & (t < -5)].std())
    rows.append(dict(id=tr.id, sta=tr.stats.station, lat=round(float(sac.stla), 4), lon=round(float(sac.stlo), 4),
                     dist=round(dist, 1), tP_sta=round(float(tP_sta), 2), tS_sta=round(float(tS_sta), 2), peak=peak,
                     tpk=round(tpk, 1), snr=peak / noise, t=t, acc=acc, bp1=bp1, bp2=bp2))

rows.sort(key=lambda r: r["dist"])

# ---------- beam picks: stack moveout-corrected envelopes, pick once with AIC ----------
dref = float(np.median([r["dist"] for r in rows]))
tc = np.arange(-60.0, 150.0, 0.01)


def beam(key, nwin, vel, norm):
    out_ = np.zeros_like(tc)
    for r in rows:
        e = np.convolve(envelope(r[key]), np.ones(nwin) / nwin, mode="same")
        e = e / np.median(e[(r["t"] > norm[0]) & (r["t"] < norm[1])])
        out_ += np.interp(tc + (r["dist"] - dref) / vel, r["t"], e)
    return out_ / len(rows)


bP = beam("bp1", 50, 6.9, (-60, -5))                  # 1-10 Hz, normalized by pre-event noise
i0 = int(np.where((tc > 20) & (bP > 3.0))[0][0])
tPb = float(tc[aic_refine(bP, i0, 300, 100)])
bS = beam("bp2", 100, 3.77, (tPb + 5, tPb + 15))      # 0.5-5 Hz, normalized by early P coda
a0, a1 = np.searchsorted(tc, tPb + 15), np.searchsorted(tc, tPb + 40)
tSb = float(tc[a0 + 5 + int(np.argmin(aic_simple(bS[a0:a1])[5:-5]))])
print(f"beam picks at {dref:.1f} km: P {tPb:.2f} s, S {tSb:.2f} s, S-P {tSb - tPb:.1f} s")
for r in rows:
    r["tP"] = round(tPb + (r["dist"] - dref) / 6.9, 2)
    r["tS"] = round(tSb + (r["dist"] - dref) / 3.77, 2)
with open(os.path.join(OUT, "picks.csv"), "w") as f:
    f.write("station,lat,lon,dist_km,P_s,S_s,S_minus_P_s,peak_counts,t_peak_s,snr,P_single_station_s,S_single_station_s\n")
    for r in rows:
        f.write(f"{r['id']},{r['lat']},{r['lon']},{r['dist']},{r['tP']},{r['tS']},{r['tS'] - r['tP']:.2f},"
                f"{r['peak']:.0f},{r['tpk']},{r['snr']:.0f},{r['tP_sta']},{r['tS_sta']}\n")
        print(f"{r['id']:16s} {r['dist']:6.1f} km  P {r['tP']:6.2f} s  S {r['tS']:6.2f} s  "
              f"peak {r['peak']:7.0f} counts at {r['tpk']:5.1f} s  SNR {r['snr']:4.0f}")

medP = float(np.median([r["tP"] for r in rows]))
medS = float(np.median([r["tS"] for r in rows]))
peaks = [r["peak"] for r in rows]
sep = max(gps2dist_azimuth(a["lat"], a["lon"], b["lat"], b["lon"])[0] / 1000 for a in rows for b in rows)
best = max(rows, key=lambda r: r["snr"])
summary = dict(medP=round(medP, 1), medS=round(medS, 1), warn=round(medS - ALERT1), warnPhone=round(medS - PHONE),
               pBefore=round(medP - ALERT1), dmin=round(min(r["dist"] for r in rows)), dmax=round(max(r["dist"] for r in rows)),
               pmin=round(min(r["tP"] for r in rows), 1), pmax=round(max(r["tP"] for r in rows), 1),
               smin=round(min(r["tS"] for r in rows), 1), smax=round(max(r["tS"] for r in rows), 1),
               ratio=round(max(peaks) / min(peaks), 1), sep=round(sep), tpkmin=round(min(r["tpk"] for r in rows)),
               tpkmax=round(max(r["tpk"] for r in rows)), hero=best["sta"], heroDist=round(best["dist"]),
               heroP=round(best["tP"], 1), heroS=round(best["tS"], 1))
json.dump(summary, open(os.path.join(OUT, "records_summary.json"), "w"), indent=1)
print("summary:", summary)

# ---------- record section (slides) ----------
T0, T1, n = -2, 100, len(rows)
fig, ax = plt.subplots(figsize=(10, 6.3), dpi=230)
for k, r in enumerate(rows):
    y0 = n - 1 - k
    m = (r["t"] >= T0) & (r["t"] <= T1)
    tt, a = r["t"][m], r["acc"][m] / r["peak"] * 0.46 + y0
    ax.add_patch(Rectangle((ALERT1, y0 - 0.46), r["tS"] - ALERT1, 0.92, color="#2A9D8F", alpha=0.09, lw=0))
    for lo, hi, c in [(T0, r["tP"], "#8A9BAA"), (r["tP"], r["tS"], "#2B7BB9"), (r["tS"], T1, "#E4572E")]:
        s = (tt >= lo) & (tt <= hi)
        ax.plot(tt[s], a[s], color=c, lw=0.55)
    for tpick, c in [(r["tP"], "#2B7BB9"), (r["tS"], "#C73E1D")]:
        ax.plot([tpick, tpick], [y0 - 0.42, y0 + 0.42], color=c, lw=1.6)
ax.axvline(ALERT1, color="#D98E04", ls=(0, (5, 3)), lw=1.8)
ax.axvline(PHONE, color="#D98E04", ls=":", lw=1.3, alpha=0.8)
ax.text(ALERT1 - 0.6, n - 0.35, f"first alert\n{ALERT1} s", color="#9A6406", fontsize=11.5, fontweight="bold", va="bottom", ha="right")
ax.text(PHONE + 0.6, n - 0.35, f"phones\n{PHONE} s", color="#9A6406", fontsize=11, va="bottom", alpha=0.9)
ax.text(rows[0]["tP"] + 0.5, n - 0.35, "P wave", color="#2B7BB9", fontsize=12, fontweight="bold", va="bottom")
ax.text(rows[0]["tS"] + 0.5, n - 0.35, "S wave", color="#C73E1D", fontsize=12, fontweight="bold", va="bottom")
ax.annotate("", xy=(medS, n + 0.3), xytext=(ALERT1, n + 0.3), arrowprops=dict(arrowstyle="<->", color="#1D7468", lw=1.6))
ax.text((ALERT1 + medS) / 2, n + 0.38, f"≈{medS - ALERT1:.0f} s between the first alert and the S wave",
        color="#1D7468", fontsize=12.5, fontweight="bold", ha="center", va="bottom")
ax.set_yticks(range(n))
ax.set_yticklabels([f"{r['sta']}  {r['dist']:.0f} km" for r in rows][::-1], fontsize=12)
ax.set_xlim(T0, T1)
ax.set_ylim(-0.6, n + 0.95)
ax.set_xlabel("Seconds after the earthquake began (00:17:26.9 local, 21 Jan 2025)", fontsize=12)
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "record_section.png"), facecolor="white")
plt.close(fig)

# ---------- title strip: best-SNR station, -5 to 95 s ----------
S0, S1 = -5, 95
m = (best["t"] >= S0) & (best["t"] <= S1)
tt, a = best["t"][m], best["acc"][m] / best["peak"]
fig = plt.figure(figsize=(16, 1.2), dpi=220)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(S0, S1)
ax.set_ylim(-1.12, 1.12)
for lo, hi, c in [(S0, best["tP"], "#7F9BB8"), (best["tP"], best["tS"], "#5DA9E9"), (best["tS"], S1, "#E4572E")]:
    s = (tt >= lo) & (tt <= hi)
    ax.plot(tt[s], a[s], color=c, lw=0.9)
fig.savefig(os.path.join(OUT, "title_strip.png"), transparent=True)
plt.close(fig)
X = lambda s: round((s - S0) / (S1 - S0) * 13.333, 3)
json.dump(dict(sta=best["sta"], dist=best["dist"], tP=best["tP"], tS=best["tS"], xP=X(best["tP"]), xS=X(best["tS"]), xA=X(ALERT1)),
          open(os.path.join(OUT, "strip_meta.json"), "w"))

# ---------- website data: min/max envelopes, 25 bins/s, -5 to 95 s, int8 base64 ----------
W0, RATE, NB = -5, 25, 2500
pmax = max(peaks)
out = []
for r in rows:
    idx = np.floor((r["t"] - W0) * RATE).astype(int)
    ok = (idx >= 0) & (idx < NB)
    v, b = r["acc"][ok] / r["peak"], idx[ok]
    mn, mx = np.full(NB, np.inf), np.full(NB, -np.inf)
    np.minimum.at(mn, b, v)
    np.maximum.at(mx, b, v)
    enc = lambda z: base64.b64encode(np.clip(np.round(z * 127), -127, 127).astype(np.int8).tobytes()).decode()
    out.append(dict(sta=r["sta"], lat=r["lat"], lon=r["lon"], dist=r["dist"], tP=r["tP"], tS=r["tS"], tpk=r["tpk"],
                    rel=round(r["peak"] / pmax, 3), mn=enc(mn), mx=enc(mx)))
data = dict(origin="2025-01-21 00:17:26.9 local (16:17:26.9 UTC)", epi=[EPI_LAT, EPI_LON], alert1=ALERT1, alert2=9.0,
            phone=PHONE, t0=W0, rate=RATE, n=NB, hero=best["sta"], stations=out)
with open(os.path.join(OUT, "dapu_records.js"), "w") as f:
    f.write("window.DAPU=" + json.dumps(data, separators=(",", ":")) + ";\n")
print("wrote", sorted(os.listdir(OUT)))
