"""
simulate_slalom.py
Simuleert hoe het vaartuig het slalom-parcour aflegt met een Pure Pursuit
controller via de padplanning_wrapper wrapper, en tekent alle boei-varianten.
"""

import math
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import your new wrapper function
from padplanning_slalom import padplanning_wrapper, Position

# ── Vaartuig-/controllerparameters ────────────────────────────────────────────
SPEED      = 3.0     # m/s   constante vaarsnelheid
DT         = 0.05    # s     simulatie-tijdstap
MAX_YAW    = 0.72    # rad/s maximale giersnelheid (draailimiet)
LOOKAHEAD  = 8.0     # m     Pure Pursuit vooruitkijkafstand
GOAL_TOL   = 3.0     # m     afstand tot laatste waypoint = klaar
MAX_TIME   = 120.0   # s     veiligheidslimiet
CLEARANCE_X = 5.0    # m     Extra clearance padding 'x'

# ── Lokale ENU-conversie (gedeeld referentiepunt voor de plots) ───────────────
R_EARTH = 6_371_000.0
REF_LAT, REF_LON = 51.920000, 4.350000
_MPD_LON = math.cos(math.radians(REF_LAT)) * math.pi / 180 * R_EARTH
_MPD_LAT = math.pi / 180 * R_EARTH


def to_m(lon: float, lat: float) -> Tuple[float, float]:
    return ((lon - REF_LON) * _MPD_LON, (lat - REF_LAT) * _MPD_LAT)


def offset(dx_m: float, dy_m: float) -> Tuple[float, float]:
    """Geeft (lat, lon) terug om te matchen met de invoerconventie."""
    return (REF_LAT + dy_m / _MPD_LAT, REF_LON + dx_m / _MPD_LON)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def pure_pursuit(
    path_xy: List[Tuple[float, float]],
    start_xy: Tuple[float, float],
    start_hdg_deg: float,
) -> List[Tuple[float, float]]:
    x, y = start_xy
    hdg = math.radians(start_hdg_deg)
    wp = np.array(path_xy, dtype=float)
    n = len(wp)
    idx = 0
    trail = [(x, y)]
    t = 0.0

    goal = wp[-1]
    while t < MAX_TIME:
        while idx < n - 1 and math.hypot(wp[idx][0] - x, wp[idx][1] - y) < LOOKAHEAD * 0.4:
            idx += 1

        target = wp[min(idx, n - 1)]
        for i in range(idx, n):
            if math.hypot(wp[i][0] - x, wp[i][1] - y) >= LOOKAHEAD:
                target = wp[i]
                break
            target = wp[i]

        dx, dy = target[0] - x, target[1] - y
        dist = math.hypot(dx, dy)
        alpha = _wrap(math.atan2(dx, dy) - hdg)
        yaw = (2.0 * math.sin(alpha) / dist) * SPEED if dist > 0.5 else 0.0
        yaw = max(-MAX_YAW, min(MAX_YAW, yaw))

        hdg += yaw * DT
        x += SPEED * math.sin(hdg) * DT
        y += SPEED * math.cos(hdg) * DT
        t += DT
        trail.append((x, y))

        if idx >= n - 2 and math.hypot(goal[0] - x, goal[1] - y) < GOAL_TOL:
            break

    return trail


# ── Plotstijl ──────────────────────────────────────────────────────────────────
C = dict(
    bg="#0f1117", panel="#171a24", grid="#21253a",
    plan="#3a5a9a", trail="#4a9eff", arc="#9d7fe8",
    b1="#ff4f6a", b2="#3dd68c", boat="#f5a623", entry="white", muted="#55596e",
    text="#c8cad4",
)


def _style_axis(ax):
    ax.set_facecolor(C["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])
    ax.grid(True, color=C["grid"], lw=0.5, ls="--", alpha=0.6)
    ax.tick_params(colors=C["muted"], labelsize=7.5)
    ax.set_aspect("equal")


def _draw_buoy_clusters(ax, buoy_positions):
    """Berekent het gemiddelde centrum en plot alle mogelijke posities."""
    labels = ["B1", "B2"]
    colors = [C["b1"], C["b2"]]
    
    for idx, cluster in enumerate(buoy_positions):
        col = colors[idx]
        lbl = labels[idx]
        
        # Bereken centrum voor de plot visualisatie
        avg_lat = sum(p[0] for p in cluster) / len(cluster)
        avg_lon = sum(p[1] for p in cluster) / len(cluster)
        cx, cy = to_m(avg_lon, avg_lat)
        
        # Plot alle losse mogelijke posities (semi-transparant)
        for lat, lon in cluster:
            px, py = to_m(lon, lat)
            ax.scatter([px], [py], s=40, color=col, alpha=0.4, zorder=6, edgecolors="white", lw=0.5)
            
        # Plot het berekende centrum
        ax.scatter([cx], [cy], s=150, color=col, zorder=7, edgecolors="white", linewidths=0.9, marker="D")
        ax.text(cx, cy + 4, lbl, color=col, fontsize=9, ha="center", va="bottom", fontweight="bold")


def make_plots(buoy_positions, scenarios, out_course, out_sim):
    # ── Figuur 1: gepland parcour per state ──────────────────────────────────
    fig1, axes = plt.subplots(1, len(scenarios), figsize=(16, 4.2))
    fig1.patch.set_facecolor(C["bg"])
    
    for ax, (label, b_latlon, hdg, state) in zip(axes, scenarios):
        # Aanroep via de nieuwe wrapper: geeft [(lat, lon, speed), ...] terug
        pad = padplanning_wrapper(buoy_positions, CLEARANCE_X, state)
        
        # wrapper output is (lat, lon, speed) -> omzetten naar meter-coördinaten (oost, noord)
        px, py = zip(*[to_m(lon, lat) for lat, lon, speed in pad])
        
        _style_axis(ax)
        ax.plot(px, py, color=C["trail"], lw=2.0, solid_capstyle="round", zorder=3)
        
        step = max(2, len(px) // 9)
        for i in range(0, len(px) - step, step):
            ddx, ddy = px[i + step] - px[i], py[i + step] - py[i]
            if math.hypot(ddx, ddy) > 0.3:
                ax.annotate("", xy=(px[i] + ddx * 0.6, py[i] + ddy * 0.6),
                            xytext=(px[i] + ddx * 0.4, py[i] + ddy * 0.4),
                            arrowprops=dict(arrowstyle="->", color=C["trail"],
                                            lw=1.2, mutation_scale=10), zorder=5)
                                            
        _draw_buoy_clusters(ax, buoy_positions)
        
        bx, by = to_m(b_latlon[1], b_latlon[0])
        ax.scatter([bx], [by], s=130, color=C["boat"], marker=(3, 0, -hdg),
                   zorder=8, edgecolors="white", linewidths=0.9)
        ax.scatter([px[0]], [py[0]], s=60, color=C["entry"], zorder=9,
                   edgecolors=C["trail"], linewidths=1.5)
                   
        length = sum(math.hypot(px[i + 1] - px[i], py[i + 1] - py[i]) for i in range(len(px) - 1))
        ax.text(0.97, 0.03, f"pad: {length:.0f} m", transform=ax.transAxes,
                color=C["muted"], fontsize=7.5, ha="right", va="bottom")
        ax.set_title(label, color=C["text"], fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("oost (m)", color=C["muted"], fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel("noord (m)", color=C["muted"], fontsize=8.5)
        ax.set_xlim(-80, 50)
        ax.set_ylim(-22, 22)

    fig1.suptitle("Slalom-parcour  ·  gepland pad per state (via padplanning_wrapper wrapper)",
                  color=C["text"], fontsize=12.5, fontweight="bold", y=1.02)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(out_course, dpi=160, bbox_inches="tight", facecolor=C["bg"], edgecolor="none")

    # ── Figuur 2: gevaren baan (Pure Pursuit) vanaf START ────────────────────
    label, b_latlon, hdg, state = scenarios[0]
    pad = padplanning_wrapper(buoy_positions, CLEARANCE_X, state)
    path_xy = [to_m(lon, lat) for lat, lon, speed in pad]
    
    trail = pure_pursuit(path_xy, to_m(b_latlon[1], b_latlon[0]), hdg)

    fig2, ax = plt.subplots(figsize=(11, 7))
    fig2.patch.set_facecolor(C["bg"])
    _style_axis(ax)

    px, py = zip(*path_xy)
    ax.plot(px, py, color=C["plan"], lw=1.4, ls="--", zorder=2, label="gepland pad")
    ax.scatter(px, py, s=8, color=C["plan"], zorder=2)

    tx, ty = zip(*trail)
    ax.plot(tx, ty, color=C["trail"], lw=2.4, solid_capstyle="round", zorder=4, label="gevaren baan")

    step = max(8, len(tx) // 22)
    for i in range(0, len(tx) - step, step):
        ddx, ddy = tx[i + step] - tx[i], ty[i + step] - ty[i]
        if math.hypot(ddx, ddy) > 0.2:
            ax.annotate("", xy=(tx[i] + ddx, ty[i] + ddy), xytext=(tx[i], ty[i]),
                        arrowprops=dict(arrowstyle="->", color=C["trail"],
                                        lw=1.1, mutation_scale=10), zorder=5)

    _draw_buoy_clusters(ax, buoy_positions)
    bx, by = to_m(b_latlon[1], b_latlon[0])
    ax.scatter([bx], [by], s=150, color=C["boat"], marker=(3, 0, -hdg),
               zorder=8, edgecolors="white", linewidths=1.0)
    ax.text(bx, by - 4, f"start  {hdg:.0f}°", color=C["boat"], fontsize=8, ha="center", va="top")

    travelled = sum(math.hypot(tx[i + 1] - tx[i], ty[i + 1] - ty[i]) for i in range(len(tx) - 1))
    sim_time = (len(trail) - 1) * DT
    ax.text(0.5, 0.04,
            f"gevaren: {travelled:.0f} m   ·   tijd: {sim_time:.1f} s   ·   "
            f"snelheid: {SPEED:.0f} m/s   ·   lookahead: {LOOKAHEAD:.0f} m",
            transform=ax.transAxes, color=C["muted"], fontsize=8.5, ha="center", va="bottom")

    handles = [
        Line2D([0], [0], color=C["plan"], lw=1.4, ls="--", label="gepland pad"),
        Line2D([0], [0], color=C["trail"], lw=2.4, label="gevaren baan"),
        Line2D([0], [0], color=C["b1"], lw=0, marker="D", ms=8, markeredgecolor="w", label="B1 Centrum + Wolk"),
        Line2D([0], [0], color=C["b2"], lw=0, marker="D", ms=8, markeredgecolor="w", label="B2 Centrum + Wolk"),
        Line2D([0], [0], color=C["boat"], lw=0, marker="^", ms=8, markeredgecolor="w", label="boot + koers"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5,
              facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], framealpha=1)
    ax.set_xlabel("oost (m)", color=C["muted"], fontsize=9)
    ax.set_ylabel("noord (m)", color=C["muted"], fontsize=9)
    ax.set_title("Gesimuleerde vaart  ·  Pure Pursuit volgt de padplanning_wrapper wrapper",
                 color=C["text"], fontsize=12.5, fontweight="bold", pad=8)
    fig2.tight_layout()
    fig2.savefig(out_sim, dpi=160, bbox_inches="tight", facecolor=C["bg"], edgecolor="none")

    return travelled, sim_time


if __name__ == "__main__":
    # Gedefinieerd als een lijst van MOGELIJKE posities per boei (lat, lon)
    # De posities variëren een beetje rond de -25m en +25m oost offsets
    buoy_a_variants = [
        offset(-25, 0),
        offset(-27, 2),
        offset(-23, -2)
    ]
    buoy_b_variants = [
        offset(25, 0),
        offset(23, 3),
        offset(26, -1),
        offset(27, 1)
    ]
    
    buoy_positions = [buoy_a_variants, buoy_b_variants]

    # Scenarios gedefinieerd met boot_posities als (lat, lon)
    scenarios = [
        ("START",    offset(-68, 4),  88.0, "START"),
        ("DETECT_1", offset(-40, 3),  85.0, "DETECT_1"),
        ("DETECT_2", offset(20, 8),  250.0, "DETECT_2"),
    ]
    
    travelled, t = make_plots(
        buoy_positions, scenarios,
        out_course="slalom_path.png",
        out_sim="slalom_simulation.png",
    )
    print(f"Gevaren baan vanaf START: {travelled:.0f} m in {t:.1f} s")