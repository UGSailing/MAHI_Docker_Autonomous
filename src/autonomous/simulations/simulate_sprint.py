"""
simulate_sprint.py
Simuleert hoe het vaartuig het teardrop-parcour rond één boei aflegt met een
Pure Pursuit controller en tekent het resultaat.

Aanpassingen t.o.v. simulate_slalom.py
────────────────────────────────────────
• Importeert padplanning_sprint i.p.v. padplanning_slalom.
• plan_path roept padplanning_sprint aan (geen heading_deg argument meer).
• _draw_buoy_clusters werkt dynamisch voor elk aantal boeien.
• Legenda, titels en testscenario's aangepast voor één boei.
• Plot-grenzen worden automatisch bepaald op basis van het geplande pad.
• Snelheidskleurenschaal loopt van 0 tot FAST_SPEED (i.p.v. SLOW_SPEED tot
  FAST_SPEED), zodat de volledige ramp tot stilstand zichtbaar is in kleur.
"""

import math
import sys
import os
import types
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# ── Lokale stubs zodat padplanning_sprint importeerbaar is zonder get_mqtt / config ──
_mqtt_stub = types.ModuleType("communication.get_mqtt")
_mqtt_stub.get_boat_position = lambda: {"longitude": 0.0, "latitude": 0.0, "heading": 0.0}
sys.modules["communication.get_mqtt"] = _mqtt_stub

_config_stub = types.ModuleType("config")
_config_stub.N_SLALOM_PTS               = 12
_config_stub.N_ARC_PTS                  = 12
_config_stub.FAST_SPEED                 = 5.0    # m/s
_config_stub.SLOW_SPEED                 = 2.0    # m/s
_config_stub.WAYPOINT_DISTANCE          = 2    # m
_config_stub.INTERPOLATE_USING_DISTANCE = True
_config_stub.N_RAMP_WAYPOINTS           = 8
_config_stub.RAMP_ACCELERATION          = .7
_config_stub.WAYPOINT_DT                = 1.
sys.modules.setdefault("config", _config_stub)

# Stubs moeten in sys.modules staan vóór de import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autonomous.padplanning_sprint_dokken import padplanning_buoy  # noqa: E402

# ── Lokale ENU-conversie ──────────────────────────────────────────────────────
R_EARTH = 6_371_000.0
REF_LAT, REF_LON = 51.920000, 4.350000
_MPD_LON = math.cos(math.radians(REF_LAT)) * math.pi / 180 * R_EARTH
_MPD_LAT = math.pi / 180 * R_EARTH


def to_m(lon: float, lat: float) -> Tuple[float, float]:
    """(lon, lat) → (oost_m, noord_m)"""
    return ((lon - REF_LON) * _MPD_LON, (lat - REF_LAT) * _MPD_LAT)


def offset(dx_m: float, dy_m: float) -> Tuple[float, float]:
    """Meter-offset → (lat, lon) (invoerconventie wrapper)."""
    return (REF_LAT + dy_m / _MPD_LAT, REF_LON + dx_m / _MPD_LON)


# ── Simulatie-wrapper ─────────────────────────────────────────────────────────

def plan_path(
    buoy_positions,                      # [[(lat, lon), ...]]  – één wolk
    marge: float,
    state: str,
    boat_latlon: Tuple[float, float],    # (lat, lon) huidige bootpositie
    start_latlon: Tuple[float, float],   # (lat, lon) startpositie aanrijroute
    start_heading_deg: float,
) -> List[Tuple[Tuple[float, float], float]]:
    """
    Berekent het pad via padplanning_buoy en geeft
    [((lat, lon), speed), ...] terug.
    """
    buoys_centers  = []
    buoys_max_dist = []

    for buoy_pos in buoy_positions:
        total_lat  = sum(c[0] for c in buoy_pos)
        total_lon  = sum(c[1] for c in buoy_pos)
        n          = len(buoy_pos)
        center_lat = total_lat / n
        center_lon = total_lon / n
        buoys_centers.append((center_lon, center_lat))   # (lon, lat)

        lat_rad = math.radians(center_lat)
        max_d   = 0.0
        for lat, lon in buoy_pos:
            dn = math.radians(lat - center_lat) * R_EARTH
            de = math.radians(lon - center_lon) * R_EARTH * math.cos(lat_rad)
            max_d = max(max_d, math.hypot(de, dn))
        buoys_max_dist.append(max_d)

    slalom_offset = max(buoys_max_dist) + marge

    boat_pos  = (boat_latlon[1],  boat_latlon[0])    # (lon, lat)
    start_pos = (start_latlon[1], start_latlon[0])   # (lon, lat)

    waypoints = padplanning_buoy(
        buoys_centers,
        boat_pos,
        state,
        slalom_offset,
        start_pos         = start_pos,
        start_heading_deg = start_heading_deg,
    )

    # (lon, lat) → (lat, lon), behoud speed
    return [((lat_wp, lon_wp), spd) for (lon_wp, lat_wp), spd in waypoints]


# ── Pure Pursuit simulator ────────────────────────────────────────────────────
SPEED     = 3.0    # m/s  (referentie; echte snelheid komt uit waypoints)
DT        = 0.05   # s
MAX_YAW   = 0.72   # rad/s
LOOKAHEAD = 3.0    # m
GOAL_TOL  = 3.0    # m
MAX_TIME  = 180.0  # s


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def pure_pursuit(
    path_xy:       List[Tuple[float, float]],
    path_speeds:   List[float],
    start_xy:      Tuple[float, float],
    start_hdg_deg: float,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    Volgt path_xy met Pure Pursuit.
    Geeft (trail_xy, trail_speeds) terug.
    """
    x, y   = start_xy
    hdg    = math.radians(start_hdg_deg)
    wp     = np.array(path_xy, dtype=float)
    speeds = np.array(path_speeds, dtype=float)
    n      = len(wp)
    idx    = 0
    trail_xy     = [(x, y)]
    trail_speeds = [speeds[0]]
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
        dist   = math.hypot(dx, dy)
        alpha  = _wrap(math.atan2(dx, dy) - hdg)
        yaw    = (2.0 * math.sin(alpha) / dist) * SPEED if dist > 0.5 else 0.0
        yaw    = max(-MAX_YAW, min(MAX_YAW, yaw))

        current_speed = float(speeds[min(idx, n - 1)])

        hdg += yaw * DT
        x   += current_speed * math.sin(hdg) * DT
        y   += current_speed * math.cos(hdg) * DT
        t   += DT
        trail_xy.append((x, y))
        trail_speeds.append(current_speed)

        if idx >= n - 2 and math.hypot(goal[0] - x, goal[1] - y) < GOAL_TOL:
            break

    return trail_xy, trail_speeds


# ── Kleurschema ───────────────────────────────────────────────────────────────
C = dict(
    bg="#0f1117", panel="#171a24", grid="#21253a",
    plan="#3a5a9a", trail="#4a9eff", arc="#9d7fe8",
    b1="#ff4f6a", boat="#f5a623", entry="white", muted="#55596e",
    text="#c8cad4",
)

SPEED_CMAP = plt.cm.plasma


def _style_axis(ax):
    ax.set_facecolor(C["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(C["grid"])
    ax.grid(True, color=C["grid"], lw=0.5, ls="--", alpha=0.6)
    ax.tick_params(colors=C["muted"], labelsize=7.5)
    ax.set_aspect("equal")


def _draw_buoy_clusters(ax, buoy_positions):
    """Tekent boei-wolken en -centra; werkt voor elk aantal boeien."""
    buoy_colors = [C["b1"], "#3dd68c", "#9d7fe8", "#f5a623"]  # uitbreidbaar
    for idx, cluster in enumerate(buoy_positions):
        col = buoy_colors[idx % len(buoy_colors)]
        lbl = f"B{idx + 1}"
        avg_lat = sum(p[0] for p in cluster) / len(cluster)
        avg_lon = sum(p[1] for p in cluster) / len(cluster)
        cx, cy = to_m(avg_lon, avg_lat)
        for lat, lon in cluster:
            px, py = to_m(lon, lat)
            ax.scatter([px], [py], s=40, color=col, alpha=0.4, zorder=6,
                       edgecolors="white", lw=0.5)
        ax.scatter([cx], [cy], s=150, color=col, zorder=7,
                   edgecolors="white", linewidths=0.9, marker="D")
        ax.text(cx, cy + 4, lbl, color=col, fontsize=9,
                ha="center", va="bottom", fontweight="bold")


def _speed_segments(
    xs: List[float], ys: List[float], speeds: List[float],
    vmin: float, vmax: float,
) -> LineCollection:
    points  = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    norm    = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors_ = SPEED_CMAP(norm(speeds[:-1]))
    lc      = LineCollection(segs, colors=colors_, linewidths=2.0,
                             capstyle="round", zorder=3)
    return lc


def _speed_colorbar(fig, ax, vmin, vmax):
    sm = plt.cm.ScalarMappable(cmap=SPEED_CMAP,
                               norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("snelheid (m/s)", color=C["muted"], fontsize=7.5)
    cb.ax.yaxis.set_tick_params(colors=C["muted"], labelsize=7)
    cb.outline.set_edgecolor(C["grid"])


def _auto_limits(ax, xs, ys, pad_frac=0.15):
    """Stel as-grenzen in op basis van de data met wat marge."""
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xpad = max((xmax - xmin) * pad_frac, 5.0)
    ypad = max((ymax - ymin) * pad_frac, 5.0)
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, ymax + ypad)


# ── Hoofd-plotfunctie ─────────────────────────────────────────────────────────

def make_plots(buoy_positions, scenarios, marge, out_course, out_sim):
    # Kleurenschaal loopt van 0 (stilstand) tot FAST_SPEED, zodat de volledige
    # ramp — inclusief het optrekken vanuit en afremmen tot stilstand — in
    # kleur zichtbaar is, in plaats van geclipt te worden op SLOW_SPEED.
    color_min = 0.0
    color_max = _config_stub.FAST_SPEED

    # ── Figuur 1: gepland parcour per scenario ────────────────────────────────
    fig1, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 4.8))
    if len(scenarios) == 1:
        axes = [axes]
    fig1.patch.set_facecolor(C["bg"])

    for ax, (label, boat_ll, state, start_ll, start_hdg) in zip(axes, scenarios):
        pad = plan_path(buoy_positions, marge, state,
                        boat_ll, start_ll, start_hdg)

        xy_m   = [to_m(lon, lat) for (lat, lon), _ in pad]
        speeds = [spd for _, spd in pad]
        px, py = zip(*xy_m)

        _style_axis(ax)

        lc = _speed_segments(list(px), list(py), speeds, color_min, color_max)
        ax.add_collection(lc)

        # Richtingspijlen
        step = max(2, len(px) // 9)
        for i in range(0, len(px) - step, step):
            ddx, ddy = px[i + step] - px[i], py[i + step] - py[i]
            if math.hypot(ddx, ddy) > 0.3:
                ax.annotate("", xy=(px[i] + ddx * 0.6, py[i] + ddy * 0.6),
                            xytext=(px[i] + ddx * 0.4, py[i] + ddy * 0.4),
                            arrowprops=dict(arrowstyle="->", color=C["trail"],
                                            lw=1.2, mutation_scale=10), zorder=5)

        _draw_buoy_clusters(ax, buoy_positions)

        bx, by = to_m(boat_ll[1], boat_ll[0])
        ax.scatter([bx], [by], s=130, color=C["boat"], zorder=8,
                   edgecolors="white", linewidths=0.9)

        sx, sy = to_m(start_ll[1], start_ll[0])
        ax.scatter([sx], [sy], s=60, color=C["entry"], zorder=9,
                   edgecolors=C["trail"], linewidths=1.5)

        length = sum(math.hypot(px[i + 1] - px[i], py[i + 1] - py[i])
                     for i in range(len(px) - 1))
        ax.text(0.97, 0.03, f"pad: {length:.0f} m", transform=ax.transAxes,
                color=C["muted"], fontsize=7.5, ha="right", va="bottom")
        ax.set_title(label, color=C["text"], fontsize=12, fontweight="bold", pad=6)
        ax.set_xlabel("oost (m)", color=C["muted"], fontsize=8.5)
        if ax is axes[0]:
            ax.set_ylabel("noord (m)", color=C["muted"], fontsize=8.5)

        _auto_limits(ax, px, py)

    _speed_colorbar(fig1, axes[-1], color_min, color_max)
    fig1.suptitle(
        "Boei-parcour  ·  gepland pad per scenario  ·  kleur = snelheid",
        color=C["text"], fontsize=12.5, fontweight="bold", y=1.02,
    )
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1.savefig(out_course, dpi=160, bbox_inches="tight",
                 facecolor=C["bg"], edgecolor="none")
    print(f"Opgeslagen: {out_course}")

    # ── Figuur 2: gevaren baan (Pure Pursuit) voor het eerste scenario ────────
    label, boat_ll, state, start_ll, start_hdg = scenarios[0]
    pad      = plan_path(buoy_positions, marge, state,
                         boat_ll, start_ll, start_hdg)
    path_xy  = [to_m(lon, lat) for (lat, lon), _ in pad]
    path_spd = [spd for _, spd in pad]

    trail_xy, trail_spd = pure_pursuit(
        path_xy, path_spd,
        to_m(start_ll[1], start_ll[0]),
        start_hdg,
    )

    fig2, ax = plt.subplots(figsize=(11, 7))
    fig2.patch.set_facecolor(C["bg"])
    _style_axis(ax)

    px, py = zip(*path_xy)
    ax.plot(px, py, color=C["plan"], lw=1.4, ls="--", zorder=2, label="gepland pad")
    ax.scatter(px, py, s=8, color=C["plan"], zorder=2)

    tx, ty = zip(*trail_xy)
    lc2 = _speed_segments(list(tx), list(ty), trail_spd, color_min, color_max)
    ax.add_collection(lc2)

    step = max(8, len(tx) // 22)
    for i in range(0, len(tx) - step, step):
        ddx, ddy = tx[i + step] - tx[i], ty[i + step] - ty[i]
        if math.hypot(ddx, ddy) > 0.2:
            ax.annotate("", xy=(tx[i] + ddx, ty[i] + ddy),
                        xytext=(tx[i], ty[i]),
                        arrowprops=dict(arrowstyle="->", color=C["trail"],
                                        lw=1.1, mutation_scale=10), zorder=5)

    _draw_buoy_clusters(ax, buoy_positions)

    bx, by = to_m(boat_ll[1], boat_ll[0])
    ax.scatter([bx], [by], s=150, color=C["boat"], zorder=8,
               edgecolors="white", linewidths=1.0)

    sx, sy = to_m(start_ll[1], start_ll[0])
    ax.scatter([sx], [sy], s=80, color=C["entry"], zorder=9,
               edgecolors=C["trail"], linewidths=1.5)
    ax.text(sx, sy + 4, f"start  {start_hdg:.0f}°", color=C["entry"],
            fontsize=8, ha="center", va="bottom")

    travelled = sum(math.hypot(tx[i + 1] - tx[i], ty[i + 1] - ty[i])
                    for i in range(len(tx) - 1))
    sim_time  = (len(trail_xy) - 1) * DT
    ax.text(
        0.5, 0.04,
        f"gevaren: {travelled:.0f} m   ·   tijd: {sim_time:.1f} s   ·   "
        f"lookahead: {LOOKAHEAD:.0f} m",
        transform=ax.transAxes, color=C["muted"],
        fontsize=8.5, ha="center", va="bottom",
    )

    _speed_colorbar(fig2, ax, color_min, color_max)

    handles = [
        Line2D([0], [0], color=C["plan"],  lw=1.4, ls="--", label="gepland pad"),
        Line2D([0], [0], color=C["trail"], lw=2.4,           label="gevaren baan"),
        Line2D([0], [0], color=C["b1"],    lw=0, marker="D", ms=8,
               markeredgecolor="w", label="B1 centrum + wolk"),
        Line2D([0], [0], color=C["boat"],  lw=0, marker="o", ms=8,
               markeredgecolor="w", label="bootpositie (referentie)"),
        Line2D([0], [0], color=C["entry"], lw=0, marker="o", ms=8,
               markeredgecolor=C["trail"], label="startpositie"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8.5,
              facecolor=C["panel"], edgecolor=C["grid"],
              labelcolor=C["text"], framealpha=1)
    ax.set_xlabel("oost (m)", color=C["muted"], fontsize=9)
    ax.set_ylabel("noord (m)", color=C["muted"], fontsize=9)
    ax.set_title(
        "Gesimuleerde vaart  ·  Pure Pursuit  ·  kleur = snelheid",
        color=C["text"], fontsize=12.5, fontweight="bold", pad=8,
    )

    all_x = list(px) + list(tx)
    all_y = list(py) + list(ty)
    _auto_limits(ax, all_x, all_y)

    fig2.tight_layout()
    fig2.savefig(out_sim, dpi=160, bbox_inches="tight",
                 facecolor=C["bg"], edgecolor="none")
    print(f"Opgeslagen: {out_sim}")

    return travelled, sim_time




# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CLEARANCE = 5.0   # m extra marge om de boei

    # Één boei met een puntenwolk van mogelijke posities
    buoy_variants  = [offset(0, 35)]
    buoy_positions = [buoy_variants]

    # Startpositie: links van de boei, koers ~oost
    start     = offset(-15, 0)
    start_hdg = 0.0   # graden (oost)

    # Scenarios: (label, boot_latlon, state, start_latlon, start_heading)
    # boot_latlon = huidige bootpositie op het moment van padberekening
    #               (dient als coördinaten-referentie in padplanning_buoy)
    scenarios = [
        ("Recht op boei af",   offset(10, 0),   "START", start, 90.0),
        ("Scheve aanvlieging",  offset(10, 10),  "START", start, -50.0),
        ("Tegengesteld",        offset(10, 0),   "START", start, 90.0),
    ]

    travelled, t = make_plots(
        buoy_positions, scenarios, CLEARANCE,
        out_course="buoy_path.png",
        out_sim="buoy_simulation.png",
    )
    print(f"\nGevaren baan (scenario 1): {travelled:.0f} m in {t:.1f} s")