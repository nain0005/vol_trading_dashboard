"""Shared color tokens for charts (see dataviz skill reference palette).

Dark, high-contrast "trading terminal" palette — every chart in the app
pulls from here (plot/paper background, gridlines, ink), so this one file
re-themes the entire dashboard at once. Chosen for WCAG-reasonable
contrast against the near-black SURFACE (INK_PRIMARY ~15.5:1, INK_MUTED
~4.6:1) and for CATEGORICAL to stay visually distinct at a glance on a
dark background, which needs more saturation/lightness than the same
hues would on white.
"""

CATEGORICAL = [
    "#4c9aff",  # blue
    "#2dd4bf",  # aqua
    "#fbbf24",  # yellow
    "#34d399",  # green
    "#a78bfa",  # violet
    "#f87171",  # red
    "#f472b6",  # magenta
    "#fb923c",  # orange
]

# Positive/negative diverging pair -- used for P&L, vega, and stress-test
# bars/heatmaps throughout the app. Green/red (not blue/red) deliberately:
# it's the near-universal finance convention for profit/loss and reads
# instantly, which matters more here than palette-internal consistency
# with CATEGORICAL[0].
DIVERGING_POS = "#22c55e"  # green — profit / positive
DIVERGING_NEG = "#f87171"  # red — loss / negative
DIVERGING_MID = "#2a323d"

STATUS_GOOD = "#22c55e"
STATUS_WARNING = "#fbbf24"
STATUS_SERIOUS = "#fb923c"
STATUS_CRITICAL = "#ef4444"

INK_PRIMARY = "#e8eaed"
INK_SECONDARY = "#9aa4b2"
INK_MUTED = "#6b7684"
GRIDLINE = "#232a34"
SURFACE = "#0d1117"

# UI chrome only (custom CSS in dashboard.py) — charts never use these three.
SURFACE_RAISED = "#161b22"  # card/metric-tile background, one step lighter than the page SURFACE for elevation
ACCENT_SOFT = "rgba(76,154,255,0.14)"  # tint of CATEGORICAL[0], for badges/active-tab backgrounds/hover states
CHROME_DEEP = "#05070a"  # darker than SURFACE -- the header gradient's darkest stop
