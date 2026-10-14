"""
Central configuration for walker_tug_of_war.

Every notebook imports this module (``import config as cfg``) and reads all
paths and shared analysis settings from here, so the pipeline can be moved to
another machine by changing one file -- or, without editing anything, by
setting environment variables:

    WALKER_DATA_ROOT        root of the model/SST data tree
                            (default: the Levante project tree used for the paper)
    WALKER_ANALYSIS_DIR     where the derived CSV tables live
                            (default: <repo>/data, the tables shipped with the repo)
    WALKER_FIGURES_DIR      where figures are written (default: <repo>/figures)
    WALKER_CMIP6_TS_DIR, WALKER_MASK_PATH, WALKER_EXPERIMENTS_DIR
                            raw inputs outside the data tree (used by 00 and 03 only)

Figures use the same file names as the manuscript, so a finished run can be
copied straight into the LaTeX ``figures/`` folder.
"""

import os

# ---------------------------------------------------------------------------
# Repository-relative locations
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGURES_DIR = os.environ.get("WALKER_FIGURES_DIR", os.path.join(REPO_ROOT, "figures"))

ANALYSIS_DIR = os.environ.get("WALKER_ANALYSIS_DIR", os.path.join(REPO_ROOT, "data"))
ANALYSIS_CSV_PATH = os.path.join(ANALYSIS_DIR, "complete_wc_analysis.csv")        # one row per (Year, Run)
CLIMATOLOGY_CSV_PATH = os.path.join(ANALYSIS_DIR, "complete_climatology.csv")     # one row, control run

# ---------------------------------------------------------------------------
# Data tree (Levante, project mh1421)
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("WALKER_DATA_ROOT", "/work/mh1421/m301201/data")

DATA_AMIP_DIR = os.path.join(DATA_ROOT, "data-amip")          # AMIP SST + FULL/UNIFORM/PATTERN boundary fields
ECHAM_OUTPUT_DIR = os.path.join(DATA_ROOT, "echam-output")    # merged ECHAM6.3 output, one file per (event, run, stream)

DATA_AMIP_ELNINO_DIR = os.path.join(DATA_AMIP_DIR, "elnino")
DATA_AMIP_LANINA_DIR = os.path.join(DATA_AMIP_DIR, "lanina")
ECHAM_OUTPUT_ELNINO_DIR = os.path.join(ECHAM_OUTPUT_DIR, "elnino")
ECHAM_OUTPUT_LANINA_DIR = os.path.join(ECHAM_OUTPUT_DIR, "lanina")

# Control simulation (vga0441): ECHAM6.3 forced with OBSERVED, time-varying SST 1976-2009.
# All anomalies are relative to its time mean over CLIM_PERIOD (see README, Definitions).
CLIMATOLOGY_ATM_PATH = os.path.join(DATA_AMIP_DIR, "vga0441_echam6_ATM_mm_merged_1976-2009.nc")
CLIMATOLOGY_BOT_PATH = os.path.join(DATA_AMIP_DIR, "vga0441_echam6_BOT_mm_merged_1976-2009.nc")

# Averaging window applied to the control run in EVERY notebook that reads it:
# omega (04, 09), F_net (05, 09) and temperature (06, 10). One window for all variables,
# so every anomaly and every fractional-anomaly denominator (Figures 2-4, S1-S2) refers
# to the same baseline. The first three years of the run (1976-1978) are not used.
# Set to None to use the full record.
CLIM_PERIOD = ("1979-01-01", "2009-12-31")

# Raw ECHAM6 experiment output, one folder per experiment id (used by 03 only)
ECHAM_EXPERIMENTS_DIR = os.environ.get(
    "WALKER_EXPERIMENTS_DIR", "/work/mh1421/m301201/model2/mpiesm-landveg/experiments"
)

# ---------------------------------------------------------------------------
# AMIP SST (00, 01, 02)
# ---------------------------------------------------------------------------
# CMIP6 MPI-ESM1.2-LR AMIP surface temperature, used as SST over the ocean
CMIP6_AMIP_TS_DIR = os.environ.get(
    "WALKER_CMIP6_TS_DIR",
    "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/amip/r1i1p1f1/Amon/ts/gn/v20190815/",
)
AMIP_TS_RAW_PATH = os.path.join(DATA_AMIP_DIR, "ts_Amon_MPI-ESM1-2-LR_amip_r1i1p1f1_gn_197901-201412.nc")
AMIP_TS_1DEG_PATH = os.path.join(DATA_AMIP_DIR, "ts_Amon_MPI-ESM1-2-LR_amip_r1i1p1f1_gn_197901-201412_1deg.nc")
AMIP_TS_1DEG_MASKED_PATH = os.path.join(
    DATA_AMIP_DIR, "ts_Amon_MPI-ESM1-2-LR_amip_r1i1p1f1_gn_197901-201412_1deg_masked.nc"
)

# Land-sea mask on the MPI-ESM grid (slm: 1 = land, 0 = ocean)
MASK_PATH = os.environ.get(
    "WALKER_MASK_PATH", "/work/mh1421/m301201/Projects/04_thesis/data/land_sea_mask_mpiesm_flipped.nc"
)
