# Derived tables

Two CSV files produced by the pipeline on DKRZ Levante. They contain every number behind Figure 2 and
Figures S1–S2, so `08`, `11` and `12` run without access to the model output.

## `complete_wc_analysis.csv` — one row per (event, run), 23 events × 3 runs = 69 rows

| Column | Unit | Written by | Meaning |
|---|---|---|---|
| `Year` | – | `01` | event year (the DJF season Dec(Year−1)–Feb(Year)) |
| `Run` | – | `01` | `Full Run`, `Uniform Run` or `Pattern Run` |
| `Event` | – | `01` | `El Niño` or `La Niña` |
| `Strength` | – | `01` | `Weak` (≥0.5), `Moderate` (≥1.0), `Strong` (≥1.5), `Very Strong` (≥2.0 °C) |
| `DJF` | °C | `01` | DJF Niño-3.4 index (detrended ONI) |
| `sst_mean_input` | °C | `02` | the UNIFORM SST anomaly of the event (same value on all three rows) |
| `{east_mean, west_mean}_omega500` | Pa s⁻¹ | `04` | box-mean ω at 500 hPa |
| `eastminuswest_omega500` | Pa s⁻¹ | `04` | ω500,E−W, the Walker circulation index (positive = stronger) |
| `{east_mean, west_mean}_{X}` | W m⁻² | `05` | box mean of term X, positive = energy into the atmosphere |
| `westminuseast_{X}` | W m⁻² | `05` | west − east contrast of X; `westminuseast_F_net` is F_net,W−E |

`X` is one of `F_clr` (TOA clear-sky net radiation), `F_cld` (TOA cloud radiative effect), `Surf_SW`,
`Surf_LW` (surface net radiation, sign-flipped), `Sensible`, `Latent` (surface turbulent fluxes,
sign-flipped) and `F_net` (their sum). The `eastminuswest_*` / `westminuseast_*` pairs are the same
contrast with opposite sign, both kept for convenience.

Values are raw (absolute) for each run; anomalies are formed against the control-run row below.

## `complete_climatology.csv` — one row, the control run

Same quantity columns as above, for the control simulation `vga0441` (ECHAM6.3 forced with observed,
time-varying SST, 1976–2009), averaged over `CLIM_PERIOD` (1979–2009).

## Provenance

Box means of the 69 runs were computed on Levante (August 2026) with the same box definitions and
formulas as `04`/`05`; all 69 values of `eastminuswest_omega500` were reproduced to 5 decimals by a
complete re-run of `04` in September 2026. The control-run row was recomputed with `CLIM_PERIOD` =
1979–2009 in October 2026.

