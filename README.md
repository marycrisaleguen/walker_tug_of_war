# walker_tug_of_war

Analysis code for

> Aleguen, M. C. M., Günther, M., & Kang, S. M. *Energetic Controls on ENSO-Driven Variability of the
> Pacific Walker Circulation.* Submitted to *Geophysical Research Letters*.

ECHAM6.3 is forced, for each of 23 observed ENSO events (1979–2013), with three SST anomalies: the
full July–June anomaly (**FULL**), its tropical-mean value applied uniformly (**UNIFORM**), and the
zero-mean residual (**PATTERN**). The notebooks here build the event catalog and the three boundary
conditions, reduce the model output to the Walker circulation index ω500,E−W and the zonal contrast
of net column energy input F_net,W−E, and draw every figure in the paper.

## Repository layout

```
src/config.py      all paths and the control-run averaging window (one place to edit)
notebooks/         the pipeline, run in numeric order (00 → 12)
data/              derived tables: one row per (event, run) + one row for the control run
figures/           figure files, named as in the manuscript
docs/PIPELINE.md   dependency graph and where each number in the paper comes from
```

## Pipeline

| Notebook | What it does | Writes |
|---|---|---|
| `00_data_ingestion` | merge the CMIP6 AMIP `ts` files, regrid to 1°, mask land | masked AMIP SST |
| `01_enso_event_detection` | detrended ONI, DJF classification, 23-event catalog | `data/complete_wc_analysis.csv` (seed) |
| `02_sst_pattern_decomposition` | FULL / UNIFORM / PATTERN boundary fields per event | `{year}_{FULL,UNIFORM,PATTERN}.nc`, `sst_mean_input` |
| `03_merge_model_output` | merge one experiment's 24 yearly ECHAM files per stream | `{year}_{RUN}_echam6_{ATM,BOT}_mm_1976-1999.nc` |
| `04_walker_circulation_omega500` | ω500 box means, E−W index, check maps | ω500 columns of both tables |
| `05_net_energy_input_fnet` | F_net and its six terms, box means, check maps | F_net columns of both tables |
| `06_temperature_stability_profile` | quick check of the ΔT cross-sections behind Figure 4 | — |
| `07_figure1_sst_decomposition_maps` | **Figure 1** | `figures/fig1_sst_decomp.{png,pdf}` |
| `08_figure2_walker_scaling` | **Figure 2** and its regression statistics; **Figure S1** (Figure 2 with the UNIFORM runs) | `figures/fig2_three_panel_scatter.{png,pdf}`, `figures/figS1_three_panel_scatter_uniform.{png,pdf}` |
| `09_figure3_omega_fnet_maps` | **Figure 3** | `figures/fig3_omega_Fnet.{png,pdf}` |
| `10_figure4_temperature_profile_maps` | **Figure 4** | `figures/fig4_temp_profile.{png,pdf}` |
| `11_figS2_fnet_component_attribution` | **Figure S2**: which flux terms carry ΔF_net,W−E | `figures/figS2_fnet_attribution.{png,pdf,csv}` |
| `12_tableS1_event_summary` | **Table S1**: the 23 events, their tropical-mean SST anomaly and the δω, δF of all three runs | `figures/tableS1_event_summary.{png,pdf,csv,tex}` |

`00`–`07`, `09` and `10` need the model output (DKRZ Levante, project mh1421). `08`, `11` and `12`
need only the two tables in `data/` and run on any laptop; their committed outputs show the results for
the real simulations.

## Definitions used throughout

* Western box 5°S–5°N, 80°E–160°E; eastern box 5°S–5°N, 160°W–80°W; cos-latitude weighted means.
* ω500,E−W = ω500,E − ω500,W (Pa s⁻¹). F_net,W−E = F_net,W − F_net,E (W m⁻²). Positive = stronger Walker
  circulation for both.
* F_net = net downward TOA radiation − net downward surface energy flux (radiation, sensible, latent),
  split exactly into F_clr, F_cld, Surf_SW, Surf_LW, Sensible and Latent (`05`). The ECHAM surface
  fluxes are positive downward, so they are sign-flipped: every term is positive when it adds energy to
  the atmospheric column, and the six terms sum to F_net within 2 × 10⁻⁵ W m⁻² in every row of `data/`.
* Anomalies are relative to the control run `vga0441` (ECHAM6.3 forced with observed, time-varying SST)
  averaged over `CLIM_PERIOD` = 1979–2009 for every variable (`src/config.py`); event runs are averaged
  over all 24 model years. Fractional anomaly δX = (X − X̄)/X̄. Evaluated at zero SST anomaly, the
  UNIFORM runs differ from this control by +0.45 % in ω500,E−W and −0.54 % in F_net,W−E (`08`, Step 4).
* Representative events in Figures 1, 3, 4: 1983 El Niño (DJF Niño-3.4 = 2.41 °C) and 1989 La Niña
  (−1.87 °C), the strongest of each sign in the catalog (`01`, Step 6).

## Figures

Every figure is drawn at its printed width (7 in; 3.5 in for Figure S2), so the text sizes in the
files are the printed sizes (7–9 pt). PNG files are 300 dpi; PDF files are vector graphics with
editable text. Figure 2, Figures S1–S2, Table S1 and `fig2_regression_stats.csv` (slope, standard error, 95 %
confidence interval, intercept, r², p for each panel) are built from `data/`; Figures 1, 3 and 4 are
built from the model output.

## Reproducing the results

```bash
conda env create -f environment.yml
conda activate walker-tug-of-war
jupyter lab notebooks/
```

On Levante the default paths in `src/config.py` point at the project data. Elsewhere, set
`WALKER_DATA_ROOT` (and, for `00`/`03`, `WALKER_CMIP6_TS_DIR`, `WALKER_MASK_PATH`,
`WALKER_EXPERIMENTS_DIR`) instead of editing the file. `WALKER_ANALYSIS_DIR` and `WALKER_FIGURES_DIR`
redirect the tables and figures, so a re-run need not overwrite the committed ones.

## Data availability

The ECHAM6.3 output is archived at `<DOI to be added>`. The derived tables in `data/` contain every
number needed for Figure 2, Figures S1–S2 and Table S1; `data/README.md` documents each column.

## Citation

See `CITATION.cff`. Please cite the paper when using this code.
