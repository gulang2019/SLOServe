Paper figure snapshots
======================

These are compact plotting inputs, not new benchmark measurements. Raw result
archives are intentionally excluded. Source paths and SHA-256 hashes in the
provenance files identify the original inputs even when those archives are not
available in a clone.

Figure 7 (main paper)
--------------------
From the SLOServe repository root:

  python scripts/plot_beam_figure7_paper_style.py --main-only

Requires numpy, matplotlib, pdflatex and Ghostscript (gs). The output directory
is analysis_outputs/beam_figure7_main. The wrapper calls the exact original
renderer in plots/chat_7b_n_device_triptych.py, whose original driver is
scripts/recover_main_multi_replica_figure.py. It keeps only the ChatBot (7B),
Coder (7B), and Mixed (7B) energy-versus-SLO panels in a single row, with a plain
BEAM legend and no embedded captions or savings extrapolations.

The main paper references copies of chat_7B.pdf, code_7B.pdf, mixed_7B.pdf and
shared_legend.pdf in SLO-Packer-MLSys-27/figs/e2e_beam/. Its original six-workload,
two-metric Figure 7 remains unchanged in figs/e2e_new/ and is included in the
appendix under fig:distributed-full. The main figure retains fig:distributed.

Inputs:
  original_e2e/*.csv: unchanged copies of the original paper aggregates.
  figure7/source_preview.json: frozen BEAM metrics and source hashes.
  figure7/unscaled_preview.json: prior raw-energy/selection provenance.
  figure7/beam_points.csv: all Coder/Mixed records, including omitted points.

Without --main-only the wrapper can still recreate the earlier six-workload
energy-only preview. The extra BEAM workloads are NOT installed in the paper.

Measurement conventions and limitations
--------------------------------------
* All original baseline values are retained; no idle-energy addback is applied.
* Chat BEAM energy uses the affine PCIe-to-SXM estimate
    P_scaled = 70 + (P_PCIe - 52) * 330/248.
  Coder and Mixed BEAM use unscaled measured PCIe energy. Neither convention
  turns the measurements into hardware-matched SXM experiments.
* Cluster energy is N times the mean of the selected independent GPU shards,
  not a simultaneous full-cluster measurement. Full monitoring windows differ.
* Coder uses the older shard-0 sweep. Only its Pareto-dominated N=7 and N=8
  measurements are omitted from the displayed curve; their data are retained.
* BEAM uses conservative upper violation bounds under cumulative-token-deadline
  scoring, TPOT=25 ms, and total TTFT allowances 100/200/200 ms (Chat/Code/Mixed).
  Original Mixed baseline scores remain at TPOT=20 ms. Historical original
  scoring/calibration is not independently revalidated by these snapshots.
* Figure 3 estimates aggregate BEAM power as 8 times the mean of shards 0/3/7.
  Figure 8 uses shard 3 as the user-approved proxy for paper device index 4.
  Those independently replayed shards mix 20 and 25 ms scheduler targets.
  See their provenance JSONs for alignment and scaling details.

Figures 3 and 8
---------------
  python scripts/plot_beam_figure3_paper_style.py
  python scripts/plot_beam_figure8_paper_style.py

The default paths use figure3/ and figure8/ aligned_power.csv snapshots plus
timeline/energy_consumption-mixed.meta.json. Use --from-raw only when the original
benchmark archives are available. Replaying a CSV reproduces plotted data; it
does not independently remeasure or validate the original raw experiments.
