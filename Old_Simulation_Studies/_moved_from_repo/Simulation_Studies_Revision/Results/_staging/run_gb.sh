set -e
cd /home/hugo_souto/Stuff/Research/DiD-BCF/DiD-BCF/Simulation_Studies_Revision
for s in D_staggered D_contamination; do
  python3 scripts/run_goodman_bacon.py --experiment $s --reps 500 --jobs 6
  mv Results/goodman_bacon_summary.csv Results/goodman_bacon_summary_$s.csv
done
python3 scripts/run_goodman_bacon.py --experiment D_staggered --reps 300 --jobs 6 --ramp-sweep
mv Results/goodman_bacon_ramp_sweep.csv Results/goodman_bacon_ramp_sweep_D_staggered.csv
python3 scripts/run_goodman_bacon.py --experiment D_contamination --reps 300 --jobs 6 --ramp-sweep
mv Results/goodman_bacon_ramp_sweep.csv Results/goodman_bacon_ramp_sweep_D_contamination.csv
echo GB_ALL_DONE
