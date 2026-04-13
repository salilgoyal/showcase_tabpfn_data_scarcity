For new experiments:

  1. Create new yaml: configs/my-new-experiment.yaml
  2. Set experiment_name: "my-new-experiment"
  3. Run generator: python generate_job_script.py --config ../configs/my-new-experiment.yaml
  4. Submit: sbatch generated/submit_tabpfn_my-new-experiment.sh



# ARCHIVE BELOW

# 'Sample test from all of Cook County' experiment

## XGBoost with Evelyn preprocessing
nlprun -q jag -p standard -a pfn_env -g 1 -c 4 -r 32G -n xgboost-evelyn-nopropertychars-20-seeds -o ../outfiles/evelyn-nopropertychars-20-seeds/xgboost.out 'python3 run_xgboost_experiment.py --experiment_name evelyn-nopropertychars-20-seeds'

## TabPFN with Evelyn preprocessing
nlprun -q jag -p standard -a pfn_env -g 1 -c 4 -r 32G -n tabpfn-evelyn-includepropertychars-20-seeds -o ../outfiles/evelyn-includepropertychars-20-seeds/tabpfn.out 'python3 run_tabpfn_experiment.py --experiment_name tabpfn-includepropertychars-20-seeds'

nlprun -q jag -p standard -a pfn_env -g 1 -c 4 -r 32G -n tabpfn-evelyn-nopropertychars-20-seeds -o ../outfiles/evelyn-nopropertychars-20-seeds/tabpfn.out 'python3 run_tabpfn_experiment.py --experiment_name tabpfn-nopropertychars-20-seeds'

## XGBOOST
nlprun -q jag -p standard -a pfn_env -g 1 -c 4 -r 32G -n xgboost-optuna-upto100Ktrain-dropped-columns-20-seeds -o ../outfiles/optuna-upto100Ktrain-dropped-columns-20-seeds/xgboost.out 'python3 run_xgboost_experiment.py --experiment_name optuna-upto100Ktrain-dropped-columns-20-seeds'

nlprun -q jag -p standard -a pfn_env -g 1 -c 4 -r 32G -n xgboost-optuna-only2021train-upto10Ktrain-dropped-columns-20-seeds -o ../outfiles/optuna-only2021train-upto10Ktrain-dropped-columns-20-seeds/xgboost.out 'python3 run_xgboost_experiment.py --experiment_name optuna-only2021train-upto10Ktrain-dropped-columns-20-seeds'

## TABPFN
nlprun -q jag -p standard -a pfn_env -g 1 -c 4 -r 32G -n tabpfn-only2021train-upto10Ktrain-dropped-columns-20-seeds -o ../outfiles/optuna-only2021train-upto10Ktrain-dropped-columns-20-seeds/tabpfn.out 'python3 run_tabpfn_experiment.py --experiment_name optuna-only2021train-upto10Ktrain-dropped-columns-20-seeds'

# 'Sample test set from within training CBGs' experiment
nlprun -q jag -p standard -a pfn_env -g 1 -c 8 -r 32G -n within-cbg-optuna-40-seeds -o ../outfiles/within-cbg-optuna-40-seeds.out 'python3 run_within_cbg_experiment.py'

# order of experiments
Tu 11/11/25 optuna-upto10Ktrain-dropped-columns-20-seeds
Tu 11/11/25 optuna-notemporal-upto10Ktrain-dropped-columns-20-seeds
Tu 11/11/25 optuna-only2021train-upto10Ktrain-dropped-columns-20-seeds

Note in these three I used seeds spaced apart (ie seeds = [100 * i for i in range(20)])
Also here I run a few kinds:
- as below, but train size up to 10K
- as the bullet above, but without temporal columns (changed in src/data_utils.py)
- as the bullet above, but training data restricted to 2021

Fri 11/7/25 optuna-dropped-columns-40-seeds. Note here and below I used sequential seeds (ie seeds = range(40))
Th 11/6 upto10ktrain-dropped-columns-40-seeds
1k-train-only-calculated-sale-amount
1k-train-dropped-columns