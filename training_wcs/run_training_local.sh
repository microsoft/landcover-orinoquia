#!/usr/bin/env bash

# run from training_wcs/

rm -rf scripts_and_config/constants
rm -rf scripts_and_config/experiment_config

mkdir scripts_and_config/constants
mkdir scripts_and_config/constants/class_lists
mkdir scripts_and_config/constants/splits

cp ../constants/landsat_bands_info.py scripts_and_config/constants
cp ../constants/class_lists/wcs_fine_coarse_label_maps.json scripts_and_config/constants/class_lists


mkdir scripts_and_config/experiment_config

cp ../viz_utils.py scripts_and_config

# MODIFY here and below at the `python` command to specify which experiment config to use
cp ./experiments/coarse_baseline/coarse_baseline_config.py scripts_and_config/experiment_config

# MODIFY for new experiment config file
# in specifying the config_module_path, remember only the config file itself is copied to folder experiment_config
# without the experiment grouping level folder present in the repo
/anaconda/envs/wcs/bin/python scripts_and_config/train.py --config_module_path scripts_and_config/experiment_config/coarse_baseline_config.py
