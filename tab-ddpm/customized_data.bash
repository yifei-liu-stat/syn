# After cloning the repo, cd to under tab-ddpm/

# 1. Organize your dataset "data/ad21" the way other datasets are organized

# 2. Tune and train an evaluation modle: check the script for argparsed arguments
# (I am using CatBoost here with rmse, since I have modfied the script to take in another argument. With original repo, you can simply discard rmse and it will use r2 as the tuning metric)
python scripts/tune_evaluation_model.py ad21 catboost val cpu rmse 
# -> this will create tuned model hyperparameters in "tuned_models/catboost/ad21_val.json"


# this is because the orginal repo doesn't provide argparsed option for using val-tuned hyperparameter for tuning DDPM
# (they do have an argument is_cv = True in the evaluation script, but not argparsed)
# (so we are essentially using validation-tuned evaluation model, just naming as <ds-name>_cv.json)
mv tuned_models/catboost/ad21_val.json tuned_models/catboost/ad21_cv.json 


# Basic configuration template to start with
mkdir exp/ad21
cp exp/adult/config.toml exp/ad21/config.toml # MAKE SURE you MODIFY the newly created config.toml file to accommodate the new customized dataset


# 3. Tune tabddpm model: check the script for argparsed arguments
python scripts/tune_ddpm.py ad21 332 synthetic catboost cb 
# -> this will create "exp/ad21/ddpm_cb_best/config.toml", the tuned tabddpm model configuration file
# -> will also save teh best checkpoint in "exp/ad21/ddpm_cb_best/model.pt" for further usage

# 3. Do the follow-up analysis anyway you want (e.g. sample script -> fake data -> analysis)
python scripts/pipeline.py --config exp/ad21/ddpm_cb_best/config.toml --train --sample


