@echo off

for /l %%x in (0, 1, 2) do (
   python KFold_train.py --train_mode %%x --ml_mode XGBoost
)
