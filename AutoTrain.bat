@echo off

for /l %%x in (0, 1, 1) do (
   python KFold_train.py --train_mode %%x
)