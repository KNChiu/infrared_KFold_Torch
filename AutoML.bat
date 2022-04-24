@echo off

for /l %%x in (0, 1, 3) do (
   python CatBoots_train.py --train_mode %%x
)
