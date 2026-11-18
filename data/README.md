# data/

- `download.py` downloads NHTSA's `FLAT_CMPL.zip` and the `CMPL.txt` layout document, and stops
  if the documented field order differs from ours.
- `build_splits.py` builds the train / validation / test / time-shift splits (see the data card in
  the top-level README).
- `splits/*.csv` hold the complaint IDs (`ODINO`) and labels of every split. They are committed so
  that `make data-from-ids` rebuilds exactly the same examples from a later copy of the file.
- `raw/` and `processed/` are created locally and git-ignored.
