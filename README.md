# topcon_log_synchronizer.py

## Usage
### make_dataset.py
You can make dataset compatible with [heading_est_nn](https://bitbucket.org/Takuto224/heading_est_nn/src/master/) with following command.
Specify source directory and target directory with `--src_dir` and `--tgt_dir` flags.
Default source directory is `synced/`.
You need to synchronize logs with `synchronize_logs.ipynb` in advance.
```sh
python make_dataset.py [--src_dir PATH_TO_SRC_DIR] --tgt_dir PATH_TO_TGT_DIR
```
