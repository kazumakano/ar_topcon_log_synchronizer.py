import os.path as path
import shutil
from glob import iglob
from os import mkdir
from typing import Optional


def make_dataset(tgt_dir: str, src_dir: Optional[str] = None) -> None:
    if src_dir is None:
        src_dir = path.join(path.dirname(__file__), "synced/")
    if not path.exists(tgt_dir):
        mkdir(tgt_dir)

    for file in iglob(path.join(src_dir, "*", "*.csv")):
        shutil.copyfile(file, path.join(tgt_dir, path.basename(path.dirname(file)) + "_" + path.basename(file)))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", help="specify source directory", metavar="PATH_TO_SRC_DIR")
    parser.add_argument("-t", "--tgt_dir", required=True, help="specify target directory", metavar="PATH_TO_TGT_DIR")
    args = parser.parse_args()

    make_dataset(args.tgt_dir, args.src_dir)
