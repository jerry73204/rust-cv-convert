#!/usr/bin/env python3

import itertools
import subprocess
from typing import List


VERSION_GROUPS = {
    "image": ["0.23", "0.24"],
    "nalgebra": ["0.26", "0.27", "0.28", "0.29", "0.30", "0.31"],
    "opencv": ["0.63", "0.64", "0.65"],
    "ndarray": ["0.15"],
    "tch": ["0.7"],
}


def main():
    feature_groups = map(vers_to_features, VERSION_GROUPS.items())
    feature_configs = itertools.product(*feature_groups)

    for features in feature_configs:
        print("Testing ", features)

        feature_arg = " ".join(features)
        subprocess.run(["cargo", "test", "--release", "--features", feature_arg])


def vers_to_features(pair):
    pkg, vers = pair
    features = list(map(lambda ver: "{}_{}".format(pkg, ver.replace(".", "-")), vers))
    return features


if __name__ == "__main__":
    main()
