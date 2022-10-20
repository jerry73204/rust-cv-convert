#!/usr/bin/env python3

import itertools
import subprocess
from dataclasses import dataclass
from typing import List
import os
from pathlib import Path
import sys

CARGO_DEPENDENCY_TEMPLATE = """[dependencies.{}]
version = '{}'
package = '{}'
default-features = {}
features = [{}]
optional = true
"""


LIB_EXPORT_TEMPLATE = """#[cfg(feature = "{}")]
pub use {} as {};
"""

CRATE_MACRO_TEMPLATE = """// {}
macro_rules! if_{} {{
    ($($item:item)*) => {{
        $(
            #[cfg(any({}))]
            $item
        )*
    }};
}}
pub(crate) use if_{};

macro_rules! has_{} {{
    ($($item:item)*) => {{
        crate::macros::if_{}! {{
            #[allow(unused_imports)]
            use crate::{} as _;
            $($item)*
        }}
    }}
}}
pub(crate) use has_{};
"""

MACRO_FILE_TEMPLATE = """#![allow(unused_macros)]
#![allow(unused_imports)]

{}
"""


@dataclass
class VerGroup:
    versions: List[str]
    default_features: bool
    features = List[str]

    def __init__(self, versions, default_features, features):
        self.versions = versions
        self.default_features = default_features
        self.features = features


@dataclass
class PkgVer:
    pkg: str
    ver: str
    rename_pkg: str
    feature: str

    def __init__(self, pkg: str, ver: str):
        self.pkg = pkg
        self.ver = ver
        self.rename_pkg = "{}_{}".format(pkg, ver.replace(".", "-"))
        self.rename_mod = "{}_{}".format(pkg, ver.replace(".", "_"))
        self.feature = "{}_{}".format(pkg, ver.replace(".", "-"))


VERSION_GROUPS = {
    "image": VerGroup(["0.23", "0.24"], True, []),
    "nalgebra": VerGroup(["0.26", "0.27", "0.28", "0.29", "0.30", "0.31"], True, []),
    "opencv": VerGroup(
        ["0.63", "0.64", "0.65", "0.66", "0.67", "0.68", "0.69", "0.70"], False, ["calib3d"]
    ),
    "ndarray": VerGroup(["0.15"], True, []),
    "tch": VerGroup(["0.8"], True, []),
    "imageproc": VerGroup(["0.23"], True, []),
}


def main():
    if len(sys.argv) != 2:
        print("Usage: generate.py test|generate")
        exit(1)

    command = sys.argv[1]

    if command == "test":
        test()
    elif command == "generate":
        generate()
    else:
        raise ValueError("unknown command {}".format(command))


def test():
    def vers_to_features(pair):
        pkg, group = pair
        features = list(map(lambda ver: PkgVer(pkg, ver).feature, group.versions))
        return features

    feature_groups = map(vers_to_features, VERSION_GROUPS.items())
    feature_configs = itertools.product(*feature_groups)

    for features in feature_configs:
        print("Testing ", features)

        feature_arg = " ".join(features)
        subprocess.run(["cargo", "test", "--release", "--features", feature_arg])


def generate():
    cargo_dep_list = list()
    lib_export_list = list()
    crate_macro_list = list()

    for pkg, ver_group in VERSION_GROUPS.items():
        for ver in ver_group.versions:
            info = PkgVer(pkg, ver)

            if ver_group.default_features:
                default_features_text = "true"
            else:
                default_features_text = "false"

            feature_list_text = ", ".join(
                map(
                    lambda feature: "'{}'".format(feature),
                    ver_group.features,
                )
            )

            cargo_dep = CARGO_DEPENDENCY_TEMPLATE.format(
                info.rename_pkg,
                info.ver,
                info.pkg,
                default_features_text,
                feature_list_text,
            )
            lib_export = LIB_EXPORT_TEMPLATE.format(
                info.feature, info.rename_mod, info.pkg
            )

            cargo_dep_list.append(cargo_dep)
            lib_export_list.append(lib_export)
            pass

        attributes = list(
            map(
                lambda ver: 'feature = "{}"'.format(PkgVer(pkg, ver).feature),
                ver_group.versions,
            )
        )
        attributes_text = ", ".join(attributes)

        crate_macro = CRATE_MACRO_TEMPLATE.format(
            pkg, pkg, attributes_text, pkg, pkg, pkg, pkg, pkg
        )
        crate_macro_list.append(crate_macro)
        pass

    cargo_dep_text = "\n".join(cargo_dep_list)
    lib_export_text = "\n".join(lib_export_list)
    crate_macro_text = "\n".join(crate_macro_list)
    macro_file_text = MACRO_FILE_TEMPLATE.format(crate_macro_text)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    cargo_file = os.path.join(script_dir, "Cargo.toml.patch")
    lib_file = os.path.join(script_dir, "lib.rs.patch")
    macro_file = os.path.join(script_dir, "macros.rs.patch")

    with open(cargo_file, "w") as f:
        f.write(cargo_dep_text)

    with open(lib_file, "w") as f:
        f.write(lib_export_text)

    with open(macro_file, "w") as f:
        f.write(macro_file_text)


if __name__ == "__main__":
    main()
