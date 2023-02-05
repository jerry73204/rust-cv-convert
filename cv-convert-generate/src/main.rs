mod config;

use anyhow::{bail, Result};
use clap::Parser;
use config::CONFIG;
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use std::{
    fs,
    process::{Command, Output},
};
use unzip_n::unzip_n;

unzip_n!(3);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Parser)]
enum Opts {
    /// Generate snipplets.
    Generate,

    /// Test all combinations of package versions.
    Test,
}

fn main() -> Result<()> {
    let opts = Opts::parse();

    match opts {
        Opts::Generate => {
            run_generate()?;
        }
        Opts::Test => {
            run_test()?;
        }
    }

    Ok(())
}

fn run_generate() -> Result<()> {
    let (cargo_dep_groups, lib_export_groups, macro_snipplets) = CONFIG
        .package
        .iter()
        .map(|(pkg_name, pkg)| {
            let (cargo_deps, lib_exports, feature_options) = pkg
                .versions
                .iter()
                .map(|version_dot| {
                    let pkg_of_ver = VersionedPackage::new(
                        pkg_name,
                        version_dot,
                        pkg.use_default_features,
                        &pkg.features,
                    );
                    let cargo_dep = pkg_of_ver.cargo_dep();
                    let lib_export = pkg_of_ver.lib_export();
                    let feature_option = pkg_of_ver.feature_option();
                    (cargo_dep, lib_export, feature_option)
                })
                .unzip_n_vec();

            let if_pkg_macro = format_ident!("if_{pkg_name}");
            let has_pkg_macro = format_ident!("has_{pkg_name}");

            let macro_snipplet = quote! {
                // #pkg_name
                macro_rules! #if_pkg_macro {
                    ($($item:item)*) => {
                        $(
                            #[cfg(any( #(#feature_options),* ))]
                            $item
                        )*
                    };
                }
                pub(crate) use #if_pkg_macro;

                macro_rules! #has_pkg_macro {
                    ($($item:item)*) => {
                        crate::macros::#if_pkg_macro! {
                            #[allow(unused_imports)]
                            use crate::#pkg_name as _;
                            $($item)*
                        }
                    }
                }
                pub(crate) use #has_pkg_macro;
            };

            (cargo_deps, lib_exports, macro_snipplet)
        })
        .unzip_n_vec();

    let cargo_dep_snipplet = cargo_dep_groups.into_iter().flatten().join("\n");
    let lib_rs_snipplet = {
        let iter = lib_export_groups.into_iter().flatten();
        quote! { #(#iter)* }
    };
    let macro_rs_snipplet = quote! {
        #![allow(unused_macros)]
        #![allow(unused_imports)]

        #(#macro_snipplets)*
    };

    fs::write("Cargo.toml.snipplet", cargo_dep_snipplet)?;
    fs::write("lib.rs.snipplet", lib_rs_snipplet.to_string())?;
    fs::write("macro.rs.snipplet", macro_rs_snipplet.to_string())?;

    Ok(())
}

fn run_test() -> Result<()> {
    let feature_combinations = CONFIG
        .package
        .iter()
        .map(|(pkg_name, pkg)| {
            pkg.versions.iter().map(|version_dot| {
                let pkg_of_ver = VersionedPackage::new(
                    pkg_name,
                    version_dot,
                    pkg.use_default_features,
                    &pkg.features,
                );
                pkg_of_ver.feature_name
            })
        })
        .multi_cartesian_product();

    for features in feature_combinations {
        let feature_arg = features.join(" ");
        eprintln!("Testing features {feature_arg}");

        let Output {
            status,
            stdout,
            stderr,
        } = Command::new("cargo")
            .arg("test")
            .arg("--release")
            .arg("--features")
            .arg(feature_arg)
            .output()?;

        if !status.success() {
            fs::write("stdout.txt", stdout)?;
            fs::write("stderr.txt", stderr)?;
            bail!(
                "Test failed for features {features:?}. Dump output to stdout.txt and stderr.txt"
            );
        }
    }

    Ok(())
}

struct VersionedPackage {
    pub pkg_name: String,
    pub version_dot: String,
    pub cargo_pkg_name: String,
    pub mod_name: String,
    pub feature_name: String,
    pub use_default_features: bool,
    pub features: Vec<String>,
}

impl VersionedPackage {
    pub fn new(
        pkg_name: &str,
        version_dot: &str,
        use_default_features: bool,
        features: &[String],
    ) -> Self {
        let pkg_name = pkg_name.to_string();
        let version_dot = version_dot.to_string();
        let version_dash = version_dot.replace('.', "-");
        let version_underscore = version_dot.replace('.', "_");
        let features = features.to_vec();
        let cargo_pkg_name = format!("{pkg_name}_{version_dash}");
        let mod_name = format!("{pkg_name}_{version_underscore}");
        let feature_name = format!("{pkg_name}_{version_dash}");

        Self {
            pkg_name,
            cargo_pkg_name,
            mod_name,
            feature_name,
            version_dot,
            use_default_features,
            features,
        }
    }

    pub fn cargo_dep(&self) -> String {
        let Self {
            pkg_name,
            cargo_pkg_name,
            version_dot,
            use_default_features,
            features,
            ..
        } = self;
        let default_feature_text = if *use_default_features {
            "true"
        } else {
            "false"
        };
        let feature_list_text = features
            .into_iter()
            .map(|feature| format!("'{feature}'"))
            .join(", ");

        format!(
            r#"[dependencies.{cargo_pkg_name}]
version = '{version_dot}'
package = '{pkg_name}'
default-features = {default_feature_text}
features = [{feature_list_text}]
optional = true
"#
        )
    }
    pub fn lib_export(&self) -> TokenStream {
        let Self {
            pkg_name,
            mod_name,
            feature_name,
            ..
        } = self;
        let mod_name = format_ident!("{mod_name}");
        let pkg_name = format_ident!("{pkg_name}");

        quote!(
            #[cfg(feature = #feature_name)]
            pub use #mod_name as #pkg_name;
        )
    }

    pub fn feature_option(&self) -> TokenStream {
        let Self { feature_name, .. } = self;
        quote! {
            feature = #feature_name
        }
    }
}
