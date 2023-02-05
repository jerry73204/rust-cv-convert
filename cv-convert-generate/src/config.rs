use once_cell::sync::Lazy;
use serde::Deserialize;
use std::{collections::HashMap, fs};

const CONFIG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/packages.toml");

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    let text = fs::read_to_string(CONFIG_PATH).unwrap();
    toml::from_str(&text).unwrap()
});

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub package: HashMap<String, Package>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Package {
    pub versions: Vec<String>,
    pub features: Vec<String>,
    pub use_default_features: bool,
}
