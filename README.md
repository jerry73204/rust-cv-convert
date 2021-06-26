# cv-convert

Type conversions among famous Rust computer vision libraries. It supports the following crates:

- [image](https://crates.io/crates/image)
- [nalgebra](https://crates.io/crates/nalgebra)
- [opencv](https://crates.io/crates/opencv)
- [tch](https://crates.io/crates/tch)

## Import to Your Crate

Add cv-convert to `Cargo.toml` to import most conversions by default.

```toml
[dependencies.cv-convert]
version = "0.9"
```

You can manually choose supported libraries to avoid bloating.

```toml
version = "0.9"
default-features = false
features = ["opencv", "nalgebra"]
```

The minimum supported `rustc` is 1.51
You may use older versions of the crate (>=0.6) in order to use `rustc` versions that do not support const-generics.
## Supported Cargo Features

opencv crate features

- `opencv-clang-runtime`: Enable `clang-runtime` in opencv crate. Useful if you get `libclang shared library is not loaded on this thread!` panic.
- `opencv-contrib`: `opencv-contrib` has been dropped by the main `opencv` crate in favour of runtime module set detection. If you need the older version, 0.5.0 was the version that last supported `opencv-0.51.0`. 
- `opencv-4`, `opencv-34`, `opencv-32`, `opencv-buildtime-bindgen`: These features have been dropped in opencv-0.53.0 in favour of runtime generation and detection. cv-convert-0.6 was the last version that supported `opencv-0.52.0`, which had these features.

image crate feature

- `image`

nalgebra crate feature

- `nalgebra`
- With the arrival of `nalgebra` 0.26, const-generic are used in this crate and are incompatible with previous versions of `nalgebra`. If you are pulling in previous versions of `nalgebra`, please use 0.6 or older.

tch crate feature

- `tch`


## Usage

The crate provides `FromCv`, `TryFromCv`, `IntoCv`, `TryIntoCv` traits, which are similar to standard library's `From` and `Into`.

```rust
use cv_convert::{FromCv, IntoCv, TryFromCv, TryIntoCv};
use nalgebra as na;
use opencv as cv;

// FromCv
let cv_point = cv::core::Point2d::new(1.0, 3.0);
let na_points = na::Point2::<f64>::from_cv(&cv_point);

// IntoCv
let cv_point = cv::core::Point2d::new(1.0, 3.0);
let na_points: na::Point2<f64> = cv_point.into_cv();

// TryFromCv
let na_mat = na::DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let cv_mat = cv::core::Mat::try_from_cv(&na_mat)?;

// TryIntoCv
let na_mat = na::DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let cv_mat: cv::core::Mat = na_mat.try_into_cv()?;
```

## License

MIT license. See [LICENSE](LICENSE.txt) file.
