# cv-convert

Type conversions among famous Rust computer vision libraries. It supports the following crates:

- [image](https://crates.io/crates/image)
- [nalgebra](https://crates.io/crates/nalgebra)
- [opencv](https://crates.io/crates/opencv)
- [tch](https://crates.io/crates/tch)
- [ndarray](https://crates.io/crates/ndarray)

## Using this crate

Add this snipplet to your `Cargo.toml` to include cv-convert with full support.

```toml
[dependencies.cv-convert]
version = "0.13"
features = ["full"]
```

To avoid bloat on unused libraries, it's suggested to specify used libraries manually.

```toml
[dependencies.cv-convert]
version = "0.13"
features = ["opencv", "nalgebra"]
```

The minimum supported `rustc` is 1.51
You may use older versions of the crate (>=0.6) in order to use `rustc` versions that do not support const-generics.

## Cargo Features

### Include everything

- `full`

### OpenCV

Enable `clang-runtime` in opencv crate. It is useful when you get `libclang shared library is not loaded on this thread!` panic.

- `opencv`
- `opencv-clang-runtime`

### Image

- `image`

### nalgebra

- `nalgebra`

### tch

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
