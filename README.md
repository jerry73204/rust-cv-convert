# cv-convert

Type conversions among famous Rust computer vision libraries. It supports the following crates:

- [image](https://crates.io/crates/image)
- [nalgebra](https://crates.io/crates/nalgebra)
- [opencv](https://crates.io/crates/opencv)
- [tch](https://crates.io/crates/tch)
- [ndarray](https://crates.io/crates/ndarray)

## Usage

This crate allows crate version selection in Cargo features.
For example, the feature `nalgebra_0-30` enables nalgebra 0.30.x.

```toml
[dependencies.cv-convert]
version = 'x.y.z'
features = [
    'image_0-23',
    'opencv_0-62',
    'tch_0-6',
    'nalgebra_0-30',
    'ndarray_0-15',
]
```

The `full` feature enable all crates with up-to-date versions.

```toml
[dependencies.cv-convert]
version = 'x.y.z'
features = ['full']
```

The minimum supported `rustc` is 1.51. You may use older versions of the crate (>=0.6) in order to use `rustc` versions that do not support const-generics.

## Cargo Features

### Include everything

- `full`

### opencv

- `opencv_0-62`
- `opencv_0-61`

Enable the corresponding feature below if you get `libclang shared library is not loaded on this thread!` panic.

- `opencv_0-62-clang-runtime`
- `opencv_0-61-clang-runtime`

### image

- `image_0-23`

### ndarray

- `ndarray_0-15`

### nalgebra

- `nalgebra_0-30`
- `nalgebra_0-29`
- `nalgebra_0-28`
- `nalgebra_0-27`
- `nalgebra_0-26`

### tch

- `tch_0-6`

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
