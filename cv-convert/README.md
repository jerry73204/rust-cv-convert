# cv-convert: Convert computer vision data types in Rust

Type conversions among famous Rust computer vision libraries. It
supports the following crates:

- [image](https://crates.io/crates/image)
- [imageproc](https://crates.io/crates/imageproc)
- [nalgebra](https://crates.io/crates/nalgebra)
- [opencv](https://crates.io/crates/opencv)
- [tch](https://crates.io/crates/tch)
- [ndarray](https://crates.io/crates/ndarray)

## Usage

Run `cargo add cv-convert` to add this crate to your project. In the
default setting, up-to-date dependency versions are used.

If you desire to enable specified dependency versions. Add
`default-features = false` and select crate versions as Cargo
features. For example, the feature `nalgebra_0-30` enables nalgebra
0.30.x.

```toml
[dependencies.cv-convert]
version = 'x.y.z'  # Please look up the recent version on crates.io
default-features = false
features = [
    'image_0-24',
    'opencv_0-84',
    'tch_0-13',
    'nalgebra_0-32',
    'ndarray_0-15',
]
```

The minimum supported `rustc` is 1.51. You may use older versions of
the crate (>=0.6) in order to use `rustc` versions that do not support
const-generics.

## Cargo Features

### opencv

- `opencv_0-84`
- `opencv_0-83`
- `opencv_0-82`
- `opencv_0-81`
- `opencv_0-80`
- `opencv_0-79`
- `opencv_0-78`
- `opencv_0-77`
- `opencv_0-76`
- `opencv_0-75`
- `opencv_0-74`
- `opencv_0-73`
- `opencv_0-72`
- `opencv_0-71`
- `opencv_0-70`
- `opencv_0-69`
- `opencv_0-68`
- `opencv_0-67`
- `opencv_0-66`
- `opencv_0-65`
- `opencv_0-64`
- `opencv_0-63`

### image

- `image_0-24`
- `image_0-23`

### imageproc

- `imageproc_0-23`

### ndarray

- `ndarray_0-15`

### nalgebra

- `nalgebra_0-32`
- `nalgebra_0-31`
- `nalgebra_0-30`
- `nalgebra_0-29`
- `nalgebra_0-28`
- `nalgebra_0-27`
- `nalgebra_0-26`

### tch

- `tch_0-13`

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

## Contribute to this Project

### Add a new dependency version

To add the new version of nalgebra 0.32 for cv-convert for example,
open `cv-convert-generate/packages.toml` in the source repository. Add
a new version to the list like this.

```toml
[package.nalgebra]
versions = ["0.26", "0.27", "0.28", "0.29", "0.30", "0.31", "0.32"]
use_default_features = true
features = []
```

Run `make generate` at the top-level directory. It modifies Rust
source files automatically. One extra step is to copy the snipplet in
`cv-convert/generated/Cargo.toml.snipplet` and paste it to
`cv-convert/Cargo.toml`.


### Add a new type conversion

To add a new type conversion, take `image::DynamicImage` and
`opencv::Mat` for example. Proceed to `cv-convert/src` and implement
the code in `with_opencv_image.rs` because it is a conversion among
opencv and image crates.


Choose `FromCv` or `TryFromCv` trait and add the trait implementation
on `image::DynamicImage` and `opencv::Mat` types. The choice of
`FromCv` or `TryFromCv` depends on whether the conversion is fallible
or not.

```rust
impl FromCv<&image::DynamicImage> for opencv::Mat { /* omit */ }
impl FromCv<&opencv::Mat> for image::DynamicImage { /* omit */ }

// or

impl TryFromCv<&image::DynamicImage> for opencv::Mat { /* omit */ }
impl TryFromCv<&opencv::Mat> for image::DynamicImage { /* omit */ }

#[cfg(test)]
mod tests {
    // Write a test
}
```

## License

MIT license. See [LICENSE](LICENSE.txt) file.
