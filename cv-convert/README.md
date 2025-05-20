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

**No crates are enabled by default.** You must specify which computer vision
libraries you want to use as features when adding cv-convert to your project.

```toml
[dependencies.cv-convert]
version = 'x.y.z'  # Please look up the recent version on crates.io
default-features = false
features = [
    'image',
    'opencv',
    'tch',
    'nalgebra',
    'ndarray',
    'imageproc',
]
```

The basic features (`image`, `opencv`, etc.) enable support for the compatible version ranges of each dependency.

The minimum supported `rustc` is 1.51. You may use older versions of
the crate (>=0.6) in order to use `rustc` versions that do not support
const-generics.

## Available Features

### Core library features

- `image` - Enable [image](https://crates.io/crates/image) crate support (latest version)
- `imageproc` - Enable [imageproc](https://crates.io/crates/imageproc) crate support (latest version)
- `nalgebra` - Enable [nalgebra](https://crates.io/crates/nalgebra) crate support (latest version)
- `ndarray` - Enable [ndarray](https://crates.io/crates/ndarray) crate support (latest version)
- `opencv` - Enable [opencv](https://crates.io/crates/opencv) crate support (latest version)
- `tch` - Enable [tch](https://crates.io/crates/tch) crate support (latest version)

### Supported version ranges

- `image` - Supports version >=0.24
- `imageproc` - Supports version >=0.22
- `nalgebra` - Supports versions >=0.26, <0.33
- `ndarray` - Supports version >=0.13
- `opencv` - Supports versions >=0.63, <0.89
- `tch` - Supports version >=0.13

## Example Usage

The crate provides `ToCv`, `TryToCv`, `AsRefCv`, `TryAsRefCv` traits, which are similar to standard library's `Into` and `TryInto`.

```rust
use cv_convert::{ToCv, TryToCv};
use nalgebra as na;
use opencv as cv;

// ToCv - infallible conversion
let cv_point = cv::core::Point2d::new(1.0, 3.0);
let na_point: na::Point2<f64> = cv_point.to_cv();

// ToCv - the other direction
let na_point = na::Point2::<f64>::new(1.0, 3.0);
let cv_point: cv::core::Point2d = na_point.to_cv();

// TryToCv - fallible conversion
let na_mat = na::DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let cv_mat = na_mat.try_to_cv()?;

// TryToCv - the other direction
let cv_mat = cv::core::Mat::from_slice_2d(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]])?;
let na_mat: na::DMatrix<f64> = cv_mat.try_to_cv()?;
```

## Contribute to this Project

### Add support for new dependency versions

Dependencies are now specified using range-based version requirements in `cv-convert/Cargo.toml`.
To support new versions of a dependency, simply update the version range in the `[dependencies]` section.

For example, to add support for nalgebra 0.33, update the nalgebra dependency:

```toml
[dependencies]
nalgebra = { version = ">=0.26, <0.34", optional = true }
```

This approach automatically supports all compatible versions within the specified range without
needing to generate code for each individual version.


### Add a new type conversion

To add a new type conversion, take `image::DynamicImage` and
`opencv::Mat` for example. Proceed to `cv-convert/src` and implement
the code in `with_opencv_image.rs` because it is a conversion among
opencv and image crates.


Choose `ToCv` or `TryToCv` trait and add the trait implementation
on `image::DynamicImage` and `opencv::Mat` types. The choice of
`ToCv` or `TryToCv` depends on whether the conversion is fallible
or not.

```rust
impl ToCv<opencv::Mat> for image::DynamicImage { /* omit */ }
impl ToCv<image::DynamicImage> for opencv::Mat { /* omit */ }

// or

impl TryToCv<opencv::Mat> for image::DynamicImage {
    type Error = SomeError;
    fn try_to_cv(&self) -> Result<opencv::Mat, Self::Error> { /* omit */ }
}
impl TryToCv<image::DynamicImage> for opencv::Mat {
    type Error = SomeError;
    fn try_to_cv(&self) -> Result<image::DynamicImage, Self::Error> { /* omit */ }
}

#[cfg(test)]
mod tests {
    // Write a test
}
```

## License

MIT license. See [LICENSE](LICENSE.txt) file.
