//! Data conversion among computer vision libraries.
//!
//! # Version Selection
//!
//! In the default setting, up-to-date dependencies are used. The
//! default dependency versions are listed in `[features]` in
//! cv-convert `Cargo.toml`.
//!
//! You can manually select desired dependency versions. The choices
//! of dependency versions are named accordingly as Cargo
//! features. For example, the feature `nalgebra_0-31` enables
//! nalgebra 0.31.x.  It allows to list crate version selections in
//! `Cargo.toml`.
//!
//! ```toml
//! [dependencies.cv-convert]
//! version = 'x.y.z'
//! default-features = false
//! features = [
//!     'image_0-24',
//!     'opencv_0-76',
//!     'tch_0-10',
//!     'nalgebra_0-32',
//!     'ndarray_0-15',
//! ]
//! ```
//!
//! It's impossible to enable two or more versions for a
//! dependency. For example, `nalgebra_0-31` and `nalgebra_0-32` are
//! incompatible.
//!
//! # Traits
//!
//! The trait [ToCv] provides `.to_cv()` method for infallible conversions, and
//! [TryToCv] provides `.try_to_cv()` method for fallible conversions.
//! Just like std's [Into] and [TryInto] traits.
//!
//! ```rust
//! # use cv_convert::{nalgebra, opencv};
//! use cv_convert::{ToCv, TryToCv};
//! use nalgebra as na;
//! use opencv as cv;
//!
//! // ToCv - infallible conversion
//! let cv_point = cv::core::Point2d::new(1.0, 3.0);
//! let na_point: na::Point2<f64> = cv_point.to_cv();
//!
//! // ToCv - the other direction
//! let na_point = na::Point2::<f64>::new(1.0, 3.0);
//! let cv_point: cv::core::Point2d = na_point.to_cv();
//!
//! // TryToCv - fallible conversion
//! let na_mat = na::DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! let cv_mat = na_mat.try_to_cv().unwrap();
//!
//! // TryToCv - the other direction
//! let cv_mat = cv::core::Mat::from_slice_2d(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();
//! let na_mat: na::DMatrix<f64> = cv_mat.try_to_cv().unwrap();
//! ```
//!
//!
//! # Supported conversions
//!
//! The notations are used for simplicity.
//!
//! - `S -> T` suggests the conversion is defined by non-fallible [ToCv].
//! - `S ->? T` suggests the conversion is defined by fallible [TryToCv].
//! - `(&)T` means the type can be either owned or borrowed.
//! - `&'a S -> &'a T` suggests that the target type borrows the source type.
//!
//! ## opencv -> opencv
//!
//! - [(&)Mat](opencv::core::Mat) ->? [Point_<T>](opencv::core::Point_)
//! - [(&)Mat](opencv::core::Mat) ->? [Point3_<T>](opencv::core::Point3_)
//! - [(&)Point_<T>](opencv::core::Point_) ->? [Mat](opencv::core::Mat)
//! - [(&)Point3_<T>](opencv::core::Point3_) ->? [Mat](opencv::core::Mat)
//!
//! ## std -> tch
//!
//! - owned/borrowed multi-dimensional [array](array) ->? [Tensor](tch::Tensor)
//!   - [(&)\[T; N\]](array) ->? [Tensor](tch::Tensor)
//!   - [(&)\[\[T; N2\]; N1\]](array) ->? [Tensor](tch::Tensor)
//!   - ... and so on up to 6 dimensions
//!
//! ## tch -> std
//!
//! - &'a [Tensor](tch::Tensor) -> &'a multi-dimensional [array](array)
//!   - &'a [Tensor](tch::Tensor) -> &'a [\[T; N\]](array)
//!   - &'a [Tensor](tch::Tensor) -> &'a [\[\[T; N2\]; N1\]](array)
//!   - ... and so on up to 6 dimensions
//! - [(&)Tensor](tch::Tensor) -> owned multi-dimensional [array](array)
//!   - [(&)Tensor](tch::Tensor) ->? [\[T; N\]](array)
//!   - [(&)Tensor](tch::Tensor) ->? [\[\[T; N2\]; N1\]](array)
//!   - ... and so on up to 6 dimensions
//!
//! ## tch -> ndarray
//!
//! - &[Tensor](tch::Tensor) ->? [Array](ndarray::Array)
//!
//! ## ndarray -> tch
//!
//! - &[Array](ndarray::Array) ->? [Tensor](tch::Tensor)
//!
//! ## ndarray -> opencv
//!
//! - &[Array](ndarray::Array) ->? [Mat](opencv::core::Mat)
//!
//! ## image -> tch
//!
//! - [(&)ImageBuffer](image::ImageBuffer) ->? [TchTensorAsImage]
//! - [(&)DynamicImage](image::DynamicImage) ->? [TchTensorAsImage]
//!
//! ## image -> opencv
//!
//! - [(&)ImageBuffer](image::ImageBuffer) ->? [Mat](opencv::core::Mat)
//! - [(&)DynamicImage](image::DynamicImage) ->? [Mat](opencv::core::Mat)
//!
//! ## opencv -> image 0.23
//!
//! - [Mat](opencv::core::Mat) ->? [(&)ImageBuffer](image::ImageBuffer)
//!
//! ## opencv -> image 0.24
//!
//! - [Mat](opencv::core::Mat) ->? [(&)ImageBuffer](image::ImageBuffer)
//! - [Mat](opencv::core::Mat) ->? [(&)DynamicImage](image::DynamicImage)
//!
//! ## opencv -> imageproc
//! - [(&)Point_<T>](opencv::core::Point_) -> [Point<T>](imageproc::point::Point)
//!
//! ## imageproc -> opencv
//! - [(&)Point<T>](imageproc::point::Point) -> [Point_<T>](opencv::core::Point_)
//!
//! ## opencv -> nalgebra
//!
//! - [(&)Mat](opencv::core::Mat) ->? [OMatrix](nalgebra::OMatrix)
//! - [(&)Point_<T>](opencv::core::Point_) -> [Point2<T>](nalgebra::Point2)
//! - [(&)Point3_<T>](opencv::core::Point3_) -> [Point2<T>](nalgebra::Point3)
//! - [(&)OpenCvPose<(&)Point3d>](OpenCvPose) ->? [Isometry3<f64>](nalgebra::Isometry3)
//! - [(&)OpenCvPose<(&)Mat>](OpenCvPose) ->? [Isometry3<f64>](nalgebra::Isometry3)
//!
//! ## nalgebra -> opencv
//!
//! - [(&)OMatrix](nalgebra::OMatrix) ->? [Mat](opencv::core::Mat)
//! - [(&)Point2<T>](nalgebra::Point2) -> [Point_<T>](opencv::core::Point_)
//! - [(&)Point3<T>](nalgebra::Point3) -> [Point3_<T>](opencv::core::Point3_)
//! - [(&)Translation<N, D>](nalgebra::Translation) ->? [Mat](opencv::core::Mat)
//! - [(&)Isometry3<T>](nalgebra::Isometry3) ->? [OpenCvPose<Point3_<T>>](OpenCvPose)
//! - [(&)Isometry3<f64>](nalgebra::Isometry3) ->? [OpenCvPose<Mat>](OpenCvPose)
//! - [(&)Isometry3<f32>](nalgebra::Isometry3) ->? [OpenCvPose<Mat>](OpenCvPose)
//!
//! ## opencv -> tch
//!
//! - [(&)Mat](opencv::core::Mat) ->? [Tensor](tch::Tensor)
//!
//!   The input [Mat](opencv::core::Mat) is regarded as an n-dimensional array with a m-channel elements.
//!   The output [Tensor](tch::Tensor) have n+1 dimensions, which last additional m-sized dimension is the channel.
//!
//! - [(&)Mat](opencv::core::Mat) ->? [TchTensorAsImage]
//!
//!   The input [Mat](opencv::core::Mat) must be a 2D image.
//!   The output [Tensor](tch::Tensor) within [TchTensorAsImage] has 3 dimensions, which last additional dimension is the channel.
//!
//! - [&Mat](opencv::core::Mat) ->? [OpenCvMatAsTchTensor]
//!
//! ## tch -> opencv
//!
//! - [(&)Tensor](tch::Tensor) ->? [Mat](opencv::core::Mat)
//!
//!    The n-dimensinoal input [Tensor](tch::Tensor) is converted to a [Mat](opencv::core::Mat)
//!    with n dimensions and a channel of size 1.
//!
//! - [(&)TchTensorAsImage](TchTensorAsImage) ->? [Mat](opencv::core::Mat)
//!
//!    The output [Mat](opencv::core::Mat) is a 2D image, which height, width and channel size
//!    are judged from the input [TchTensorAsImage](TchTensorAsImage) shape.
//!
//!
//! ## opencv -> ndarray
//!
//! - [&Mat](opencv::core::Mat) ->? [ArrayView](ndarray::ArrayView)
//! - [(&)Mat](opencv::core::Mat) ->? [Array](ndarray::Array)
//!
//!
//! # Notes for OpenCV
//!
//! For opencv older than 0.66, some systems requires `clang-runtime`
//! feature to build successfully. Otherwise you will get `libclang
//! shared library is not loaded on this thread!` panic. Add `opencv`
//! dependency along side `cv-convert` in your project Cargo.toml to
//! enable this feature.
//!
//! ```toml
//! cv-convert = { version = "0.22.0", default-features = false, features = ["opencv_0-65"] }
//! opencv = { version = "0.65", features = ["clang-runtime"] }
//! ```
//!
//! Most opencv modules, such as `videoio` and `aruco`, are disabled
//! by default to avoid bloating. Add opencv dependency to your
//! project Cargo.toml to enable default modules in your project.

mod traits;
pub use traits::*;

pub mod prelude {
    pub use crate::traits::{ToCv, TryToCv};
}

use cfg_if::cfg_if;

// modules

cfg_if! {
    if #[cfg(feature = "image")] {
    pub use image;
    }
}

cfg_if! {
    if #[cfg(feature = "imageproc")] {
    pub use imageproc;
    }
}

cfg_if! {
    if #[cfg(feature = "nalgebra")] {
    pub use nalgebra;
    }
}

cfg_if! {
    if #[cfg(feature = "ndarray")] {
    pub use ndarray;
    }
}

cfg_if! {
    if #[cfg(feature = "opencv")] {
    pub use opencv;

    mod with_opencv;
    #[allow(unused)]
    pub use with_opencv::*;
    }
}

cfg_if! {
    if #[cfg(feature = "tch")] {
    pub use tch;

    mod with_tch;
    #[allow(unused)]
    pub use with_tch::*;
    }
}

cfg_if! {
    if #[cfg(all(feature = "tch", feature = "image"))] {
        mod with_tch_image;
    #[allow(unused)]
        pub use with_tch_image::*;
    }
}

cfg_if! {
    if #[cfg(all(feature = "tch", feature = "ndarray"))] {
        mod with_tch_ndarray;
    #[allow(unused)]
        pub use with_tch_ndarray::*;
    }
}

cfg_if! {
    if #[cfg(all(feature = "image", feature = "opencv"))] {
        mod with_opencv_image;
    #[allow(unused)]
        pub use with_opencv_image::*;
    }
}

cfg_if! {
    if #[cfg(all(feature = "imageproc", feature = "opencv"))] {
        mod with_opencv_imageproc;
    #[allow(unused)]
        pub use with_opencv_imageproc::*;
    }
}

cfg_if! {
    if #[cfg(all(feature = "nalgebra", feature = "opencv"))] {
        mod with_opencv_nalgebra;
    #[allow(unused)]
        pub use with_opencv_nalgebra::*;
    }
}

cfg_if! {
    if #[cfg(all(feature = "tch", feature = "opencv"))] {
        mod with_opencv_tch;
    #[allow(unused)]
        pub use with_opencv_tch::*;
    }
}

cfg_if! {
    if #[cfg(all(feature = "ndarray", feature = "opencv"))] {
        mod with_opencv_ndarray;
    #[allow(unused)]
        pub use with_opencv_ndarray::*;
    }
}
