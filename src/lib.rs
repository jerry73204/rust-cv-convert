//! Data conversion among computer vision libraries.
//!
//! # Traits
//!
//! The traits [FromCv] and [IntoCv] respectively define `.from_cv()` and `.into_cv()` methods.
//! The usage is similar to std's [From] and [Into]. The traits [TryFromCv] and [TryIntoCv] are fallible counterparts.
//! They respective define `.try_from_cv()` and `.try_into_cv()` methods. These traits can be imported from prelude.
//!
//! ```rust
//! use cv_convert::prelude::*;
//! ```
//! # Supported conversions
//!
//! The notations are used for simplicity.
//!
//! - `S -> T` suggests the conversion is defined by non-fallible [FromCv].
//! - `S ->? T` suggests the conversion is defined by fallible [TryFromCv].
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
//! ## image -> tch
//!
//! - [(&)ImageBuffer](image::ImageBuffer) ->? [Tensor](tch::Tensor)
//! - [(&)DynamicImage](image::DynamicImage) ->? [Tensor](tch::Tensor)
//!
//! ## image -> opencv
//!
//! - [(&)ImageBuffer](image::ImageBuffer) ->? [Mat](opencv::core::Mat)
//! - [(&)DynamicImage](image::DynamicImage) ->? [Mat](opencv::core::Mat)
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
//! - [Mat](opencv::core::Mat) ->? [TensorFromMat]
//!
//! ## tch -> opencv
//!
//! - [(&)Tensor](tch::Tensor) ->? [Mat](opencv::core::Mat)
//! - [(&)TensorAsImage](TensorAsImage) ->? [Mat](opencv::core::Mat)

mod common;

mod traits;
pub use traits::*;

pub mod prelude {
    pub use crate::traits::{FromCv, IntoCv, TryFromCv, TryIntoCv};
}

#[cfg(feature = "opencv")]
mod with_opencv;
#[cfg(feature = "opencv")]
pub use opencv;
#[cfg(feature = "opencv")]
pub use with_opencv::*;

#[cfg(feature = "image")]
pub use image;

#[cfg(feature = "nalgebra")]
pub use nalgebra;

#[cfg(feature = "ndarray")]
pub use ndarray;

#[cfg(feature = "tch")]
pub use tch;

#[cfg(all(feature = "opencv", feature = "image"))]
mod with_opencv_image;
#[cfg(all(feature = "opencv", feature = "image"))]
pub use with_opencv_image::*;

#[cfg(all(feature = "opencv", feature = "nalgebra"))]
mod with_opencv_nalgebra;
#[cfg(all(feature = "opencv", feature = "nalgebra"))]
pub use with_opencv_nalgebra::*;

#[cfg(all(feature = "opencv", feature = "tch"))]
mod with_opencv_tch;
#[cfg(all(feature = "opencv", feature = "tch"))]
pub use with_opencv_tch::*;

#[cfg(feature = "tch")]
mod with_tch;

#[cfg(all(feature = "tch", feature = "image"))]
mod with_tch_image;
#[cfg(all(feature = "tch", feature = "image"))]
pub use with_tch_image::*;

#[cfg(all(feature = "tch", feature = "ndarray"))]
mod with_tch_ndarray;
#[cfg(all(feature = "tch", feature = "ndarray"))]
pub use with_tch_ndarray::*;
