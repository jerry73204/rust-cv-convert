//! Types and traits for conversion between types from popular computer vision libraries.

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
