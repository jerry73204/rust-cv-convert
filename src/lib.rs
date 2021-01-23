//! Types and traits for conversion between types from popular computer vision libraries.

mod common;
mod traits;
#[cfg(feature = "opencv")]
mod with_opencv;
#[cfg(all(feature = "opencv", feature = "nalgebra"))]
mod with_opencv_nalgebra;
#[cfg(all(feature = "tch", feature = "image"))]
mod with_tch_image;

pub use traits::*;

#[cfg(feature = "opencv")]
pub use with_opencv::*;

#[cfg(all(feature = "opencv", feature = "nalgebra"))]
pub use with_opencv_nalgebra::*;

#[cfg(all(feature = "tch", feature = "image"))]
pub use with_tch_image::*;
