#![allow(unused_macros)]
#![allow(unused_imports)]

// image
macro_rules! if_image {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "image_0-23", feature = "image_0-24"))]
            $item
        )*
    };
}
pub(crate) use if_image;

macro_rules! has_image {
    ($($item:item)*) => {
        crate::utils::if_image! {
            #[allow(unused_imports)]
            use crate::image as _;
            $($item)*
        }
    }
}
pub(crate) use has_image;

// nalgebra
macro_rules! if_nalgebra {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "nalgebra_0-26", feature = "nalgebra_0-27", feature = "nalgebra_0-28", feature = "nalgebra_0-29", feature = "nalgebra_0-30", feature = "nalgebra_0-31"))]
            $item
        )*
    };
}
pub(crate) use if_nalgebra;

macro_rules! has_nalgebra {
    ($($item:item)*) => {
        crate::utils::if_nalgebra! {
            #[allow(unused_imports)]
            use crate::nalgebra as _;
            $($item)*
        }
    }
}
pub(crate) use has_nalgebra;

// opencv
macro_rules! if_opencv {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "opencv_0-63", feature = "opencv_0-64", feature = "opencv_0-65", feature = "opencv_0-66"))]
            $item
        )*
    };
}
pub(crate) use if_opencv;

macro_rules! has_opencv {
    ($($item:item)*) => {
        crate::utils::if_opencv! {
            #[allow(unused_imports)]
            use crate::opencv as _;
            $($item)*
        }
    }
}
pub(crate) use has_opencv;

// ndarray
macro_rules! if_ndarray {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "ndarray_0-15"))]
            $item
        )*
    };
}
pub(crate) use if_ndarray;

macro_rules! has_ndarray {
    ($($item:item)*) => {
        crate::utils::if_ndarray! {
            #[allow(unused_imports)]
            use crate::ndarray as _;
            $($item)*
        }
    }
}
pub(crate) use has_ndarray;

// tch
macro_rules! if_tch {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "tch_0-8"))]
            $item
        )*
    };
}
pub(crate) use if_tch;

macro_rules! has_tch {
    ($($item:item)*) => {
        crate::utils::if_tch! {
            #[allow(unused_imports)]
            use crate::tch as _;
            $($item)*
        }
    }
}
pub(crate) use has_tch;

// imageproc
macro_rules! if_imageproc {
    ($($item:item)*) => {
        $(
            #[cfg(any(feature = "imageproc_0-23"))]
            $item
        )*
    };
}
pub(crate) use if_imageproc;

macro_rules! has_imageproc {
    ($($item:item)*) => {
        crate::utils::if_imageproc! {
            #[allow(unused_imports)]
            use crate::imageproc as _;
            $($item)*
        }
    }
}
pub(crate) use has_imageproc;
