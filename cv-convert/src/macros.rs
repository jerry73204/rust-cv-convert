#![allow(unused_macros)]
#![allow(unused_imports)]

macro_rules! if_image {
    ($($item:item)*) => {
	$(
            #[cfg(feature = "image")]
	    $item
	)*
    };
}
pub(crate) use if_image;

macro_rules! if_imageproc {
    ($($item:item)*) => {
	$(
            #[cfg(feature = "imageproc")]
	    $item
	)*
    };
}
pub(crate) use if_imageproc;

macro_rules! if_nalgebra {
    ($($item:item)*) => {
	$(
            #[cfg(feature = "nalgebra")]
	    $item
	)*
    };
}
pub(crate) use if_nalgebra;

macro_rules! if_ndarray {
    ($($item:item)*) => {
	$(
            #[cfg(feature = "ndarray")]
	    $item
	)*
    };
}
pub(crate) use if_ndarray;

macro_rules! if_opencv {
    ($($item:item)*) => {
	$(
            #[cfg(feature = "opencv")]
	    $item
	)*
    };
}
pub(crate) use if_opencv;

macro_rules! if_tch {
    ($($item:item)*) => {
	$(
            #[cfg(feature = "tch")]
	    $item
	)*
    };
}
pub(crate) use if_tch;
