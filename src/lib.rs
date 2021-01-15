mod common;
mod with_opencv;
mod with_opencv_nalgebra;
mod with_tch_image;

pub use from::*;
pub use try_from::*;

#[cfg(feature = "opencv")]
pub use with_opencv::*;

#[cfg(all(feature = "opencv", feature = "nalgebra"))]
pub use with_opencv_nalgebra::*;

#[cfg(all(feature = "tch", feature = "image"))]
pub use with_tch_image::*;

mod try_from {
    pub trait TryFromCv<T>
    where
        Self: Sized,
    {
        type Error;

        fn try_from_cv(from: T) -> Result<Self, Self::Error>;
    }

    pub trait TryIntoCv<T> {
        type Error;

        fn try_into_cv(self) -> Result<T, Self::Error>;
    }

    impl<T, U> TryIntoCv<U> for T
    where
        U: TryFromCv<T>,
    {
        type Error = <U as TryFromCv<T>>::Error;

        fn try_into_cv(self) -> Result<U, Self::Error> {
            U::try_from_cv(self)
        }
    }
}

mod from {
    pub trait FromCv<T> {
        fn from_cv(from: T) -> Self;
    }

    pub trait IntoCv<T> {
        fn into_cv(self) -> T;
    }

    impl<T, U> IntoCv<U> for T
    where
        U: FromCv<T>,
    {
        fn into_cv(self) -> U {
            U::from_cv(self)
        }
    }
}
