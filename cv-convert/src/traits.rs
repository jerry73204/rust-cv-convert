pub use as_ref_cv::*;
pub use from::*;
pub use try_as_ref_cv::*;
pub use try_from::*;

mod try_from {
    /// Fallible type conversion that is analogous to [TryFrom](std::convert::TryFrom).
    pub trait TryFromCv<T>
    where
        Self: Sized,
    {
        type Error;

        fn try_from_cv(from: &T) -> Result<Self, Self::Error>;
    }

    /// Fallible type conversion that is analogous to [TryInto](std::convert::TryInto).
    pub trait TryToCv<T> {
        type Error;

        fn try_to_cv(&self) -> Result<T, Self::Error>;
    }

    impl<T, U> TryToCv<U> for T
    where
        U: TryFromCv<T>,
    {
        type Error = <U as TryFromCv<T>>::Error;

        fn try_to_cv(&self) -> Result<U, Self::Error> {
            U::try_from_cv(self)
        }
    }
}

mod from {
    /// Type conversion that is analogous to [From](std::convert::From).
    pub trait FromCv<T> {
        fn from_cv(from: &T) -> Self;
    }

    /// Type conversion that is analogous to [Into](std::convert::Into).
    pub trait ToCv<T> {
        fn into_cv(&self) -> T;
    }

    impl<T, U> ToCv<U> for T
    where
        U: FromCv<T>,
    {
        fn into_cv(&self) -> U {
            U::from_cv(self)
        }
    }
}

mod as_ref_cv {
    pub trait AsRefCv<'a, T>
    where
        T: 'a,
    {
        fn as_ref_cv(&'a self) -> T;
    }
}

mod try_as_ref_cv {
    pub trait TryAsRefCv<'a, T>
    where
        T: 'a,
    {
        type Error;

        fn try_as_ref_cv(&'a self) -> Result<T, Self::Error>;
    }
}
