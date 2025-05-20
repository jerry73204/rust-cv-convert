pub use as_ref_cv::*;
pub use to::*;
pub use try_as_ref_cv::*;
pub use try_to::*;

mod try_to {
    /// Fallible type conversion that is analogous to [TryInto](std::convert::TryInto).
    pub trait TryToCv<T> {
        type Error;

        fn try_to_cv(&self) -> Result<T, Self::Error>;
    }
}

mod to {
    /// Type conversion that is analogous to [Into](std::convert::Into).
    pub trait ToCv<T> {
        fn to_cv(&self) -> T;
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
