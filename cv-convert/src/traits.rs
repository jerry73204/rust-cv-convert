pub use from::*;
pub use try_from::*;

mod try_from {
    /// Fallible type conversion that is analogous to [TryFrom](std::convert::TryFrom).
    pub trait TryFromCv<T>
    where
        Self: Sized,
    {
        type Error;

        fn try_from_cv(from: T) -> Result<Self, Self::Error>;
    }

    /// Fallible type conversion that is analogous to [TryInto](std::convert::TryInto).
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
    /// Type conversion that is analogous to [From](std::convert::From).
    pub trait FromCv<T> {
        fn from_cv(from: T) -> Self;
    }

    /// Type conversion that is analogous to [Into](std::convert::Into).
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
