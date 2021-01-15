use crate::{common::*, TryFromCv};
use opencv::{core, prelude::*};

impl<T> TryFromCv<&core::Mat> for core::Point_<T>
where
    T: core::DataType + core::ValidPointType,
{
    type Error = Error;

    fn try_from_cv(from: &core::Mat) -> Result<Self> {
        let slice = from.data_typed::<T>()?;
        ensure!(slice.len() == 2, "invalid length");
        let point = Self {
            x: slice[0],
            y: slice[1],
        };
        Ok(point)
    }
}

impl<T> TryFromCv<core::Mat> for core::Point_<T>
where
    T: core::DataType + core::ValidPointType,
{
    type Error = Error;

    fn try_from_cv(from: core::Mat) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&core::Mat> for core::Point3_<T>
where
    T: core::DataType + core::ValidPoint3Type,
{
    type Error = Error;

    fn try_from_cv(from: &core::Mat) -> Result<Self> {
        let slice = from.data_typed::<T>()?;
        ensure!(slice.len() == 3, "invalid length");
        let point = Self {
            x: slice[0],
            y: slice[1],
            z: slice[2],
        };
        Ok(point)
    }
}

impl<T> TryFromCv<core::Mat> for core::Point3_<T>
where
    T: core::DataType + core::ValidPoint3Type,
{
    type Error = Error;

    fn try_from_cv(from: core::Mat) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&core::Point_<T>> for core::Mat
where
    T: core::DataType + core::ValidPointType,
{
    type Error = Error;

    fn try_from_cv(from: &core::Point_<T>) -> Result<Self> {
        let core::Point_ { x, y, .. } = *from;
        let mat = core::Mat::from_slice(&[x, y])?;
        Ok(mat)
    }
}

impl<T> TryFromCv<core::Point_<T>> for core::Mat
where
    T: core::DataType + core::ValidPointType,
{
    type Error = Error;

    fn try_from_cv(from: core::Point_<T>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&core::Point3_<T>> for core::Mat
where
    T: core::DataType + core::ValidPoint3Type,
{
    type Error = Error;

    fn try_from_cv(from: &core::Point3_<T>) -> Result<Self> {
        let core::Point3_ { x, y, z, .. } = *from;
        let mat = core::Mat::from_slice(&[x, y, z])?;
        Ok(mat)
    }
}

impl<T> TryFromCv<core::Point3_<T>> for core::Mat
where
    T: core::DataType + core::ValidPoint3Type,
{
    type Error = Error;

    fn try_from_cv(from: core::Point3_<T>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}
