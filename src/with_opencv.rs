use crate::{common::*, TryFromCv};
use crate::opencv::{core as core_cv, prelude::*};

impl<T> TryFromCv<&core_cv::Mat> for core_cv::Point_<T>
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(from: &core_cv::Mat) -> Result<Self> {
        let slice = from.data_typed::<T>()?;
        ensure!(slice.len() == 2, "invalid length");
        let point = Self {
            x: slice[0],
            y: slice[1],
        };
        Ok(point)
    }
}

impl<T> TryFromCv<core_cv::Mat> for core_cv::Point_<T>
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(from: core_cv::Mat) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&core_cv::Mat> for core_cv::Point3_<T>
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(from: &core_cv::Mat) -> Result<Self> {
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

impl<T> TryFromCv<core_cv::Mat> for core_cv::Point3_<T>
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(from: core_cv::Mat) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&core_cv::Point_<T>> for core_cv::Mat
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(from: &core_cv::Point_<T>) -> Result<Self> {
        let core_cv::Point_ { x, y, .. } = *from;
        let mat = core_cv::Mat::from_slice(&[x, y])?;
        Ok(mat)
    }
}

impl<T> TryFromCv<core_cv::Point_<T>> for core_cv::Mat
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(from: core_cv::Point_<T>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> TryFromCv<&core_cv::Point3_<T>> for core_cv::Mat
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(from: &core_cv::Point3_<T>) -> Result<Self> {
        let core_cv::Point3_ { x, y, z, .. } = *from;
        let mat = core_cv::Mat::from_slice(&[x, y, z])?;
        Ok(mat)
    }
}

impl<T> TryFromCv<core_cv::Point3_<T>> for core_cv::Mat
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(from: core_cv::Point3_<T>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}
