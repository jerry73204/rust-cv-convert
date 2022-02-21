use crate::opencv::{core as core_cv, prelude::*};
use crate::{common::*, TryFromCv};
use half::f16;

pub use element_type::*;
mod element_type {
    use super::*;

    pub trait OpenCvElement {
        const DEPTH: i32;
    }

    impl OpenCvElement for u8 {
        const DEPTH: i32 = core_cv::CV_8U;
    }

    impl OpenCvElement for i8 {
        const DEPTH: i32 = core_cv::CV_8S;
    }

    impl OpenCvElement for u16 {
        const DEPTH: i32 = core_cv::CV_16U;
    }

    impl OpenCvElement for i16 {
        const DEPTH: i32 = core_cv::CV_16S;
    }

    impl OpenCvElement for i32 {
        const DEPTH: i32 = core_cv::CV_32S;
    }

    impl OpenCvElement for f16 {
        const DEPTH: i32 = core_cv::CV_16F;
    }

    impl OpenCvElement for f32 {
        const DEPTH: i32 = core_cv::CV_32F;
    }

    impl OpenCvElement for f64 {
        const DEPTH: i32 = core_cv::CV_64F;
    }
}

pub(crate) use mat_ext::*;
mod mat_ext {
    use super::*;

    pub trait MatExt {
        fn shape(&self) -> Vec<usize>;

        fn as_slice<T>(&self) -> Result<&[T]>
        where
            T: OpenCvElement;

        #[cfg(test)]
        fn new_randn<T>(shape: &[usize]) -> Result<Self>
        where
            Self: Sized,
            T: OpenCvElement;
    }

    impl MatExt for core_cv::Mat {
        fn shape(&self) -> Vec<usize> {
            self.mat_size()
                .iter()
                .map(|&dim| dim as usize)
                .chain([self.channels() as usize])
                .collect()
        }

        fn as_slice<T>(&self) -> Result<&[T]>
        where
            T: OpenCvElement,
        {
            ensure!(self.depth() == T::DEPTH, "element type mismatch");
            ensure!(self.is_continuous(), "Mat data must be continuous");

            let numel = self.total();
            let ptr = self.ptr(0)? as *const T;

            let slice = unsafe { slice::from_raw_parts(ptr, numel) };
            Ok(slice)
        }

        #[cfg(test)]
        fn new_randn<T>(shape: &[usize]) -> Result<Self>
        where
            T: OpenCvElement,
        {
            let shape: Vec<_> = shape.iter().map(|&val| val as i32).collect();
            let mut mat = Self::zeros_nd(&shape, T::DEPTH)?.to_mat()?;
            core_cv::randn(&mut mat, &0.0, &1.0)?;
            Ok(mat)
        }
    }
}

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
