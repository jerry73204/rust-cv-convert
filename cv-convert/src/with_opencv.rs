use crate::TryToCv;
use anyhow::{ensure, Error, Result};
use half::f16;
use opencv::{core as core_cv, prelude::*};
use std::slice;

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
        fn size_with_depth(&self) -> Vec<usize>;

        fn numel(&self) -> usize {
            self.size_with_depth().iter().product()
        }

        fn as_slice<T>(&self) -> Result<&[T]>
        where
            T: OpenCvElement;

        fn type_name(&self) -> String;

        #[cfg(test)]
        fn new_randn_2d(rows: i32, cols: i32, typ: i32) -> Result<Self>
        where
            Self: Sized;

        #[cfg(test)]
        fn new_randn_nd<T>(shape: &[usize]) -> Result<Self>
        where
            Self: Sized,
            T: OpenCvElement;
    }

    impl MatExt for core_cv::Mat {
        fn size_with_depth(&self) -> Vec<usize> {
            let size = self.mat_size();
            let size = size.iter().map(|&dim| dim as usize);
            let channels = self.channels() as usize;
            size.chain([channels]).collect()
        }

        fn as_slice<T>(&self) -> Result<&[T]>
        where
            T: OpenCvElement,
        {
            ensure!(self.depth() == T::DEPTH, "element type mismatch");
            ensure!(self.is_continuous(), "Mat data must be continuous");

            let numel = self.numel();
            let ptr = self.ptr(0)? as *const T;

            let slice = unsafe { slice::from_raw_parts(ptr, numel) };
            Ok(slice)
        }

        fn type_name(&self) -> String {
            core_cv::type_to_string(self.typ()).unwrap()
        }

        #[cfg(test)]
        fn new_randn_2d(rows: i32, cols: i32, typ: i32) -> Result<Self>
        where
            Self: Sized,
        {
            let mut mat = Self::zeros(rows, cols, typ)?.to_mat()?;
            core_cv::randn(&mut mat, &0.0, &1.0)?;
            Ok(mat)
        }

        #[cfg(test)]
        fn new_randn_nd<T>(shape: &[usize]) -> Result<Self>
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

impl<T> TryToCv<core_cv::Point_<T>> for core_cv::Mat
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<core_cv::Point_<T>> {
        let slice = self.data_typed::<T>()?;
        ensure!(slice.len() == 2, "invalid length");
        let point = core_cv::Point_::<T> {
            x: slice[0],
            y: slice[1],
        };
        Ok(point)
    }
}

impl<T> TryToCv<core_cv::Point3_<T>> for core_cv::Mat
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<core_cv::Point3_<T>> {
        let slice = self.data_typed::<T>()?;
        ensure!(slice.len() == 3, "invalid length");
        let point = core_cv::Point3_::<T> {
            x: slice[0],
            y: slice[1],
            z: slice[2],
        };
        Ok(point)
    }
}

impl<T> TryToCv<core_cv::Mat> for core_cv::Point_<T>
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<core_cv::Mat> {
        let core_cv::Point_ { x, y, .. } = *self;
        let mat = core_cv::Mat::from_slice(&[x, y])?;
        Ok(mat)
    }
}

impl<T> TryToCv<core_cv::Mat> for core_cv::Point3_<T>
where
    T: core_cv::DataType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<core_cv::Mat> {
        let core_cv::Point3_ { x, y, z, .. } = *self;
        let mat = core_cv::Mat::from_slice(&[x, y, z])?;
        Ok(mat)
    }
}
