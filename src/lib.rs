use anyhow::{bail, ensure, Error, Result};

pub use from::*;
pub use try_from::*;

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

#[cfg(feature = "opencv")]
mod with_opencv {
    use super::*;
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
}

#[cfg(all(feature = "opencv", feature = "nalgebra"))]
mod with_opencv_nalgebra {
    use super::*;
    use nalgebra::{self as na, geometry as geo};
    use opencv::{calib3d, core, prelude::*};

    #[derive(Debug, Clone)]
    pub struct OpenCvPose<T> {
        pub rvec: T,
        pub tvec: T,
    }

    impl TryFromCv<OpenCvPose<&core::Point3d>> for geo::Isometry3<f64> {
        type Error = Error;

        fn try_from_cv(pose: OpenCvPose<&core::Point3d>) -> Result<Self> {
            let OpenCvPose { rvec, tvec } = pose;
            let rotation = {
                let rvec_mat = {
                    let core::Point3_ { x, y, z, .. } = *rvec;
                    core::Mat::from_slice(&[x, y, z])?
                };
                let mut rotation_mat = core::Mat::zeros(3, 3, core::CV_64FC1)?.to_mat()?;
                calib3d::rodrigues(&rvec_mat, &mut rotation_mat, &mut core::no_array()?)?;
                let rotation_matrix: na::Matrix3<f64> = TryFromCv::try_from_cv(rotation_mat)?;
                geo::UnitQuaternion::from_matrix(&rotation_matrix)
            };

            let translation = {
                let core::Point3_ { x, y, z } = *tvec;
                geo::Translation3::new(x, y, z)
            };

            let isometry = geo::Isometry3::from_parts(translation, rotation);
            Ok(isometry)
        }
    }

    impl TryFromCv<&OpenCvPose<core::Point3d>> for geo::Isometry3<f64> {
        type Error = Error;

        fn try_from_cv(from: &OpenCvPose<core::Point3d>) -> Result<Self> {
            let OpenCvPose { rvec, tvec } = from;
            TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })
        }
    }

    impl TryFromCv<OpenCvPose<core::Point3d>> for geo::Isometry3<f64> {
        type Error = Error;

        fn try_from_cv(from: OpenCvPose<core::Point3d>) -> Result<Self> {
            TryFromCv::try_from_cv(&from)
        }
    }

    impl TryFromCv<OpenCvPose<&core::Mat>> for geo::Isometry3<f64> {
        type Error = Error;

        fn try_from_cv(from: OpenCvPose<&core::Mat>) -> Result<Self> {
            let OpenCvPose {
                rvec: rvec_mat,
                tvec: tvec_mat,
            } = from;
            let rvec = core::Point3d::try_from_cv(rvec_mat)?;
            let tvec = core::Point3d::try_from_cv(tvec_mat)?;
            let isometry = TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })?;
            Ok(isometry)
        }
    }

    impl TryFromCv<&OpenCvPose<core::Mat>> for geo::Isometry3<f64> {
        type Error = Error;

        fn try_from_cv(from: &OpenCvPose<core::Mat>) -> Result<Self> {
            let OpenCvPose { rvec, tvec } = from;
            TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })
        }
    }

    impl TryFromCv<OpenCvPose<core::Mat>> for geo::Isometry3<f64> {
        type Error = Error;

        fn try_from_cv(from: OpenCvPose<core::Mat>) -> Result<Self> {
            let OpenCvPose { rvec, tvec } = &from;
            TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })
        }
    }

    impl<T> TryFromCv<&geo::Isometry3<T>> for OpenCvPose<core::Point3_<T>>
    where
        T: core::DataType + core::ValidPoint3Type + na::RealField,
    {
        type Error = Error;

        fn try_from_cv(from: &geo::Isometry3<T>) -> Result<OpenCvPose<core::Point3_<T>>> {
            let geo::Isometry3 {
                rotation,
                translation,
                ..
            } = from;

            let rvec = {
                let rotation_mat =
                    core::Mat::try_from_cv(rotation.to_rotation_matrix().into_inner())?;
                let mut rvec_mat = core::Mat::zeros(3, 1, core::CV_64FC1)?.to_mat()?;
                calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut core::no_array()?)?;
                let rvec = core::Point3_::new(
                    *rvec_mat.at_2d::<T>(0, 0)?,
                    *rvec_mat.at_2d::<T>(1, 0)?,
                    *rvec_mat.at_2d::<T>(2, 0)?,
                );
                rvec
            };
            let tvec = core::Point3_::new(translation.x, translation.y, translation.z);

            Ok(OpenCvPose { rvec, tvec })
        }
    }

    impl<T> TryFromCv<geo::Isometry3<T>> for OpenCvPose<core::Point3_<T>>
    where
        T: core::DataType + core::ValidPoint3Type + na::RealField,
    {
        type Error = Error;

        fn try_from_cv(from: geo::Isometry3<T>) -> Result<OpenCvPose<core::Point3_<T>>> {
            TryFromCv::try_from_cv(&from)
        }
    }

    impl TryFromCv<&geo::Isometry3<f64>> for OpenCvPose<core::Mat> {
        type Error = Error;

        fn try_from_cv(from: &geo::Isometry3<f64>) -> Result<OpenCvPose<core::Mat>> {
            let geo::Isometry3 {
                rotation,
                translation,
                ..
            } = from;

            let rvec = {
                let rotation_mat: Mat =
                    TryFromCv::try_from_cv(rotation.to_rotation_matrix().into_inner())?;
                let mut rvec_mat = core::Mat::zeros(3, 1, core::CV_64FC1)?.to_mat()?;
                calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut core::no_array()?)?;
                rvec_mat
            };
            let tvec = core::Mat::from_slice(&[translation.x, translation.y, translation.z])?;

            Ok(OpenCvPose { rvec, tvec })
        }
    }

    impl TryFromCv<geo::Isometry3<f64>> for OpenCvPose<core::Mat> {
        type Error = Error;

        fn try_from_cv(from: geo::Isometry3<f64>) -> Result<OpenCvPose<core::Mat>> {
            TryFromCv::try_from_cv(&from)
        }
    }

    impl TryFromCv<&geo::Isometry3<f32>> for OpenCvPose<core::Mat> {
        type Error = Error;

        fn try_from_cv(from: &geo::Isometry3<f32>) -> Result<OpenCvPose<core::Mat>> {
            let geo::Isometry3 {
                rotation,
                translation,
                ..
            } = from;

            let rvec = {
                let rotation_mat = Mat::try_from_cv(rotation.to_rotation_matrix().into_inner())?;
                let mut rvec_mat = Mat::zeros(3, 1, core::CV_32FC1)?.to_mat()?;
                calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut core::no_array()?)?;
                rvec_mat
            };
            let tvec = Mat::from_slice(&[translation.x, translation.y, translation.z])?;

            Ok(OpenCvPose { rvec, tvec })
        }
    }

    impl TryFromCv<geo::Isometry3<f32>> for OpenCvPose<core::Mat> {
        type Error = Error;

        fn try_from_cv(from: geo::Isometry3<f32>) -> Result<OpenCvPose<core::Mat>> {
            TryFromCv::try_from_cv(&from)
        }
    }

    impl<N, R, C> TryFromCv<&core::Mat> for na::MatrixMN<N, R, C>
    where
        N: na::Scalar + core::DataType,
        R: na::Dim + na::DimName,
        C: na::Dim + na::DimName,
        na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, R, C>,
    {
        type Error = Error;

        fn try_from_cv(from: &core::Mat) -> Result<Self> {
            let shape = from.size()?;
            let check_height = R::try_to_usize().map(|size| size == shape.height as usize);
            let check_width = C::try_to_usize().map(|size| size == shape.width as usize);
            let is_same_shape = check_height
                .into_iter()
                .chain(check_width.into_iter())
                .all(|is_same| is_same);
            ensure!(is_same_shape, "input and output matrix shapes differ");

            // ensure!(size.height == R::value() as i32 && size.width == R::value() as i32);
            let values = from.to_vec_2d()?.into_iter().flatten().collect::<Vec<N>>();
            Ok(Self::from_row_slice(&values))
        }
    }

    impl<N, R, C> TryFromCv<core::Mat> for na::MatrixMN<N, R, C>
    where
        N: na::Scalar + core::DataType,
        R: na::Dim + na::DimName,
        C: na::Dim + na::DimName,
        na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, R, C>,
    {
        type Error = Error;

        fn try_from_cv(from: core::Mat) -> Result<Self> {
            TryFromCv::try_from_cv(&from)
        }
    }

    impl<N, R, C, S> TryFromCv<&na::Matrix<N, R, C, S>> for core::Mat
    where
        N: na::Scalar + core::DataType,
        R: na::Dim + na::DimName,
        C: na::Dim + na::DimName,
        S: na::base::storage::Storage<N, R, C>,
        na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, C, R>,
    {
        type Error = Error;

        fn try_from_cv(from: &na::Matrix<N, R, C, S>) -> Result<Self> {
            let nrows = from.nrows();
            let mat =
                core::Mat::from_slice(from.transpose().as_slice())?.reshape(1, nrows as i32)?;
            Ok(mat)
        }
    }

    impl<N, R, C, S> TryFromCv<na::Matrix<N, R, C, S>> for core::Mat
    where
        N: na::Scalar + core::DataType,
        R: na::Dim + na::DimName,
        C: na::Dim + na::DimName,
        S: na::base::storage::Storage<N, R, C>,
        na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, C, R>,
    {
        type Error = Error;

        fn try_from_cv(from: na::Matrix<N, R, C, S>) -> Result<Self> {
            TryFromCv::try_from_cv(&from)
        }
    }

    impl<T> FromCv<&na::Point2<T>> for core::Point_<T>
    where
        T: na::Scalar + core::ValidPointType,
    {
        fn from_cv(from: &na::Point2<T>) -> Self {
            core::Point_::new(from.x, from.y)
        }
    }

    impl<T> FromCv<na::Point2<T>> for core::Point_<T>
    where
        T: na::Scalar + core::ValidPointType,
    {
        fn from_cv(from: na::Point2<T>) -> Self {
            FromCv::from_cv(&from)
        }
    }

    impl<T> FromCv<&core::Point_<T>> for na::Point2<T>
    where
        T: na::Scalar + core::ValidPointType,
    {
        fn from_cv(from: &core::Point_<T>) -> Self {
            Self::new(from.x, from.y)
        }
    }

    impl<T> FromCv<core::Point_<T>> for na::Point2<T>
    where
        T: na::Scalar + core::ValidPointType,
    {
        fn from_cv(from: core::Point_<T>) -> Self {
            FromCv::from_cv(&from)
        }
    }

    impl<T> FromCv<&na::Point3<T>> for core::Point3_<T>
    where
        T: na::Scalar + core::ValidPoint3Type,
    {
        fn from_cv(from: &na::Point3<T>) -> Self {
            Self::new(from.x, from.y, from.z)
        }
    }

    impl<T> FromCv<na::Point3<T>> for core::Point3_<T>
    where
        T: na::Scalar + core::ValidPoint3Type,
    {
        fn from_cv(from: na::Point3<T>) -> Self {
            FromCv::from_cv(&from)
        }
    }

    impl<T> FromCv<&core::Point3_<T>> for na::Point3<T>
    where
        T: na::Scalar + core::ValidPoint3Type,
    {
        fn from_cv(from: &core::Point3_<T>) -> Self {
            Self::new(from.x, from.y, from.z)
        }
    }

    impl<T> FromCv<core::Point3_<T>> for na::Point3<T>
    where
        T: na::Scalar + core::ValidPoint3Type,
    {
        fn from_cv(from: core::Point3_<T>) -> Self {
            FromCv::from_cv(&from)
        }
    }

    impl<N, D> TryFromCv<&geo::Translation<N, D>> for core::Mat
    where
        N: na::Scalar + core::DataType,
        D: na::DimName,
        na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, D>,
    {
        type Error = Error;

        fn try_from_cv(translation: &geo::Translation<N, D>) -> Result<Self> {
            let mat = core::Mat::from_exact_iter(translation.vector.into_iter().map(|val| *val))?;
            Ok(mat)
        }
    }

    impl<N, D> TryFromCv<geo::Translation<N, D>> for core::Mat
    where
        N: na::Scalar + core::DataType,
        D: na::DimName,
        na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, D>,
    {
        type Error = Error;

        fn try_from_cv(translation: geo::Translation<N, D>) -> Result<Self> {
            TryFromCv::try_from_cv(&translation)
        }
    }
}

#[cfg(all(feature = "tch", feature = "image"))]
mod with_tch_image {
    use super::*;
    use std::ops::Deref;

    impl<P, Container> FromCv<&image::ImageBuffer<P, Container>> for tch::Tensor
    where
        P: image::Pixel + 'static,
        P::Subpixel: 'static + tch::kind::Element,
        Container: Deref<Target = [P::Subpixel]>,
    {
        fn from_cv(from: &image::ImageBuffer<P, Container>) -> Self {
            let (width, height) = from.dimensions();
            let height = height as usize;
            let width = width as usize;
            let channels = P::CHANNEL_COUNT as usize;

            let buffer = unsafe {
                let buf_len = channels * height * width;
                let mut buffer: Vec<P::Subpixel> = Vec::with_capacity(buf_len);
                let ptr = buffer.as_mut_ptr();
                from.enumerate_pixels().for_each(|(x, y, pixel)| {
                    let x = x as usize;
                    let y = y as usize;
                    pixel
                        .channels()
                        .iter()
                        .cloned()
                        .enumerate()
                        .for_each(|(c, component)| {
                            *ptr.add(x + width * (y + height * c)) = component;
                        });
                });
                buffer.set_len(buf_len);
                buffer
            };

            Self::of_slice(&buffer).view([channels as i64, height as i64, width as i64])
        }
    }

    impl<P, Container> FromCv<image::ImageBuffer<P, Container>> for tch::Tensor
    where
        P: image::Pixel + 'static,
        P::Subpixel: 'static + tch::kind::Element,
        Container: Deref<Target = [P::Subpixel]>,
    {
        fn from_cv(from: image::ImageBuffer<P, Container>) -> Self {
            Self::from_cv(&from)
        }
    }

    impl TryFromCv<&image::DynamicImage> for tch::Tensor {
        type Error = Error;

        fn try_from_cv(from: &image::DynamicImage) -> Result<Self, Self::Error> {
            use image::DynamicImage;

            let tensor = match from {
                DynamicImage::ImageLuma8(image) => image.into_cv(),
                DynamicImage::ImageLumaA8(image) => image.into_cv(),
                DynamicImage::ImageRgb8(image) => image.into_cv(),
                DynamicImage::ImageRgba8(image) => image.into_cv(),
                DynamicImage::ImageBgr8(image) => image.into_cv(),
                DynamicImage::ImageBgra8(image) => image.into_cv(),
                _ => bail!("cannot convert an image with u16 components to a tensor"),
            };
            Ok(tensor)
        }
    }

    impl TryFromCv<image::DynamicImage> for tch::Tensor {
        type Error = Error;

        fn try_from_cv(from: image::DynamicImage) -> Result<Self, Self::Error> {
            Self::try_from_cv(&from)
        }
    }
}
