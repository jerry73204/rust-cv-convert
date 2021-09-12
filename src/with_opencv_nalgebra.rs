use crate::{common::*, FromCv, TryFromCv};
use nalgebra::{self as na, geometry as geo};
use opencv::{calib3d, core as core_cv, prelude::*};

// Note for future maintainers: Since the matrixes need to accommodate any size Matrix, we are using na::OMatrix instead of SMatrix.

/// A pair of rvec and tvec from OpenCV, standing for rotation and translation.
#[derive(Debug, Clone)]
pub struct OpenCvPose<T> {
    pub rvec: T,
    pub tvec: T,
}

impl TryFromCv<OpenCvPose<&core_cv::Point3d>> for geo::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(pose: OpenCvPose<&core_cv::Point3d>) -> Result<Self> {
        let OpenCvPose { rvec, tvec } = pose;
        let rotation = {
            let rvec_mat = {
                let core_cv::Point3_ { x, y, z, .. } = *rvec;
                core_cv::Mat::from_slice(&[x, y, z])?
            };
            let mut rotation_mat = core_cv::Mat::zeros(3, 3, core_cv::CV_64FC1)?.to_mat()?;
            calib3d::rodrigues(&rvec_mat, &mut rotation_mat, &mut core_cv::no_array()?)?;
            let rotation_matrix: na::Matrix3<f64> = TryFromCv::try_from_cv(rotation_mat)?;
            geo::UnitQuaternion::from_matrix(&rotation_matrix)
        };

        let translation = {
            let core_cv::Point3_ { x, y, z } = *tvec;
            geo::Translation3::new(x, y, z)
        };

        let isometry = geo::Isometry3::from_parts(translation, rotation);
        Ok(isometry)
    }
}

impl TryFromCv<&OpenCvPose<core_cv::Point3d>> for geo::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: &OpenCvPose<core_cv::Point3d>) -> Result<Self> {
        let OpenCvPose { rvec, tvec } = from;
        TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })
    }
}

impl TryFromCv<OpenCvPose<core_cv::Point3d>> for geo::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: OpenCvPose<core_cv::Point3d>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl TryFromCv<OpenCvPose<&core_cv::Mat>> for geo::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: OpenCvPose<&core_cv::Mat>) -> Result<Self> {
        let OpenCvPose {
            rvec: rvec_mat,
            tvec: tvec_mat,
        } = from;
        let rvec = core_cv::Point3d::try_from_cv(rvec_mat)?;
        let tvec = core_cv::Point3d::try_from_cv(tvec_mat)?;
        let isometry = TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })?;
        Ok(isometry)
    }
}

impl TryFromCv<&OpenCvPose<core_cv::Mat>> for geo::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: &OpenCvPose<core_cv::Mat>) -> Result<Self> {
        let OpenCvPose { rvec, tvec } = from;
        TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })
    }
}

impl TryFromCv<OpenCvPose<core_cv::Mat>> for geo::Isometry3<f64> {
    type Error = Error;

    fn try_from_cv(from: OpenCvPose<core_cv::Mat>) -> Result<Self> {
        let OpenCvPose { rvec, tvec } = &from;
        TryFromCv::try_from_cv(OpenCvPose { rvec, tvec })
    }
}

impl<T> TryFromCv<&geo::Isometry3<T>> for OpenCvPose<core_cv::Point3_<T>>
where
    T: core_cv::DataType + core_cv::ValidPoint3Type + na::RealField,
{
    type Error = Error;

    fn try_from_cv(from: &geo::Isometry3<T>) -> Result<OpenCvPose<core_cv::Point3_<T>>> {
        let geo::Isometry3 {
            rotation,
            translation,
            ..
        } = from;

        let rvec = {
            let rotation_mat =
                core_cv::Mat::try_from_cv(rotation.to_rotation_matrix().into_inner())?;
            let mut rvec_mat = core_cv::Mat::zeros(3, 1, core_cv::CV_64FC1)?.to_mat()?;
            calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut core_cv::no_array()?)?;
            let rvec = core_cv::Point3_::new(
                *rvec_mat.at_2d::<T>(0, 0)?,
                *rvec_mat.at_2d::<T>(1, 0)?,
                *rvec_mat.at_2d::<T>(2, 0)?,
            );
            rvec
        };
        let tvec = core_cv::Point3_::new(translation.x, translation.y, translation.z);

        Ok(OpenCvPose { rvec, tvec })
    }
}

impl<T> TryFromCv<geo::Isometry3<T>> for OpenCvPose<core_cv::Point3_<T>>
where
    T: core_cv::DataType + core_cv::ValidPoint3Type + na::RealField,
{
    type Error = Error;

    fn try_from_cv(from: geo::Isometry3<T>) -> Result<OpenCvPose<core_cv::Point3_<T>>> {
        TryFromCv::try_from_cv(&from)
    }
}

impl TryFromCv<&geo::Isometry3<f64>> for OpenCvPose<core_cv::Mat> {
    type Error = Error;

    fn try_from_cv(from: &geo::Isometry3<f64>) -> Result<OpenCvPose<core_cv::Mat>> {
        let geo::Isometry3 {
            rotation,
            translation,
            ..
        } = from;

        let rvec = {
            let rotation_mat: Mat =
                TryFromCv::try_from_cv(rotation.to_rotation_matrix().into_inner())?;
            let mut rvec_mat = core_cv::Mat::zeros(3, 1, core_cv::CV_64FC1)?.to_mat()?;
            calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut core_cv::no_array()?)?;
            rvec_mat
        };
        let tvec = core_cv::Mat::from_slice(&[translation.x, translation.y, translation.z])?;

        Ok(OpenCvPose { rvec, tvec })
    }
}

impl TryFromCv<geo::Isometry3<f64>> for OpenCvPose<core_cv::Mat> {
    type Error = Error;

    fn try_from_cv(from: geo::Isometry3<f64>) -> Result<OpenCvPose<core_cv::Mat>> {
        TryFromCv::try_from_cv(&from)
    }
}

impl TryFromCv<&geo::Isometry3<f32>> for OpenCvPose<core_cv::Mat> {
    type Error = Error;

    fn try_from_cv(from: &geo::Isometry3<f32>) -> Result<OpenCvPose<core_cv::Mat>> {
        let geo::Isometry3 {
            rotation,
            translation,
            ..
        } = from;

        let rvec = {
            let rotation_mat = Mat::try_from_cv(rotation.to_rotation_matrix().into_inner())?;
            let mut rvec_mat = Mat::zeros(3, 1, core_cv::CV_32FC1)?.to_mat()?;
            calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut core_cv::no_array()?)?;
            rvec_mat
        };
        let tvec = Mat::from_slice(&[translation.x, translation.y, translation.z])?;

        Ok(OpenCvPose { rvec, tvec })
    }
}

impl TryFromCv<geo::Isometry3<f32>> for OpenCvPose<core_cv::Mat> {
    type Error = Error;

    fn try_from_cv(from: geo::Isometry3<f32>) -> Result<OpenCvPose<core_cv::Mat>> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<N, R, C> TryFromCv<&core_cv::Mat> for na::OMatrix<N, R, C>
where
    N: na::Scalar + core_cv::DataType,
    R: na::Dim,
    C: na::Dim,
    na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, R, C>,
{
    type Error = Error;

    fn try_from_cv(from: &core_cv::Mat) -> Result<Self> {
        let shape = from.size()?;
        {
            let check_height = R::try_to_usize()
                .map(|size| size == shape.height as usize)
                .unwrap_or(true);
            let check_width = C::try_to_usize()
                .map(|size| size == shape.width as usize)
                .unwrap_or(true);
            let has_same_shape = check_height && check_width;
            ensure!(has_same_shape, "input and output matrix shapes differ");
        }

        let rows: Result<Vec<&[N]>, _> = (0..shape.height)
            .map(|row_idx| from.at_row::<N>(row_idx))
            .collect();
        let rows = rows?;
        let values: Vec<N> = rows
            .into_iter()
            .flat_map(|row| row.iter().cloned())
            .collect();

        Ok(Self::from_row_slice_generic(
            R::from_usize(shape.height as usize),
            C::from_usize(shape.width as usize),
            &values,
        ))
    }
}

impl<N, R, C> TryFromCv<core_cv::Mat> for na::OMatrix<N, R, C>
where
    N: na::Scalar + core_cv::DataType,
    R: na::Dim,
    C: na::Dim,
    na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, R, C>,
{
    type Error = Error;

    fn try_from_cv(from: core_cv::Mat) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<N, R, C, S> TryFromCv<&na::Matrix<N, R, C, S>> for core_cv::Mat
where
    N: na::Scalar + core_cv::DataType,
    R: na::Dim,
    C: na::Dim,
    S: na::base::storage::Storage<N, R, C>,
    na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, C, R>,
{
    type Error = Error;

    fn try_from_cv(from: &na::Matrix<N, R, C, S>) -> Result<Self> {
        let nrows = from.nrows();
        let mat =
            core_cv::Mat::from_slice(from.transpose().as_slice())?.reshape(1, nrows as i32)?;
        Ok(mat)
    }
}

impl<N, R, C, S> TryFromCv<na::Matrix<N, R, C, S>> for core_cv::Mat
where
    N: na::Scalar + core_cv::DataType,
    R: na::Dim,
    C: na::Dim,
    S: na::base::storage::Storage<N, R, C>,
    na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, C, R>,
{
    type Error = Error;

    fn try_from_cv(from: na::Matrix<N, R, C, S>) -> Result<Self> {
        TryFromCv::try_from_cv(&from)
    }
}

impl<T> FromCv<&na::Point2<T>> for core_cv::Point_<T>
where
    T: na::Scalar + core_cv::ValidPointType,
{
    fn from_cv(from: &na::Point2<T>) -> Self {
        core_cv::Point_::new(from.x, from.y)
    }
}

impl<T> FromCv<na::Point2<T>> for core_cv::Point_<T>
where
    T: na::Scalar + core_cv::ValidPointType,
{
    fn from_cv(from: na::Point2<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<T> FromCv<&core_cv::Point_<T>> for na::Point2<T>
where
    T: na::Scalar + core_cv::ValidPointType,
{
    fn from_cv(from: &core_cv::Point_<T>) -> Self {
        Self::new(from.x, from.y)
    }
}

impl<T> FromCv<core_cv::Point_<T>> for na::Point2<T>
where
    T: na::Scalar + core_cv::ValidPointType,
{
    fn from_cv(from: core_cv::Point_<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<T> FromCv<&na::Point3<T>> for core_cv::Point3_<T>
where
    T: na::Scalar + core_cv::ValidPoint3Type,
{
    fn from_cv(from: &na::Point3<T>) -> Self {
        Self::new(from.x, from.y, from.z)
    }
}

impl<T> FromCv<na::Point3<T>> for core_cv::Point3_<T>
where
    T: na::Scalar + core_cv::ValidPoint3Type,
{
    fn from_cv(from: na::Point3<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<T> FromCv<&core_cv::Point3_<T>> for na::Point3<T>
where
    T: na::Scalar + core_cv::ValidPoint3Type,
{
    fn from_cv(from: &core_cv::Point3_<T>) -> Self {
        Self::new(from.x, from.y, from.z)
    }
}

impl<T> FromCv<core_cv::Point3_<T>> for na::Point3<T>
where
    T: na::Scalar + core_cv::ValidPoint3Type,
{
    fn from_cv(from: core_cv::Point3_<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<N, const D: usize> TryFromCv<&geo::Translation<N, D>> for core_cv::Mat
where
    N: na::Scalar + core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(translation: &geo::Translation<N, D>) -> Result<Self> {
        let mat = core_cv::Mat::from_exact_iter(translation.vector.into_iter().copied())?;
        Ok(mat)
    }
}

impl<N, const D: usize> TryFromCv<geo::Translation<N, D>> for core_cv::Mat
where
    N: na::Scalar + core_cv::DataType,
{
    type Error = Error;

    fn try_from_cv(translation: geo::Translation<N, D>) -> Result<Self> {
        TryFromCv::try_from_cv(&translation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IntoCv, TryIntoCv};
    use anyhow::Result;
    use approx::abs_diff_eq;
    use nalgebra::{U2, U3};
    use opencv::core as core_cv;
    use rand::prelude::*;
    use std::f64;

    #[test]
    fn convert_opencv_nalgebra() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..5000 {
            // FromCv
            {
                let cv_point = core_cv::Point2d::new(rng.gen(), rng.gen());
                let na_point = na::Point2::<f64>::from_cv(&cv_point);
                ensure!(
                    abs_diff_eq!(cv_point.x, na_point.x) && abs_diff_eq!(cv_point.y, na_point.y),
                    "point conversion failed"
                );
            }

            // IntoCv
            {
                let cv_point = core_cv::Point2d::new(rng.gen(), rng.gen());
                let na_point: na::Point2<f64> = cv_point.into_cv();
                ensure!(
                    abs_diff_eq!(cv_point.x, na_point.x) && abs_diff_eq!(cv_point.y, na_point.y),
                    "point conversion failed"
                );
            }

            // TryFromCv
            {
                let na_mat = na::DMatrix::<f64>::from_vec(
                    2,
                    3,
                    vec![
                        rng.gen(),
                        rng.gen(),
                        rng.gen(),
                        rng.gen(),
                        rng.gen(),
                        rng.gen(),
                    ],
                );
                let cv_mat = core_cv::Mat::try_from_cv(&na_mat)?;
                ensure!(
                    abs_diff_eq!(cv_mat.at_2d(0, 0)?, na_mat.get((0, 0)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(0, 1)?, na_mat.get((0, 1)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(0, 2)?, na_mat.get((0, 2)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 0)?, na_mat.get((1, 0)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 1)?, na_mat.get((1, 1)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 2)?, na_mat.get((1, 2)).unwrap()),
                    "matrix conversion failed"
                );
            }

            // TryIntoCv
            {
                let na_mat = na::DMatrix::<f64>::from_vec(
                    2,
                    3,
                    vec![
                        rng.gen(),
                        rng.gen(),
                        rng.gen(),
                        rng.gen(),
                        rng.gen(),
                        rng.gen(),
                    ],
                );
                let cv_mat: core_cv::Mat = (&na_mat).try_into_cv()?;
                ensure!(
                    abs_diff_eq!(cv_mat.at_2d(0, 0)?, na_mat.get((0, 0)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(0, 1)?, na_mat.get((0, 1)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(0, 2)?, na_mat.get((0, 2)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 0)?, na_mat.get((1, 0)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 1)?, na_mat.get((1, 1)).unwrap())
                        && abs_diff_eq!(cv_mat.at_2d(1, 2)?, na_mat.get((1, 2)).unwrap()),
                    "matrix conversion failed"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn matrix_nalgebra_to_opencv_test() -> Result<()> {
        let input = na::OMatrix::<i32, U3, U2>::from_row_slice(&[1, 2, 3, 4, 5, 6]);
        let (nrows, ncols) = input.shape();
        let output = core_cv::Mat::try_from_cv(input)?;
        let output_shape = output.size()?;
        ensure!(
            output.channels()? == 1
                && nrows == output_shape.height as usize
                && ncols == output_shape.width as usize,
            "the shape does not match"
        );
        Ok(())
    }

    #[test]
    fn matrix_opencv_to_nalgebra_test() -> Result<()> {
        let input = Mat::from_slice_2d(&[&[1, 2, 3], &[4, 5, 6]])?;
        let input_shape = input.size()?;
        let output = na::OMatrix::<i32, U2, U3>::try_from_cv(input)?;
        ensure!(
            output.nrows() == input_shape.height as usize
                && output.ncols() == input_shape.width as usize,
            "the shape does not match"
        );
        Ok(())
    }

    #[test]
    fn rvec_tvec_conversion() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..5000 {
            let orig_isometry = {
                let rotation = na::UnitQuaternion::from_euler_angles(
                    rng.gen_range(0.0..(f64::consts::PI * 2.0)),
                    rng.gen_range(0.0..(f64::consts::PI * 2.0)),
                    rng.gen_range(0.0..(f64::consts::PI * 2.0)),
                );
                let translation = na::Translation3::new(rng.gen(), rng.gen(), rng.gen());
                na::Isometry3::from_parts(translation, rotation)
            };
            let pose = OpenCvPose::<Mat>::try_from_cv(orig_isometry)?;
            let recovered_isometry = na::Isometry3::<f64>::try_from_cv(pose)?;

            ensure!(
                (orig_isometry.to_homogeneous() - recovered_isometry.to_homogeneous()).norm()
                    <= 1e-6,
                "the recovered isometry is not consistent the original isometry"
            );
        }
        Ok(())
    }
}
