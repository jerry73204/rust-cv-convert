use crate::{ToCv, TryToCv};
use anyhow::{ensure, Error, Result};
use nalgebra::{self as na, geometry as geo};
use opencv::{calib3d, core as core_cv, prelude::*};

// Note for future maintainers: Since the matrixes need to accommodate any size Matrix, we are using na::OMatrix instead of SMatrix.

/// A pair of rvec and tvec from OpenCV, standing for rotation and translation.
#[derive(Debug, Clone)]
pub struct OpenCvPose<T> {
    pub rvec: T,
    pub tvec: T,
}

impl TryToCv<geo::Isometry3<f64>> for OpenCvPose<&core_cv::Point3d> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<geo::Isometry3<f64>, Self::Error> {
        let OpenCvPose { rvec, tvec } = *self;
        let rotation = {
            let rvec_mat = {
                let core_cv::Point3_ { x, y, z, .. } = *rvec;
                core_cv::Mat::from_slice(&[x, y, z])?
            };
            let mut rotation_mat = core_cv::Mat::zeros(3, 3, core_cv::CV_64FC1)?.to_mat()?;
            calib3d::rodrigues(&rvec_mat, &mut rotation_mat, &mut core_cv::Mat::default())?;
            let rotation_matrix: na::Matrix3<f64> = rotation_mat.try_to_cv()?;
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

impl TryToCv<geo::Isometry3<f64>> for OpenCvPose<core_cv::Point3d> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<geo::Isometry3<f64>, Self::Error> {
        let OpenCvPose { rvec, tvec } = self;
        OpenCvPose { rvec, tvec }.try_to_cv()
    }
}

impl TryToCv<geo::Isometry3<f64>> for OpenCvPose<&core_cv::Mat> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<geo::Isometry3<f64>, Self::Error> {
        let OpenCvPose {
            rvec: rvec_mat,
            tvec: tvec_mat,
        } = *self;
        let rvec: core_cv::Point3d = rvec_mat.try_to_cv()?;
        let tvec: core_cv::Point3d = tvec_mat.try_to_cv()?;
        let isometry = (&OpenCvPose {
            rvec: &rvec,
            tvec: &tvec,
        })
            .try_to_cv()?;
        Ok(isometry)
    }
}

impl TryToCv<geo::Isometry3<f64>> for OpenCvPose<core_cv::Mat> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<geo::Isometry3<f64>, Self::Error> {
        let OpenCvPose { rvec, tvec } = self;
        OpenCvPose { rvec, tvec }.try_to_cv()
    }
}

impl<T> TryToCv<OpenCvPose<core_cv::Point3_<T>>> for geo::Isometry3<T>
where
    T: core_cv::DataType + na::RealField,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<OpenCvPose<core_cv::Point3_<T>>, Self::Error> {
        let geo::Isometry3 {
            rotation,
            translation,
            ..
        } = self;

        let rvec = {
            let rotation_mat = rotation.to_rotation_matrix().into_inner().try_to_cv()?;
            let mut rvec_mat = core_cv::Mat::zeros(3, 1, core_cv::CV_64FC1)?.to_mat()?;
            calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut core_cv::Mat::default())?;
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

impl TryToCv<OpenCvPose<core_cv::Mat>> for geo::Isometry3<f64> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<OpenCvPose<core_cv::Mat>, Self::Error> {
        let geo::Isometry3 {
            rotation,
            translation,
            ..
        } = self;

        let rvec = {
            let rotation_mat: Mat = rotation.to_rotation_matrix().into_inner().try_to_cv()?;
            let mut rvec_mat = core_cv::Mat::zeros(3, 1, core_cv::CV_64FC1)?.to_mat()?;
            calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut core_cv::Mat::default())?;
            rvec_mat
        };
        let tvec = core_cv::Mat::from_slice(&[translation.x, translation.y, translation.z])?;

        Ok(OpenCvPose { rvec, tvec })
    }
}

impl TryToCv<OpenCvPose<core_cv::Mat>> for geo::Isometry3<f32> {
    type Error = Error;

    fn try_to_cv(&self) -> Result<OpenCvPose<core_cv::Mat>, Self::Error> {
        let geo::Isometry3 {
            rotation,
            translation,
            ..
        } = self;

        let rvec = {
            let rotation_mat = rotation.to_rotation_matrix().into_inner().try_to_cv()?;
            let mut rvec_mat = Mat::zeros(3, 1, core_cv::CV_32FC1)?.to_mat()?;
            calib3d::rodrigues(&rotation_mat, &mut rvec_mat, &mut core_cv::Mat::default())?;
            rvec_mat
        };
        let tvec = Mat::from_slice(&[translation.x, translation.y, translation.z])?
            .t()?
            .to_mat()?;

        Ok(OpenCvPose { rvec, tvec })
    }
}

impl<N, R, C> TryToCv<na::OMatrix<N, R, C>> for core_cv::Mat
where
    N: na::Scalar + core_cv::DataType,
    R: na::Dim,
    C: na::Dim,
    na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, R, C>,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<na::OMatrix<N, R, C>, Self::Error> {
        let shape = self.size()?;
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
            .map(|row_idx| self.at_row::<N>(row_idx))
            .collect();
        let rows = rows?;
        let values: Vec<N> = rows
            .into_iter()
            .flat_map(|row| row.iter().cloned())
            .collect();

        Ok(na::OMatrix::<N, R, C>::from_row_slice_generic(
            R::from_usize(shape.height as usize),
            C::from_usize(shape.width as usize),
            &values,
        ))
    }
}

impl<N, R, C, S> TryToCv<core_cv::Mat> for na::Matrix<N, R, C, S>
where
    N: na::Scalar + core_cv::DataType,
    R: na::Dim,
    C: na::Dim,
    S: na::base::storage::Storage<N, R, C>,
    na::base::default_allocator::DefaultAllocator: na::base::allocator::Allocator<N, C, R>,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<core_cv::Mat, Self::Error> {
        let nrows = self.nrows();
        let mat =
            core_cv::Mat::from_slice(self.transpose().as_slice())?.reshape(1, nrows as i32)?;
        Ok(mat)
    }
}

impl<T> ToCv<core_cv::Point_<T>> for na::Point2<T>
where
    T: na::Scalar + Copy,
{
    fn to_cv(&self) -> core_cv::Point_<T> {
        core_cv::Point_::new(self.x, self.y)
    }
}

impl<T> ToCv<na::Point2<T>> for core_cv::Point_<T>
where
    T: na::Scalar + Copy,
{
    fn to_cv(&self) -> na::Point2<T> {
        na::Point2::new(self.x, self.y)
    }
}

impl<T> ToCv<core_cv::Point3_<T>> for na::Point3<T>
where
    T: na::Scalar + Copy,
{
    fn to_cv(&self) -> core_cv::Point3_<T> {
        core_cv::Point3_::new(self.x, self.y, self.z)
    }
}

impl<T> ToCv<na::Point3<T>> for core_cv::Point3_<T>
where
    T: na::Scalar + Copy,
{
    fn to_cv(&self) -> na::Point3<T> {
        na::Point3::new(self.x, self.y, self.z)
    }
}

impl<N, const D: usize> TryToCv<core_cv::Mat> for geo::Translation<N, D>
where
    N: na::Scalar + core_cv::DataType,
{
    type Error = Error;

    fn try_to_cv(&self) -> Result<core_cv::Mat, Self::Error> {
        let mat = core_cv::Mat::from_exact_iter(self.vector.into_iter().copied())?;
        Ok(mat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ToCv, TryToCv};
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
            // ToCv
            {
                let cv_point = core_cv::Point2d::new(rng.gen(), rng.gen());
                let na_point: na::Point2<f64> = cv_point.to_cv();
                ensure!(
                    abs_diff_eq!(cv_point.x, na_point.x) && abs_diff_eq!(cv_point.y, na_point.y),
                    "point conversion failed"
                );
            }

            // ToCv
            {
                let cv_point = core_cv::Point2d::new(rng.gen(), rng.gen());
                let na_point: na::Point2<f64> = cv_point.to_cv();
                ensure!(
                    abs_diff_eq!(cv_point.x, na_point.x) && abs_diff_eq!(cv_point.y, na_point.y),
                    "point conversion failed"
                );
            }

            // ToCv
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
                let cv_mat = na_mat.try_to_cv()?;
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

            // ToCv
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
                let cv_mat: core_cv::Mat = na_mat.try_to_cv()?;
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
        let output: Mat = input.try_to_cv()?;
        let output_shape = output.size()?;
        ensure!(
            output.channels() == 1
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
        let output: na::OMatrix<i32, U2, U3> = input.try_to_cv()?;
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
            let pose: OpenCvPose<core_cv::Mat> = orig_isometry.try_to_cv()?;
            let recovered_isometry: geo::Isometry3<f64> = pose.try_to_cv()?;

            ensure!(
                (orig_isometry.to_homogeneous() - recovered_isometry.to_homogeneous()).norm()
                    <= 1e-6,
                "the recovered isometry is not consistent the original isometry"
            );
        }
        Ok(())
    }
}
