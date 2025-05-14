use crate::FromCv;
use opencv::core as core_cv;

impl<T> FromCv<&imageproc::point::Point<T>> for core_cv::Point_<T>
where
    T: num_traits::Num + Copy,
{
    fn from_cv(from: &imageproc::point::Point<T>) -> Self {
        core_cv::Point_::new(from.x, from.y)
    }
}

impl<T> FromCv<imageproc::point::Point<T>> for core_cv::Point_<T>
where
    T: num_traits::Num + Copy,
{
    fn from_cv(from: imageproc::point::Point<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

impl<T> FromCv<&core_cv::Point_<T>> for imageproc::point::Point<T>
where
    T: num_traits::Num + Copy,
{
    fn from_cv(from: &core_cv::Point_<T>) -> Self {
        Self::new(from.x, from.y)
    }
}

impl<T> FromCv<core_cv::Point_<T>> for imageproc::point::Point<T>
where
    T: num_traits::Num + Copy,
{
    fn from_cv(from: core_cv::Point_<T>) -> Self {
        FromCv::from_cv(&from)
    }
}

#[cfg(test)]
mod tests {
    use crate::imageproc;
    use crate::opencv::core as core_cv;
    use crate::{FromCv, IntoCv};
    use anyhow::{ensure, Result};
    use approx::abs_diff_eq;
    use rand::prelude::*;
    use std::f64;

    #[test]
    fn convert_opencv_imageproc() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..5000 {
            // FromCv
            // opencv to imageproc
            {
                let cv_point = core_cv::Point2d::new(rng.gen(), rng.gen());
                let imageproc_point = imageproc::point::Point::<f64>::from_cv(&cv_point);
                ensure!(
                    abs_diff_eq!(cv_point.x, imageproc_point.x)
                        && abs_diff_eq!(cv_point.y, imageproc_point.y),
                    "point conversion failed"
                );
            }

            // imageproc to opencv
            {
                let imageproc_point = imageproc::point::Point::<f64>::new(rng.gen(), rng.gen());
                let cv_point = core_cv::Point2d::from_cv(&imageproc_point);
                ensure!(
                    abs_diff_eq!(imageproc_point.x, cv_point.x)
                        && abs_diff_eq!(imageproc_point.y, cv_point.y),
                    "point conversion failed"
                );
            }

            // IntoCv
            // opencv to imageproc
            {
                let cv_point = core_cv::Point2d::new(rng.gen(), rng.gen());
                let imageproc_point: imageproc::point::Point<f64> = cv_point.into_cv();
                ensure!(
                    abs_diff_eq!(cv_point.x, imageproc_point.x)
                        && abs_diff_eq!(cv_point.y, imageproc_point.y),
                    "point conversion failed"
                );
            }

            // imageproc to opencv
            {
                let imageproc_point = imageproc::point::Point::<f64>::new(rng.gen(), rng.gen());
                let cv_point: core_cv::Point2d = imageproc_point.into_cv();
                ensure!(
                    abs_diff_eq!(imageproc_point.x, cv_point.x)
                        && abs_diff_eq!(imageproc_point.y, cv_point.y),
                    "point conversion failed"
                );
            }
        }
        Ok(())
    }
}
