use crate::ToCv;
use opencv::core as core_cv;

impl<T> ToCv<core_cv::Point_<T>> for imageproc::point::Point<T>
where
    T: num_traits::Num + Copy,
{
    fn to_cv(&self) -> core_cv::Point_<T> {
        core_cv::Point_::new(self.x, self.y)
    }
}

impl<T> ToCv<imageproc::point::Point<T>> for core_cv::Point_<T>
where
    T: num_traits::Num + Copy,
{
    fn to_cv(&self) -> imageproc::point::Point<T> {
        imageproc::point::Point::new(self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use crate::ToCv;
    use anyhow::{ensure, Result};
    use approx::abs_diff_eq;
    use opencv::core as core_cv;
    use rand::prelude::*;
    use std::f64;

    #[test]
    fn convert_opencv_imageproc() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..5000 {
            // ToCv
            // opencv to imageproc
            {
                let cv_point = core_cv::Point2d::new(rng.gen(), rng.gen());
                let imageproc_point: imageproc::point::Point<f64> = cv_point.to_cv();
                ensure!(
                    abs_diff_eq!(cv_point.x, imageproc_point.x)
                        && abs_diff_eq!(cv_point.y, imageproc_point.y),
                    "point conversion failed"
                );
            }

            // imageproc to opencv
            {
                let imageproc_point = imageproc::point::Point::<f64>::new(rng.gen(), rng.gen());
                let cv_point = imageproc_point.to_cv();
                ensure!(
                    abs_diff_eq!(imageproc_point.x, cv_point.x)
                        && abs_diff_eq!(imageproc_point.y, cv_point.y),
                    "point conversion failed"
                );
            }

            // ToCv
            // opencv to imageproc
            {
                let cv_point = core_cv::Point2d::new(rng.gen(), rng.gen());
                let imageproc_point: imageproc::point::Point<f64> = cv_point.to_cv();
                ensure!(
                    abs_diff_eq!(cv_point.x, imageproc_point.x)
                        && abs_diff_eq!(cv_point.y, imageproc_point.y),
                    "point conversion failed"
                );
            }

            // imageproc to opencv
            {
                let imageproc_point = imageproc::point::Point::<f64>::new(rng.gen(), rng.gen());
                let cv_point: core_cv::Point2d = imageproc_point.to_cv();
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
