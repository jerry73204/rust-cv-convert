#[cfg(feature = "image_0-23")]
mod with_image_0_23 {
    use crate::image;
    use crate::opencv::{core as cv, prelude::*};
    use crate::{common::*, OpenCvElement, TryFromCv, TryIntoCv};
    use std::ops::Deref;

    impl<P, Container> TryFromCv<image::ImageBuffer<P, Container>> for cv::Mat
    where
        P: image::Pixel + 'static,
        P::Subpixel: 'static + OpenCvElement,
        Container: Deref<Target = [P::Subpixel]> + Clone,
    {
        type Error = Error;
        fn try_from_cv(from: image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    impl<P, Container> TryFromCv<&image::ImageBuffer<P, Container>> for cv::Mat
    where
        P: image::Pixel + 'static,
        P::Subpixel: 'static + OpenCvElement,
        Container: Deref<Target = [P::Subpixel]> + Clone,
    {
        type Error = Error;
        fn try_from_cv(from: &image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
            let (width, height) = from.dimensions();
            let cv_type = cv::CV_MAKETYPE(P::Subpixel::DEPTH, P::CHANNEL_COUNT as i32);
            let mat = unsafe {
                cv::Mat::new_rows_cols_with_data(
                    height as i32,
                    width as i32,
                    cv_type,
                    from.as_ptr() as *mut _,
                    cv::Mat_AUTO_STEP,
                )?
                .try_clone()?
            };
            Ok(mat)
        }
    }

    impl TryFromCv<&image::DynamicImage> for cv::Mat {
        type Error = Error;

        fn try_from_cv(from: &image::DynamicImage) -> Result<Self, Self::Error> {
            use image::DynamicImage as D;

            let mat = match from {
                D::ImageLuma8(image) => image.try_into_cv()?,
                D::ImageLumaA8(image) => image.try_into_cv()?,
                D::ImageBgr8(image) => image.try_into_cv()?,
                D::ImageBgra8(image) => image.try_into_cv()?,
                D::ImageRgb8(image) => image.try_into_cv()?,
                D::ImageRgba8(image) => image.try_into_cv()?,
                D::ImageLuma16(image) => image.try_into_cv()?,
                D::ImageLumaA16(image) => image.try_into_cv()?,
                D::ImageRgb16(image) => image.try_into_cv()?,
                D::ImageRgba16(image) => image.try_into_cv()?,
            };
            Ok(mat)
        }
    }

    impl TryFromCv<image::DynamicImage> for cv::Mat {
        type Error = Error;
        fn try_from_cv(from: image::DynamicImage) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }
}

#[cfg(feature = "image_0-24")]
mod with_image_0_24 {
    use crate::image;
    use crate::opencv::{core as cv, prelude::*};
    use crate::{common::*, OpenCvElement, TryFromCv, TryIntoCv};
    use std::ops::Deref;

    impl<P, Container> TryFromCv<image::ImageBuffer<P, Container>> for cv::Mat
    where
        P: image::Pixel + 'static,
        P::Subpixel: 'static + OpenCvElement,
        Container: Deref<Target = [P::Subpixel]> + Clone,
    {
        type Error = Error;
        fn try_from_cv(from: image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    impl<P, Container> TryFromCv<&image::ImageBuffer<P, Container>> for cv::Mat
    where
        P: image::Pixel + 'static,
        P::Subpixel: 'static + OpenCvElement,
        Container: Deref<Target = [P::Subpixel]> + Clone,
    {
        type Error = Error;
        fn try_from_cv(from: &image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
            let (width, height) = from.dimensions();
            let cv_type = cv::CV_MAKETYPE(P::Subpixel::DEPTH, P::CHANNEL_COUNT as i32);
            let mat = unsafe {
                cv::Mat::new_rows_cols_with_data(
                    height as i32,
                    width as i32,
                    cv_type,
                    from.as_ptr() as *mut _,
                    cv::Mat_AUTO_STEP,
                )?
                .try_clone()?
            };
            Ok(mat)
        }
    }

    impl TryFromCv<&image::DynamicImage> for cv::Mat {
        type Error = Error;

        fn try_from_cv(from: &image::DynamicImage) -> Result<Self, Self::Error> {
            use image::DynamicImage as D;

            let mat = match from {
                D::ImageLuma8(image) => image.try_into_cv()?,
                D::ImageLumaA8(image) => image.try_into_cv()?,
                D::ImageRgb8(image) => image.try_into_cv()?,
                D::ImageRgba8(image) => image.try_into_cv()?,
                D::ImageLuma16(image) => image.try_into_cv()?,
                D::ImageLumaA16(image) => image.try_into_cv()?,
                D::ImageRgb16(image) => image.try_into_cv()?,
                D::ImageRgba16(image) => image.try_into_cv()?,
                D::ImageRgb32F(image) => image.try_into_cv()?,
                D::ImageRgba32F(image) => image.try_into_cv()?,
                image => bail!("the color type {:?} is not supported", image.color()),
            };
            Ok(mat)
        }
    }

    impl TryFromCv<image::DynamicImage> for cv::Mat {
        type Error = Error;
        fn try_from_cv(from: image::DynamicImage) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }
}
