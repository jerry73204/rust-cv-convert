#[cfg(feature = "image_0-23")]
mod with_image_0_23 {
    use crate::image;
    use crate::opencv::{core as cv, prelude::*};
    use crate::with_opencv::MatExt;
    use crate::{common::*, OpenCvElement, TryFromCv, TryIntoCv};
    use std::ops::Deref;

    // ImageBuffer -> Mat
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

    // &ImageBuffer -> Mat
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

    // &DynamicImage -> Mat
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

    // DynamicImage -> Mat
    impl TryFromCv<image::DynamicImage> for cv::Mat {
        type Error = Error;
        fn try_from_cv(from: image::DynamicImage) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    // &Mat -> gray ImageBuffer
    impl<T> TryFromCv<&cv::Mat> for image::ImageBuffer<image::Luma<T>, Vec<T>>
    where
        image::Luma<T>: image::Pixel,
        T: OpenCvElement + image::Primitive + DataType + 'static,
    {
        type Error = Error;

        fn try_from_cv(from: &cv::Mat) -> Result<Self, Self::Error> {
            let rows = from.rows();
            let cols = from.rows();
            ensure!(
                rows != -1 && cols != -1,
                "Mat with more than 2 dimensions is not supported."
            );

            let depth = from.depth();
            let n_channels = from.channels();
            let width = cols as u32;
            let height = rows as u32;

            ensure!(
                n_channels == 1,
                "Unable to convert a multi-channel Mat to a gray image"
            );
            ensure!(depth == T::DEPTH, "Subpixel type is not supported");

            let image = mat_to_image_buffer_gray::<T>(from, width, height);
            Ok(image)
        }
    }

    // Mat -> gray ImageBuffer
    impl<T> TryFromCv<cv::Mat> for image::ImageBuffer<image::Luma<T>, Vec<T>>
    where
        image::Luma<T>: image::Pixel,
        T: OpenCvElement + image::Primitive + DataType + 'static,
    {
        type Error = Error;

        fn try_from_cv(from: cv::Mat) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    // &Mat -> rgb ImageBuffer
    impl<T> TryFromCv<&cv::Mat> for image::ImageBuffer<image::Rgb<T>, Vec<T>>
    where
        image::Rgb<T>: image::Pixel<Subpixel = T>,
        T: OpenCvElement + image::Primitive + DataType + 'static,
    {
        type Error = Error;

        fn try_from_cv(from: &cv::Mat) -> Result<Self, Self::Error> {
            let rows = from.rows();
            let cols = from.rows();
            ensure!(
                rows != -1 && cols != -1,
                "Mat with more than 2 dimensions is not supported."
            );

            let depth = from.depth();
            let n_channels = from.channels();
            let width = cols as u32;
            let height = rows as u32;

            ensure!(
                n_channels == 3,
                "Expect 3 channels, but get {n_channels} channels"
            );
            ensure!(depth == T::DEPTH, "Subpixel type is not supported");

            let image = mat_to_image_buffer_rgb::<T>(from, width, height);
            Ok(image)
        }
    }

    // Mat -> rgb ImageBuffer
    impl<T> TryFromCv<cv::Mat> for image::ImageBuffer<image::Rgb<T>, Vec<T>>
    where
        image::Rgb<T>: image::Pixel<Subpixel = T>,
        T: OpenCvElement + image::Primitive + DataType + 'static,
    {
        type Error = Error;

        fn try_from_cv(from: cv::Mat) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    // Utility functions

    fn mat_to_image_buffer_gray<T>(
        mat: &cv::Mat,
        width: u32,
        height: u32,
    ) -> image::ImageBuffer<image::Luma<T>, Vec<T>>
    where
        T: image::Primitive + OpenCvElement + DataType + 'static,
    {
        type Image<T> = image::ImageBuffer<image::Luma<T>, Vec<T>>;

        match mat.as_slice::<T>() {
            Ok(slice) => Image::<T>::from_vec(width, height, slice.to_vec()).unwrap(),
            Err(_) => Image::<T>::from_fn(width, height, |col, row| {
                let pixel: T = *mat.at_2d(row as i32, col as i32).unwrap();
                image::Luma([pixel])
            }),
        }
    }

    fn mat_to_image_buffer_rgb<T>(
        mat: &cv::Mat,
        width: u32,
        height: u32,
    ) -> image::ImageBuffer<image::Rgb<T>, Vec<T>>
    where
        T: image::Primitive + OpenCvElement + DataType + 'static,
        image::Rgb<T>: image::Pixel<Subpixel = T>,
    {
        type Image<T> = image::ImageBuffer<image::Rgb<T>, Vec<T>>;

        match mat.as_slice::<T>() {
            Ok(slice) => Image::<T>::from_vec(width, height, slice.to_vec()).unwrap(),
            Err(_) => Image::<T>::from_fn(width, height, |col, row| {
                let cv::Point3_::<T> { x, y, z } = *mat.at_2d(row as i32, col as i32).unwrap();
                image::Rgb([x, y, z])
            }),
        }
    }
}

#[cfg(feature = "image_0-24")]
mod with_image_0_24 {
    use crate::image;
    use crate::opencv::{core as cv, prelude::*};
    use crate::with_opencv::MatExt;
    use crate::{common::*, OpenCvElement, TryFromCv, TryIntoCv};
    use std::ops::Deref;

    // ImageBuffer -> Mat
    impl<P, Container> TryFromCv<image::ImageBuffer<P, Container>> for cv::Mat
    where
        P: image::Pixel,
        P::Subpixel: OpenCvElement,
        Container: Deref<Target = [P::Subpixel]> + Clone,
    {
        type Error = Error;
        fn try_from_cv(from: image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    // &ImageBuffer -> Mat
    impl<P, Container> TryFromCv<&image::ImageBuffer<P, Container>> for cv::Mat
    where
        P: image::Pixel,
        P::Subpixel: OpenCvElement,
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

    // &DynamicImage -> Mat
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

    // DynamicImage -> Mat
    impl TryFromCv<image::DynamicImage> for cv::Mat {
        type Error = Error;
        fn try_from_cv(from: image::DynamicImage) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    // &Mat -> DynamicImage
    impl TryFromCv<&cv::Mat> for image::DynamicImage {
        type Error = Error;

        fn try_from_cv(from: &cv::Mat) -> Result<Self, Self::Error> {
            let rows = from.rows();
            let cols = from.rows();
            ensure!(
                rows != -1 && cols != -1,
                "Mat with more than 2 dimensions is not supported."
            );

            let depth = from.depth();
            let n_channels = from.channels();
            let width = cols as u32;
            let height = rows as u32;

            let image: image::DynamicImage = match (depth, n_channels) {
                (cv::CV_8U, 1) => mat_to_image_buffer_gray::<u8>(from, width, height).into(),
                (cv::CV_16U, 1) => mat_to_image_buffer_gray::<u16>(from, width, height).into(),
                (cv::CV_8U, 3) => mat_to_image_buffer_rgb::<u8>(from, width, height).into(),
                (cv::CV_16U, 3) => mat_to_image_buffer_rgb::<u16>(from, width, height).into(),
                (cv::CV_32F, 3) => mat_to_image_buffer_rgb::<f32>(from, width, height).into(),
                _ => bail!("Mat of type {} is not supported", from.type_name()),
            };

            Ok(image)
        }
    }

    // Mat -> DynamicImage
    impl TryFromCv<cv::Mat> for image::DynamicImage {
        type Error = Error;

        fn try_from_cv(from: cv::Mat) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    // &Mat -> gray ImageBuffer
    impl<T> TryFromCv<&cv::Mat> for image::ImageBuffer<image::Luma<T>, Vec<T>>
    where
        image::Luma<T>: image::Pixel,
        T: OpenCvElement + image::Primitive + DataType,
    {
        type Error = Error;

        fn try_from_cv(from: &cv::Mat) -> Result<Self, Self::Error> {
            let rows = from.rows();
            let cols = from.rows();
            ensure!(
                rows != -1 && cols != -1,
                "Mat with more than 2 dimensions is not supported."
            );

            let depth = from.depth();
            let n_channels = from.channels();
            let width = cols as u32;
            let height = rows as u32;

            ensure!(
                n_channels == 1,
                "Unable to convert a multi-channel Mat to a gray image"
            );
            ensure!(depth == T::DEPTH, "Subpixel type is not supported");

            let image = mat_to_image_buffer_gray::<T>(from, width, height);
            Ok(image)
        }
    }

    // Mat -> gray ImageBuffer
    impl<T> TryFromCv<cv::Mat> for image::ImageBuffer<image::Luma<T>, Vec<T>>
    where
        image::Luma<T>: image::Pixel,
        T: OpenCvElement + image::Primitive + DataType,
    {
        type Error = Error;

        fn try_from_cv(from: cv::Mat) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    // &Mat -> rgb ImageBuffer
    impl<T> TryFromCv<&cv::Mat> for image::ImageBuffer<image::Rgb<T>, Vec<T>>
    where
        image::Rgb<T>: image::Pixel<Subpixel = T>,
        T: OpenCvElement + image::Primitive + DataType,
    {
        type Error = Error;

        fn try_from_cv(from: &cv::Mat) -> Result<Self, Self::Error> {
            let rows = from.rows();
            let cols = from.rows();
            ensure!(
                rows != -1 && cols != -1,
                "Mat with more than 2 dimensions is not supported."
            );

            let depth = from.depth();
            let n_channels = from.channels();
            let width = cols as u32;
            let height = rows as u32;

            ensure!(
                n_channels == 3,
                "Expect 3 channels, but get {n_channels} channels"
            );
            ensure!(depth == T::DEPTH, "Subpixel type is not supported");

            let image = mat_to_image_buffer_rgb::<T>(from, width, height);
            Ok(image)
        }
    }

    // Mat -> rgb ImageBuffer
    impl<T> TryFromCv<cv::Mat> for image::ImageBuffer<image::Rgb<T>, Vec<T>>
    where
        image::Rgb<T>: image::Pixel<Subpixel = T>,
        T: OpenCvElement + image::Primitive + DataType,
    {
        type Error = Error;

        fn try_from_cv(from: cv::Mat) -> Result<Self, Self::Error> {
            (&from).try_into_cv()
        }
    }

    // Utility functions

    fn mat_to_image_buffer_gray<T>(
        mat: &cv::Mat,
        width: u32,
        height: u32,
    ) -> image::ImageBuffer<image::Luma<T>, Vec<T>>
    where
        T: image::Primitive + OpenCvElement + DataType,
    {
        type Image<T> = image::ImageBuffer<image::Luma<T>, Vec<T>>;

        match mat.as_slice::<T>() {
            Ok(slice) => Image::<T>::from_vec(width, height, slice.to_vec()).unwrap(),
            Err(_) => Image::<T>::from_fn(width, height, |col, row| {
                let pixel: T = *mat.at_2d(row as i32, col as i32).unwrap();
                image::Luma([pixel])
            }),
        }
    }

    fn mat_to_image_buffer_rgb<T>(
        mat: &cv::Mat,
        width: u32,
        height: u32,
    ) -> image::ImageBuffer<image::Rgb<T>, Vec<T>>
    where
        T: image::Primitive + OpenCvElement + DataType,
        image::Rgb<T>: image::Pixel<Subpixel = T>,
    {
        type Image<T> = image::ImageBuffer<image::Rgb<T>, Vec<T>>;

        match mat.as_slice::<T>() {
            Ok(slice) => Image::<T>::from_vec(width, height, slice.to_vec()).unwrap(),
            Err(_) => Image::<T>::from_fn(width, height, |col, row| {
                let cv::Point3_::<T> { x, y, z } = *mat.at_2d(row as i32, col as i32).unwrap();
                image::Rgb([x, y, z])
            }),
        }
    }
}
