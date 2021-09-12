use crate::{common::*, TryFromCv, TryIntoCv};
use image::{buffer::ConvertBuffer, Bgr, Bgra, ColorType, DynamicImage, ImageBuffer};
use opencv::{core as core_cv, prelude::*};
use std::ops::Deref;

type BgrImage = ImageBuffer<Bgr<u8>, Vec<u8>>;
type BgraImage = ImageBuffer<Bgra<u8>, Vec<u8>>;

impl<P, Container> TryFromCv<image::ImageBuffer<P, Container>> for core_cv::Mat
where
    P: image::Pixel + 'static,
    P::Subpixel: 'static,
    Container: Deref<Target = [P::Subpixel]> + Clone,
{
    type Error = Error;
    fn try_from_cv(from: image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl<P, Container> TryFromCv<&image::ImageBuffer<P, Container>> for core_cv::Mat
where
    P: image::Pixel + 'static,
    P::Subpixel: 'static,
    Container: Deref<Target = [P::Subpixel]> + Clone,
{
    type Error = Error;
    fn try_from_cv(from: &image::ImageBuffer<P, Container>) -> Result<Self, Self::Error> {
        let (width, height) = from.dimensions();
        let height = height as usize;
        let width = width as usize;
        let cv_type = match P::COLOR_TYPE {
            ColorType::L8 => core_cv::CV_8UC1,
            ColorType::La8 => core_cv::CV_8UC2,
            ColorType::Bgr8 => core_cv::CV_8UC3,
            ColorType::Bgra8 => core_cv::CV_8UC4,
            typ => bail!("Wrong color type: {:?}", typ),
        };
        let mat = unsafe {
            core_cv::Mat::new_rows_cols_with_data(
                height as i32,
                width as i32,
                cv_type,
                from.as_ptr() as *mut _,
                core_cv::Mat_AUTO_STEP,
            )?
            .try_clone()?
        };
        Ok(mat)
    }
}

impl TryFromCv<&DynamicImage> for core_cv::Mat {
    type Error = Error;

    fn try_from_cv(from: &image::DynamicImage) -> Result<Self, Self::Error> {
        // let mat = from.to_bgra8().try_into_cv()?;
        let mat = match from {
            DynamicImage::ImageLuma8(image) => image.try_into_cv()?,
            DynamicImage::ImageLumaA8(image) => image.try_into_cv()?,
            DynamicImage::ImageBgr8(image) => image.try_into_cv()?,
            DynamicImage::ImageBgra8(image) => image.try_into_cv()?,

            // Reorder color channels to the default BGR format used by OpenCv
            DynamicImage::ImageRgb8(image) => {
                let image: BgrImage = image.convert();
                image.try_into_cv()?
            }
            DynamicImage::ImageRgba8(image) => {
                let image: BgraImage = image.convert();
                image.try_into_cv()?
            }

            // Convert 16-bit data to 8-bit since OpenCV only supports 8-bit in integer
            DynamicImage::ImageLuma16(image) => {
                let image: image::GrayImage = image.convert();
                image.try_into_cv()?
            }
            DynamicImage::ImageLumaA16(image) => {
                let image: image::GrayAlphaImage = image.convert();
                image.try_into_cv()?
            }
            DynamicImage::ImageRgb16(image) => {
                let image: image::RgbImage = image.convert();
                image.try_into_cv()?
            }
            DynamicImage::ImageRgba16(image) => {
                let image: image::RgbaImage = image.convert();
                image.try_into_cv()?
            }
        };
        Ok(mat)
    }
}

impl TryFromCv<DynamicImage> for core_cv::Mat {
    type Error = Error;
    fn try_from_cv(from: DynamicImage) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}
