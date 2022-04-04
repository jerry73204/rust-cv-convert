#[cfg(feature = "image_0-23")]
mod with_image_0_23 {
    use crate::image;
    use crate::tch;
    use crate::{common::*, FromCv, IntoCv, TryFromCv};
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

#[cfg(feature = "image_0-24")]
mod with_image_0_24 {
    use crate::image;
    use crate::tch;
    use crate::{common::*, FromCv, IntoCv, TryFromCv};
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
