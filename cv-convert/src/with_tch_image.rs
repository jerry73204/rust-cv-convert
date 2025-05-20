use crate::image;
use crate::tch;
use crate::{TchTensorAsImage, TchTensorImageShape, ToCv, TryToCv};
use anyhow::{bail, Error, Result};
use std::ops::Deref;

impl<P, Container> ToCv<TchTensorAsImage> for image::ImageBuffer<P, Container>
where
    P: image::Pixel + 'static,
    P::Subpixel: 'static + tch::kind::Element,
    Container: Deref<Target = [P::Subpixel]>,
{
    fn to_cv(&self) -> TchTensorAsImage {
        let (width, height) = self.dimensions();
        let channels = P::CHANNEL_COUNT;
        let tensor =
            tch::Tensor::from_slice(&**self).view([width as i64, height as i64, channels as i64]);
        TchTensorAsImage {
            tensor,
            kind: TchTensorImageShape::Whc,
        }
    }
}

impl TryToCv<TchTensorAsImage> for image::DynamicImage {
    type Error = Error;

    fn try_to_cv(&self) -> Result<TchTensorAsImage, Self::Error> {
        use image::DynamicImage as D;

        let tensor = match self {
            D::ImageLuma8(image) => image.to_cv(),
            D::ImageLumaA8(image) => image.to_cv(),
            D::ImageRgb8(image) => image.to_cv(),
            D::ImageRgba8(image) => image.to_cv(),
            D::ImageRgb32F(image) => image.to_cv(),
            D::ImageRgba32F(image) => image.to_cv(),
            _ => bail!("the color type {:?} is not supported", self.color()),
        };
        Ok(tensor)
    }
}
