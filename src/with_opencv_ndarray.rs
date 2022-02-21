use crate::ndarray as nd;
use crate::opencv::{core as cv_core, prelude::*};
use crate::with_opencv::MatExt as _;
use crate::with_opencv::OpenCvElement;
use crate::{common::*, TryFromCv, TryIntoCv};

impl<'a, A, D> TryFromCv<&'a cv_core::Mat> for nd::ArrayView<'a, A, D>
where
    A: OpenCvElement,
    D: nd::Dimension,
    Vec<usize>: Into<nd::StrideShape<D>>,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &'a cv_core::Mat) -> Result<Self, Self::Error> {
        let src_shape = from.shape();
        let array = Self::from_shape(src_shape, from.as_slice()?)?;
        Ok(array)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn mat_ref_to_array_view_conversion() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let ndim: usize = rng.gen_range(0..=6);
            let shape: Vec<usize> = (0..ndim).map(|_| rng.gen_range(0..=100)).collect();

            let mat = Mat::new_randn::<f32>(&shape)?;
            let array: nd::ArrayD<f32> = (&mat).try_into()?;
        }

        Ok(())
    }
}
