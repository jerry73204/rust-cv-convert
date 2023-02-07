use crate::ndarray as nd;
use crate::opencv::core as cv;
use crate::with_opencv::MatExt as _;
use crate::with_opencv::OpenCvElement;
use crate::{common::*, TryFromCv, TryIntoCv};

impl<'a, A, D> TryFromCv<&'a cv::Mat> for nd::ArrayView<'a, A, D>
where
    A: OpenCvElement,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &'a cv::Mat) -> Result<Self, Self::Error> {
        let src_shape = from.shape();
        let array = nd::ArrayViewD::from_shape(src_shape, from.as_slice()?)?;
        let array = array.into_dimensionality()?;
        Ok(array)
    }
}

impl<A, D> TryFromCv<&cv::Mat> for nd::Array<A, D>
where
    A: OpenCvElement + Clone,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv::Mat) -> Result<Self, Self::Error> {
        let src_shape = from.shape();
        let array = nd::ArrayViewD::from_shape(src_shape, from.as_slice()?)?;
        let array = array.into_dimensionality()?;
        let array = array.into_owned();
        Ok(array)
    }
}

impl<A, D> TryFromCv<cv::Mat> for nd::Array<A, D>
where
    A: OpenCvElement + Clone,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: cv::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opencv::prelude::*;
    use itertools::chain;
    use itertools::Itertools as _;
    use rand::prelude::*;

    #[test]
    fn opencv_ndarray_conversion() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..5 {
            let ndim: usize = rng.gen_range(2..=4);
            let shape: Vec<usize> = (0..ndim).map(|_| rng.gen_range(1..=32)).collect();

            let mat = cv::Mat::new_randn_nd::<f32>(&shape)?;
            let view: nd::ArrayViewD<f32> = (&mat).try_into_cv()?;
            let array: nd::ArrayD<f32> = (&mat).try_into_cv()?;

            shape
                .iter()
                .map(|&size| 0..size)
                .multi_cartesian_product()
                .try_for_each(|index| {
                    // opencv expects &[i32] index
                    let index_cv: Vec<_> = index.iter().map(|&size| size as i32).collect();
                    let e1: f32 = *mat.at_nd(&index_cv)?;

                    // converting to ndarray adds an extra dimension for channels.
                    let index_nd: Vec<_> = chain!(index, [0]).collect();
                    let e2 = view[index_nd.as_slice()];

                    // converting to ndarray adds an extra dimension for channels.
                    let e3 = array[index_nd.as_slice()];

                    ensure!(e1 == e2 && e1 == e3);
                    anyhow::Ok(())
                })?;
        }

        Ok(())
    }
}
