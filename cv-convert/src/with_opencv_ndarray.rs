use crate::with_opencv::MatExt as _;
use crate::with_opencv::OpenCvElement;
use crate::TryAsRefCv;
use crate::TryFromCv;
use anyhow::{Error, Result};
use ndarray as nd;
use opencv::{core as cv, prelude::*};

impl<'a, A, D> TryAsRefCv<'a, nd::ArrayView<'a, A, D>> for cv::Mat
where
    A: OpenCvElement,
    D: nd::Dimension + 'a,
{
    type Error = anyhow::Error;

    fn try_as_ref_cv(&'a self) -> Result<nd::ArrayView<'a, A, D>, Self::Error> {
        let src_shape = self.size_with_depth();
        let array = nd::ArrayViewD::from_shape(src_shape, self.as_slice()?)?;
        let array = array.into_dimensionality()?;
        Ok(array)
    }
}

impl<A, D> TryFromCv<cv::Mat> for nd::Array<A, D>
where
    A: OpenCvElement + Clone,
    D: nd::Dimension,
{
    type Error = anyhow::Error;

    fn try_from_cv(from: &cv::Mat) -> Result<Self, Self::Error> {
        let src_shape = from.size_with_depth();
        let array = nd::ArrayViewD::from_shape(src_shape, from.as_slice()?)?;
        let array = array.into_dimensionality()?;
        let array = array.into_owned();
        Ok(array)
    }
}

impl<A, S, D> TryFromCv<nd::ArrayBase<S, D>> for cv::Mat
where
    A: cv::DataType,
    S: nd::RawData<Elem = A> + nd::Data,
    D: nd::Dimension,
{
    type Error = Error;

    fn try_from_cv(from: &nd::ArrayBase<S, D>) -> Result<Self> {
        let shape_with_channels: Vec<i32> = from.shape().iter().map(|&sz| sz as i32).collect();
        let (channels, shape) = match shape_with_channels.split_last() {
            Some(split) => split,
            None => {
                return Ok(Mat::default());
            }
        };
        let array = from.as_standard_layout();
        let slice = array.as_slice().unwrap();
        let mat = cv::Mat::from_slice(slice)?.reshape_nd(*channels, shape)?;
        Ok(mat)
    }
}

#[cfg(test)]
mod tests {
    use crate::TryToCv;

    use super::*;
    use anyhow::ensure;
    use itertools::chain;
    use itertools::Itertools as _;
    use rand::prelude::*;

    #[test]
    fn opencv_ndarray_conversion() -> Result<()> {
        let mut rng = rand::thread_rng();

        for _ in 0..5 {
            // Generate a random shape
            let ndim: usize = rng.gen_range(2..=4);
            let shape: Vec<usize> = (0..ndim).map(|_| rng.gen_range(1..=32)).collect();

            let in_mat = cv::Mat::new_randn_nd::<f32>(&shape)?;
            let view: nd::ArrayViewD<f32> = in_mat.try_as_ref_cv()?;
            let array: nd::ArrayD<f32> = in_mat.try_to_cv()?;
            let out_mat: cv::Mat = array.try_to_cv()?;

            shape
                .iter()
                .map(|&size| 0..size)
                .multi_cartesian_product()
                .try_for_each(|index| {
                    // OpenCV expects a &[i32] index.
                    let index_cv: Vec<_> = index.iter().map(|&size| size as i32).collect();
                    let e1: f32 = *in_mat.at_nd(&index_cv)?;

                    // It adds an extra dimension for Mat ->
                    // nd::ArrayView conversion.
                    let index_nd: Vec<_> = chain!(index, [0]).collect();
                    let e2 = view[index_nd.as_slice()];

                    // It adds an extra dimension for Mat -> nd::Array
                    // conversion.
                    let e3 = array[index_nd.as_slice()];

                    // Ensure the path Mat -> nd::Array -> Mat
                    // preserves the values.
                    let e4: f32 = *out_mat.at_nd(&index_cv)?;

                    ensure!(e1 == e2);
                    ensure!(e1 == e3);
                    ensure!(e1 == e4);
                    anyhow::Ok(())
                })?;
        }

        Ok(())
    }
}
