use crate::{common::*, TryFromCv, TryIntoCv};
use opencv::{core as core_cv, prelude::*};

use mat_ext::*;
pub use tensor_from_mat::*;
pub use tensor_with_convention::*;

mod tensor_with_convention {
    use super::*;

    /// A tensor with image shape convention that is used to convert to [Tensor](tch::Tensor).
    #[derive(Debug)]
    pub struct TensorAsImage<T>
    where
        T: Borrow<tch::Tensor>,
    {
        pub(crate) tensor: T,
        pub(crate) convention: ShapeConvention,
    }

    /// Describes the image channel order of a [Tensor](tch::Tensor).
    #[derive(Debug, Clone, Copy)]
    pub enum ShapeConvention {
        Whc,
        Hwc,
        Chw,
        Cwh,
    }

    impl<T> TensorAsImage<T>
    where
        T: Borrow<tch::Tensor>,
    {
        pub fn new(tensor: T, convention: ShapeConvention) -> Result<Self> {
            let size = tensor.borrow().size();
            match size.len() {
                2 | 3 => (),
                _ => bail!("tensor size {:?} is not supported", size),
            }
            Ok(Self { tensor, convention })
        }

        pub fn into_inner(self) -> T {
            self.tensor
        }
    }
}

mod mat_ext {
    use super::*;

    pub trait MatExt {
        fn tch_kind_shape_2d(&self) -> Result<(tch::Kind, [i64; 3])>;
        fn tch_kind_shape_nd(&self) -> Result<(tch::Kind, Vec<i64>)>;
    }

    impl MatExt for core_cv::Mat {
        fn tch_kind_shape_2d(&self) -> Result<(tch::Kind, [i64; 3])> {
            let core_cv::Size { height, width } = self.size()?;
            let (kind, n_channels) = match self.typ()? {
                core_cv::CV_8UC1 => (tch::Kind::Uint8, 1),
                core_cv::CV_8UC2 => (tch::Kind::Uint8, 2),
                core_cv::CV_8UC3 => (tch::Kind::Uint8, 3),
                core_cv::CV_8UC4 => (tch::Kind::Uint8, 4),
                core_cv::CV_8SC1 => (tch::Kind::Int8, 1),
                core_cv::CV_8SC2 => (tch::Kind::Int8, 2),
                core_cv::CV_8SC3 => (tch::Kind::Int8, 3),
                core_cv::CV_8SC4 => (tch::Kind::Int8, 4),
                core_cv::CV_16SC1 => (tch::Kind::Int16, 1),
                core_cv::CV_16SC2 => (tch::Kind::Int16, 2),
                core_cv::CV_16SC3 => (tch::Kind::Int16, 3),
                core_cv::CV_16SC4 => (tch::Kind::Int16, 4),
                core_cv::CV_16FC1 => (tch::Kind::Half, 1),
                core_cv::CV_16FC2 => (tch::Kind::Half, 2),
                core_cv::CV_16FC3 => (tch::Kind::Half, 3),
                core_cv::CV_16FC4 => (tch::Kind::Half, 4),
                core_cv::CV_32FC1 => (tch::Kind::Float, 1),
                core_cv::CV_32FC2 => (tch::Kind::Float, 2),
                core_cv::CV_32FC3 => (tch::Kind::Float, 3),
                core_cv::CV_32FC4 => (tch::Kind::Float, 4),
                core_cv::CV_32SC1 => (tch::Kind::Int, 1),
                core_cv::CV_32SC2 => (tch::Kind::Int, 2),
                core_cv::CV_32SC3 => (tch::Kind::Int, 3),
                core_cv::CV_32SC4 => (tch::Kind::Int, 4),
                core_cv::CV_64FC1 => (tch::Kind::Double, 1),
                core_cv::CV_64FC2 => (tch::Kind::Double, 2),
                core_cv::CV_64FC3 => (tch::Kind::Double, 3),
                core_cv::CV_64FC4 => (tch::Kind::Double, 4),
                other => bail!("unsupported Mat type {}", other),
            };
            Ok((kind, [width as i64, height as i64, n_channels as i64]))
        }

        fn tch_kind_shape_nd(&self) -> Result<(tch::Kind, Vec<i64>)> {
            let size: Vec<_> = self
                .mat_size()
                .iter()
                .cloned()
                .map(|dim| dim as i64)
                .chain(iter::once(self.channels()? as i64))
                .collect();

            let kind = match self.typ()? {
                core_cv::CV_8UC1 => tch::Kind::Uint8,
                core_cv::CV_8UC2 => tch::Kind::Uint8,
                core_cv::CV_8UC3 => tch::Kind::Uint8,
                core_cv::CV_8UC4 => tch::Kind::Uint8,
                core_cv::CV_8SC1 => tch::Kind::Int8,
                core_cv::CV_8SC2 => tch::Kind::Int8,
                core_cv::CV_8SC3 => tch::Kind::Int8,
                core_cv::CV_8SC4 => tch::Kind::Int8,
                core_cv::CV_16SC1 => tch::Kind::Int16,
                core_cv::CV_16SC2 => tch::Kind::Int16,
                core_cv::CV_16SC3 => tch::Kind::Int16,
                core_cv::CV_16SC4 => tch::Kind::Int16,
                core_cv::CV_16FC1 => tch::Kind::Half,
                core_cv::CV_16FC2 => tch::Kind::Half,
                core_cv::CV_16FC3 => tch::Kind::Half,
                core_cv::CV_16FC4 => tch::Kind::Half,
                core_cv::CV_32FC1 => tch::Kind::Float,
                core_cv::CV_32FC2 => tch::Kind::Float,
                core_cv::CV_32FC3 => tch::Kind::Float,
                core_cv::CV_32FC4 => tch::Kind::Float,
                core_cv::CV_32SC1 => tch::Kind::Int,
                core_cv::CV_32SC2 => tch::Kind::Int,
                core_cv::CV_32SC3 => tch::Kind::Int,
                core_cv::CV_32SC4 => tch::Kind::Int,
                core_cv::CV_64FC1 => tch::Kind::Double,
                core_cv::CV_64FC2 => tch::Kind::Double,
                core_cv::CV_64FC3 => tch::Kind::Double,
                core_cv::CV_64FC4 => tch::Kind::Double,
                other => bail!("unsupported Mat type {}", other),
            };
            Ok((kind, size))
        }
    }
}

mod tensor_from_mat {
    use super::*;

    /// A [Tensor](tch::Tensor) which data reference borrows from a [Mat](core_cv::Mat). It can be dereferenced to a [Tensor](tch::Tensor).
    #[derive(Debug)]
    pub struct TensorFromMat {
        pub(super) tensor: ManuallyDrop<tch::Tensor>,
        pub(super) mat: ManuallyDrop<core_cv::Mat>,
    }

    impl TensorFromMat {
        pub fn tensor(&self) -> &tch::Tensor {
            &self.tensor
        }
    }

    impl Drop for TensorFromMat {
        fn drop(&mut self) {
            unsafe {
                ManuallyDrop::drop(&mut self.tensor);
                ManuallyDrop::drop(&mut self.mat);
            }
        }
    }

    impl Deref for TensorFromMat {
        type Target = tch::Tensor;

        fn deref(&self) -> &Self::Target {
            self.tensor.deref()
        }
    }

    impl DerefMut for TensorFromMat {
        fn deref_mut(&mut self) -> &mut Self::Target {
            self.tensor.deref_mut()
        }
    }
}

impl TryFromCv<core_cv::Mat> for TensorFromMat {
    type Error = Error;

    fn try_from_cv(from: core_cv::Mat) -> Result<Self, Self::Error> {
        ensure!(from.is_continuous()?, "non-continuous Mat is not supported");

        let (kind, shape) = from.tch_kind_shape_nd()?;
        let strides = {
            let mut strides: Vec<_> = shape
                .iter()
                .rev()
                .cloned()
                .scan(1, |prev, dim| {
                    let stride = *prev;
                    *prev *= dim;
                    Some(stride)
                })
                .collect();
            strides.reverse();
            strides
        };
        // let stride = vec![12, 3, 1];

        let tensor = unsafe {
            let ptr = from.ptr(0)? as *const u8;
            tch::Tensor::f_of_blob(ptr, &shape, &strides, kind, tch::Device::Cpu)?
        };

        Ok(Self {
            tensor: ManuallyDrop::new(tensor),
            mat: ManuallyDrop::new(from),
        })
    }
}

impl TryFromCv<&core_cv::Mat> for tch::Tensor {
    type Error = Error;

    fn try_from_cv(from: &core_cv::Mat) -> Result<Self, Self::Error> {
        ensure!(from.is_continuous()?, "non-continuous Mat is not supported");
        let (kind, shape) = from.tch_kind_shape_nd()?;

        let tensor = unsafe {
            let ptr = from.ptr(0)? as *const u8;
            let slice_size =
                shape.iter().cloned().product::<i64>() as usize * kind.elt_size_in_bytes();
            let slice = slice::from_raw_parts(ptr, slice_size);
            tch::Tensor::f_of_data_size(slice, shape.as_ref(), kind)?
        };

        Ok(tensor)
    }
}

impl TryFromCv<core_cv::Mat> for tch::Tensor {
    type Error = Error;

    fn try_from_cv(from: core_cv::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl<T> TryFromCv<&TensorAsImage<T>> for core_cv::Mat
where
    T: Borrow<tch::Tensor>,
{
    type Error = Error;

    fn try_from_cv(from: &TensorAsImage<T>) -> Result<Self, Self::Error> {
        let TensorAsImage {
            ref tensor,
            convention,
        } = *from;
        let tensor = tensor.borrow();
        let (tensor, [channels, rows, cols]) = match (tensor.size().as_slice(), convention) {
            (&[w, h], ShapeConvention::Whc) => (tensor.f_permute(&[1, 0])?, [1, h, w]),
            (&[h, w], ShapeConvention::Hwc) => (tensor.shallow_clone(), [1, h, w]),
            (&[w, h], ShapeConvention::Cwh) => (tensor.f_permute(&[1, 0])?, [1, h, w]),
            (&[h, w], ShapeConvention::Chw) => (tensor.shallow_clone(), [1, h, w]),
            (&[w, h, c], ShapeConvention::Whc) => (tensor.f_permute(&[1, 0, 2])?, [c, h, w]),
            (&[h, w, c], ShapeConvention::Hwc) => (tensor.shallow_clone(), [c, h, w]),
            (&[c, w, h], ShapeConvention::Cwh) => (tensor.f_permute(&[2, 1, 0])?, [c, h, w]),
            (&[c, h, w], ShapeConvention::Chw) => (tensor.f_permute(&[1, 2, 0])?, [c, h, w]),
            (shape, _convention) => bail!("unsupported tensor shape {:?}", shape),
        };
        let tensor = tensor.f_contiguous()?.f_to_device(tch::Device::Cpu)?;

        let kind = tensor.f_kind()?;
        let typ = match (kind, channels) {
            (tch::Kind::Uint8, 1) => core_cv::CV_8UC1,
            (tch::Kind::Uint8, 2) => core_cv::CV_8UC2,
            (tch::Kind::Uint8, 3) => core_cv::CV_8UC3,
            (tch::Kind::Uint8, 4) => core_cv::CV_8UC4,
            (tch::Kind::Int8, 1) => core_cv::CV_8SC1,
            (tch::Kind::Int8, 2) => core_cv::CV_8SC2,
            (tch::Kind::Int8, 3) => core_cv::CV_8SC3,
            (tch::Kind::Int8, 4) => core_cv::CV_8SC4,
            (tch::Kind::Int16, 1) => core_cv::CV_16SC1,
            (tch::Kind::Int16, 2) => core_cv::CV_16SC2,
            (tch::Kind::Int16, 3) => core_cv::CV_16SC3,
            (tch::Kind::Int16, 4) => core_cv::CV_16SC4,
            (tch::Kind::Half, 1) => core_cv::CV_16FC1,
            (tch::Kind::Half, 2) => core_cv::CV_16FC2,
            (tch::Kind::Half, 3) => core_cv::CV_16FC3,
            (tch::Kind::Half, 4) => core_cv::CV_16FC4,
            (tch::Kind::Int, 1) => core_cv::CV_32SC1,
            (tch::Kind::Int, 2) => core_cv::CV_32SC2,
            (tch::Kind::Int, 3) => core_cv::CV_32SC3,
            (tch::Kind::Int, 4) => core_cv::CV_32SC4,
            (tch::Kind::Float, 1) => core_cv::CV_32FC1,
            (tch::Kind::Float, 2) => core_cv::CV_32FC2,
            (tch::Kind::Float, 3) => core_cv::CV_32FC3,
            (tch::Kind::Float, 4) => core_cv::CV_32FC4,
            (tch::Kind::Double, 1) => core_cv::CV_64FC1,
            (tch::Kind::Double, 2) => core_cv::CV_64FC2,
            (tch::Kind::Double, 3) => core_cv::CV_64FC3,
            (tch::Kind::Double, 4) => core_cv::CV_64FC4,
            (kind, channels) => bail!(
                "unsupported tensor kind {:?} and channels {}",
                kind,
                channels
            ),
        };

        let mat = unsafe {
            core_cv::Mat::new_rows_cols_with_data(
                rows as i32,
                cols as i32,
                typ,
                tensor.data_ptr(),
                /* step = */
                core_cv::Mat_AUTO_STEP,
            )?
            .try_clone()?
        };

        Ok(mat)
    }
}

impl<T> TryFromCv<TensorAsImage<T>> for core_cv::Mat
where
    T: Borrow<tch::Tensor>,
{
    type Error = Error;

    fn try_from_cv(from: TensorAsImage<T>) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl TryFromCv<&tch::Tensor> for core_cv::Mat {
    type Error = Error;

    fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
        let tensor = from.f_contiguous()?.f_to_device(tch::Device::Cpu)?;
        let size: Vec<_> = tensor.size().into_iter().map(|dim| dim as i32).collect();
        let typ = match tensor.f_kind()? {
            tch::Kind::Uint8 => core_cv::CV_8UC1,
            tch::Kind::Int8 => core_cv::CV_8SC1,
            tch::Kind::Int16 => core_cv::CV_16SC1,
            tch::Kind::Half => core_cv::CV_16FC1,
            tch::Kind::Int => core_cv::CV_32SC1,
            tch::Kind::Float => core_cv::CV_32FC1,
            tch::Kind::Double => core_cv::CV_64FC1,
            kind => bail!("unsupported tensor kind {:?}", kind),
        };

        let mat = unsafe { core_cv::Mat::new_nd_with_data(&size, typ, tensor.data_ptr(), None)? };
        Ok(mat)
    }
}

impl TryFromCv<tch::Tensor> for core_cv::Mat {
    type Error = Error;

    fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::IndexOp;

    // const EPSILON: f64 = 1e-8;
    const ROUNDS: usize = 1000;

    #[test]
    fn tensor_mat_conv() -> Result<()> {
        let size = [2, 3, 4, 5];

        for _ in 0..ROUNDS {
            let before = tch::Tensor::randn(size.as_ref(), tch::kind::FLOAT_CPU);
            let mat = core_cv::Mat::try_from_cv(&before)?;
            let after = tch::Tensor::try_from_cv(&mat)?.f_view(size)?;

            // compare Tensor and Mat values
            {
                fn enumerate_reversed_index(dims: &[i64]) -> Vec<Vec<i64>> {
                    match dims {
                        [] => vec![vec![]],
                        [dim, remaining @ ..] => {
                            let dim = *dim;
                            let indexes: Vec<_> = (0..dim)
                                .flat_map(move |val| {
                                    enumerate_reversed_index(remaining).into_iter().map(
                                        move |mut tail| {
                                            tail.push(val);
                                            tail
                                        },
                                    )
                                })
                                .collect();
                            indexes
                        }
                    }
                }

                enumerate_reversed_index(&before.size())
                    .into_iter()
                    .map(|mut index| {
                        index.reverse();
                        index
                    })
                    .try_for_each(|tch_index| -> Result<_> {
                        let cv_index: Vec<_> =
                            tch_index.iter().cloned().map(|val| val as i32).collect();
                        let tch_index: Vec<_> = tch_index
                            .iter()
                            .cloned()
                            .map(|val| Some(tch::Tensor::from(val)))
                            .collect();
                        let tch_val: f32 = before.f_index(&tch_index)?.into();
                        let mat_val: f32 = *mat.at_nd(&cv_index)?;
                        ensure!(tch_val == mat_val, "value mismatch");
                        Ok(())
                    })?;
            }

            // compare original and recovered Tensor values
            ensure!(before == after, "value mismatch",);
        }

        Ok(())
    }

    #[test]
    fn tensor_as_image_and_mat_conv() -> Result<()> {
        for _ in 0..ROUNDS {
            let channels = 3;
            let height = 16;
            let width = 8;

            let before = tch::Tensor::randn(&[channels, height, width], tch::kind::FLOAT_CPU);
            let mat: core_cv::Mat =
                TensorAsImage::new(&before, ShapeConvention::Chw)?.try_into_cv()?;
            let after = tch::Tensor::try_from_cv(&mat)?.f_permute(&[2, 0, 1])?; // hwc -> chw

            // compare Tensor and Mat values
            for row in 0..height {
                for col in 0..width {
                    let pixel: &core_cv::Vec3f = mat.at_2d(row as i32, col as i32)?;
                    let [red, green, blue] = **pixel;
                    ensure!(f32::from(before.i((0, row, col))) == red, "value mismatch");
                    ensure!(
                        f32::from(before.i((1, row, col))) == green,
                        "value mismatch"
                    );
                    ensure!(f32::from(before.i((2, row, col))) == blue, "value mismatch");
                }
            }

            // compare original and recovered Tensor values
            {
                let before_size = before.size();
                let after_size = after.size();
                ensure!(
                    before_size == after_size,
                    "size mismatch: {:?} vs. {:?}",
                    before_size,
                    after_size
                );
                ensure!(before == after, "value mismatch");
            }
        }
        Ok(())
    }

    #[test]
    fn tensor_from_mat_conv() -> Result<()> {
        for _ in 0..ROUNDS {
            let channel = 3;
            let height = 16;
            let width = 8;

            let before = tch::Tensor::randn(&[channel, height, width], tch::kind::FLOAT_CPU);
            let mat: core_cv::Mat =
                TensorAsImage::new(&before, ShapeConvention::Chw)?.try_into_cv()?;
            let after = TensorFromMat::try_from_cv(mat)?; // in hwc

            // compare original and recovered Tensor values
            {
                ensure!(after.size() == [height, width, channel], "size mismatch",);
                ensure!(
                    &before.f_permute(&[1, 2, 0])? == after.tensor(),
                    "value mismatch"
                );
            }
        }
        Ok(())
    }
}
