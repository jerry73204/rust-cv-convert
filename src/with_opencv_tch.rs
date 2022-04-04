use crate::opencv::{core as core_cv, prelude::*};
use crate::tch;
use crate::{common::*, TchTensorAsImage, TchTensorImageShape, TryFromCv, TryIntoCv};

use mat_ext::*;
pub use tensor_from_mat::*;

mod mat_ext {
    use super::*;

    pub trait MatExt {
        fn tch_kind_shape_2d(&self) -> Result<(tch::Kind, [i64; 3])>;
        fn tch_kind_shape_nd(&self) -> Result<(tch::Kind, Vec<i64>)>;
    }

    impl MatExt for core_cv::Mat {
        fn tch_kind_shape_2d(&self) -> Result<(tch::Kind, [i64; 3])> {
            let core_cv::Size { height, width } = self.size()?;

            let kind = {
                use core_cv as c;
                use tch::Kind as K;

                match self.depth() {
                    c::CV_8U => K::Uint8,
                    c::CV_8S => K::Int8,
                    c::CV_16S => K::Int16,
                    c::CV_32S => K::Int,
                    c::CV_16F => K::Half,
                    c::CV_32F => K::Float,
                    c::CV_64F => K::Double,
                    depth => bail!("unsupported Mat depth {}", depth),
                }
            };
            let channels = self.channels();

            Ok((kind, [width as i64, height as i64, channels as i64]))
        }

        fn tch_kind_shape_nd(&self) -> Result<(tch::Kind, Vec<i64>)> {
            let size: Vec<_> = self
                .mat_size()
                .iter()
                .cloned()
                .map(|dim| dim as i64)
                .chain(iter::once(self.channels() as i64))
                .collect();

            let kind = {
                use core_cv as c;
                use tch::Kind as K;

                match self.depth() {
                    c::CV_8U => K::Uint8,
                    c::CV_8S => K::Int8,
                    c::CV_16S => K::Int16,
                    c::CV_32S => K::Int,
                    c::CV_16F => K::Half,
                    c::CV_32F => K::Float,
                    c::CV_64F => K::Double,
                    depth => bail!("unsupported Mat depth {}", depth),
                }
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
        ensure!(from.is_continuous(), "non-continuous Mat is not supported");

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
        ensure!(from.is_continuous(), "non-continuous Mat is not supported");
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

impl TryFromCv<&TchTensorAsImage> for core_cv::Mat {
    type Error = Error;

    fn try_from_cv(from: &TchTensorAsImage) -> Result<Self, Self::Error> {
        let TchTensorAsImage {
            ref tensor,
            kind: convention,
        } = *from;
        let tensor = tensor.borrow();
        let (tensor, [channels, rows, cols]) = match (tensor.size().as_slice(), convention) {
            (&[w, h], TchTensorImageShape::Whc) => (tensor.f_permute(&[1, 0])?, [1, h, w]),
            (&[h, w], TchTensorImageShape::Hwc) => (tensor.shallow_clone(), [1, h, w]),
            (&[w, h], TchTensorImageShape::Cwh) => (tensor.f_permute(&[1, 0])?, [1, h, w]),
            (&[h, w], TchTensorImageShape::Chw) => (tensor.shallow_clone(), [1, h, w]),
            (&[w, h, c], TchTensorImageShape::Whc) => (tensor.f_permute(&[1, 0, 2])?, [c, h, w]),
            (&[h, w, c], TchTensorImageShape::Hwc) => (tensor.shallow_clone(), [c, h, w]),
            (&[c, w, h], TchTensorImageShape::Cwh) => (tensor.f_permute(&[2, 1, 0])?, [c, h, w]),
            (&[c, h, w], TchTensorImageShape::Chw) => (tensor.f_permute(&[1, 2, 0])?, [c, h, w]),
            (shape, _convention) => bail!("unsupported tensor shape {:?}", shape),
        };
        let tensor = tensor.f_contiguous()?.f_to_device(tch::Device::Cpu)?;

        let kind = tensor.f_kind()?;

        let depth = {
            use core_cv as c;
            use tch::Kind as K;

            match kind {
                K::Uint8 => c::CV_8U,
                K::Int8 => c::CV_8S,
                K::Int16 => c::CV_16S,
                K::Int => c::CV_32S,
                K::Half => c::CV_16F,
                K::Float => c::CV_32F,
                K::Double => c::CV_64F,
                kind => bail!(
                    "Conversion from torch tensor type {:?} to OpenCV is not supported",
                    kind,
                ),
            }
        };
        let typ = core_cv::CV_MAKE_TYPE(depth, channels as i32);

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

impl TryFromCv<TchTensorAsImage> for core_cv::Mat {
    type Error = Error;

    fn try_from_cv(from: TchTensorAsImage) -> Result<Self, Self::Error> {
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
    use crate::tch::{self, IndexOp, Tensor};

    // const EPSILON: f64 = 1e-8;
    const ROUNDS: usize = 1000;

    #[test]
    fn tensor_mat_conv() -> Result<()> {
        let size = [2, 3, 4, 5];

        for _ in 0..ROUNDS {
            let before = Tensor::randn(size.as_ref(), tch::kind::FLOAT_CPU);
            let mat = core_cv::Mat::try_from_cv(&before)?;
            let after = Tensor::try_from_cv(&mat)?.f_view(size)?;

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
                            .map(|val| Some(Tensor::from(val)))
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

            let before = Tensor::randn(&[channels, height, width], tch::kind::FLOAT_CPU);
            let mat: core_cv::Mat =
                TchTensorAsImage::new(before.shallow_clone(), TchTensorImageShape::Chw)?
                    .try_into_cv()?;
            let after = Tensor::try_from_cv(&mat)?.f_permute(&[2, 0, 1])?; // hwc -> chw

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

            let before = Tensor::randn(&[channel, height, width], tch::kind::FLOAT_CPU);
            let mat: core_cv::Mat =
                TchTensorAsImage::new(before.shallow_clone(), TchTensorImageShape::Chw)?
                    .try_into_cv()?;
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
