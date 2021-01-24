use crate::{common::*, TryFromCv, TryIntoCv};
use opencv::{core, platform_types, prelude::*};

use mat_ext::*;
pub use tensor_as_image::*;
pub use tensor_from_mat::*;

mod tensor_as_image {
    use super::*;

    #[derive(Debug)]
    pub struct TensorAsImage<T>
    where
        T: Borrow<tch::Tensor>,
    {
        pub(crate) tensor: T,
        pub(crate) convention: ShapeConvention,
    }

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
        fn tch_kind_shape(&self) -> Result<(tch::Kind, [i64; 3])>;
    }

    impl MatExt for core::Mat {
        fn tch_kind_shape(&self) -> Result<(tch::Kind, [i64; 3])> {
            let core::Size { height, width } = self.size()?;
            let (kind, n_channels) = match self.typ()? {
                core::CV_8UC1 => (tch::Kind::Uint8, 1),
                core::CV_8UC2 => (tch::Kind::Uint8, 2),
                core::CV_8UC3 => (tch::Kind::Uint8, 3),
                core::CV_8UC4 => (tch::Kind::Uint8, 4),
                core::CV_8SC1 => (tch::Kind::Int8, 1),
                core::CV_8SC2 => (tch::Kind::Int8, 2),
                core::CV_8SC3 => (tch::Kind::Int8, 3),
                core::CV_8SC4 => (tch::Kind::Int8, 4),
                core::CV_16SC1 => (tch::Kind::Int16, 1),
                core::CV_16SC2 => (tch::Kind::Int16, 2),
                core::CV_16SC3 => (tch::Kind::Int16, 3),
                core::CV_16SC4 => (tch::Kind::Int16, 4),
                core::CV_16FC1 => (tch::Kind::Half, 1),
                core::CV_16FC2 => (tch::Kind::Half, 2),
                core::CV_16FC3 => (tch::Kind::Half, 3),
                core::CV_16FC4 => (tch::Kind::Half, 4),
                core::CV_32FC1 => (tch::Kind::Float, 1),
                core::CV_32FC2 => (tch::Kind::Float, 2),
                core::CV_32FC3 => (tch::Kind::Float, 3),
                core::CV_32FC4 => (tch::Kind::Float, 4),
                core::CV_32SC1 => (tch::Kind::Int, 1),
                core::CV_32SC2 => (tch::Kind::Int, 2),
                core::CV_32SC3 => (tch::Kind::Int, 3),
                core::CV_32SC4 => (tch::Kind::Int, 4),
                core::CV_64FC1 => (tch::Kind::Double, 1),
                core::CV_64FC2 => (tch::Kind::Double, 2),
                core::CV_64FC3 => (tch::Kind::Double, 3),
                core::CV_64FC4 => (tch::Kind::Double, 4),
                other => bail!("unsupported Mat type {}", other),
            };
            Ok((kind, [width as i64, height as i64, n_channels as i64]))
        }
    }
}

mod tensor_from_mat {
    use super::*;

    pub struct TensorFromMat {
        pub(super) tensor: ManuallyDrop<tch::Tensor>,
        pub(super) mat: ManuallyDrop<core::Mat>,
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

impl TryFromCv<core::Mat> for TensorFromMat {
    type Error = Error;

    fn try_from_cv(from: core::Mat) -> Result<Self, Self::Error> {
        ensure!(from.is_continuous()?, "input Mat memory is not continuous");

        let (kind, shape) = from.tch_kind_shape()?;

        let tensor = unsafe {
            let ptr = from.ptr(0)? as *const u8 as *mut u8;
            tch::Tensor::f_of_blob(ptr, shape.as_ref(), &[0, 0, 0], kind, tch::Device::Cpu)?
        };

        Ok(Self {
            tensor: ManuallyDrop::new(tensor),
            mat: ManuallyDrop::new(from),
        })
    }
}

impl TryFromCv<&core::Mat> for tch::Tensor {
    type Error = Error;

    fn try_from_cv(from: &core::Mat) -> Result<Self, Self::Error> {
        let (kind, shape) = from.tch_kind_shape()?;
        let [w, h, c] = shape;

        let tensor = unsafe {
            if from.is_continuous()? {
                let ptr = from.ptr(0)? as *const u8;
                let slice_size = (w * h * c) as usize * kind.elt_size_in_bytes();
                let slice = slice::from_raw_parts(ptr, slice_size);
                tch::Tensor::f_of_data_size(slice, shape.as_ref(), kind)?
            } else {
                let bytes: Vec<_> = (0..(w as i32))
                    .map(|row_index| -> Result<_> {
                        let ptr = from.ptr(row_index)? as *const u8;
                        let slice_size = (h * c) as usize * kind.elt_size_in_bytes();
                        let slice = slice::from_raw_parts(ptr, slice_size);
                        Ok(slice)
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .flat_map(|row| row.iter().cloned())
                    .collect();
                tch::Tensor::f_of_data_size(&bytes, shape.as_ref(), kind)?
            }
        };

        Ok(tensor)
    }
}

impl TryFromCv<core::Mat> for tch::Tensor {
    type Error = Error;

    fn try_from_cv(from: core::Mat) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl<T> TryFromCv<&TensorAsImage<T>> for core::Mat
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
        let (tensor, [channels, cols, rows]) = match (tensor.size().as_slice(), convention) {
            (&[w, h], ShapeConvention::Whc) => (tensor.shallow_clone(), [1, h, w]),
            (&[h, w], ShapeConvention::Hwc) => (tensor.f_permute(&[1, 0])?, [1, h, w]),
            (&[w, h], ShapeConvention::Cwh) => (tensor.shallow_clone(), [1, h, w]),
            (&[h, w], ShapeConvention::Chw) => (tensor.f_permute(&[1, 0])?, [1, h, w]),
            (&[w, h, c], ShapeConvention::Whc) => (tensor.shallow_clone(), [c, h, w]),
            (&[h, w, c], ShapeConvention::Hwc) => (tensor.f_permute(&[1, 0, 2])?, [c, h, w]),
            (&[c, w, h], ShapeConvention::Cwh) => (tensor.f_permute(&[1, 2, 0])?, [c, h, w]),
            (&[c, h, w], ShapeConvention::Chw) => (tensor.f_permute(&[2, 1, 0])?, [c, h, w]),
            (shape, _convention) => bail!("unsupported tensor shape {:?}", shape),
        };
        let tensor = tensor.f_contiguous()?.f_to_device(tch::Device::Cpu)?;

        let typ = match (tensor.f_kind()?, channels) {
            (tch::Kind::Uint8, 1) => core::CV_8UC1,
            (tch::Kind::Uint8, 2) => core::CV_8UC2,
            (tch::Kind::Uint8, 3) => core::CV_8UC3,
            (tch::Kind::Uint8, 4) => core::CV_8UC4,
            (tch::Kind::Int8, 1) => core::CV_8SC1,
            (tch::Kind::Int8, 2) => core::CV_8SC2,
            (tch::Kind::Int8, 3) => core::CV_8SC3,
            (tch::Kind::Int8, 4) => core::CV_8SC4,
            (tch::Kind::Int16, 1) => core::CV_16SC1,
            (tch::Kind::Int16, 2) => core::CV_16SC2,
            (tch::Kind::Int16, 3) => core::CV_16SC3,
            (tch::Kind::Int16, 4) => core::CV_16SC4,
            (tch::Kind::Half, 1) => core::CV_16FC1,
            (tch::Kind::Half, 2) => core::CV_16FC2,
            (tch::Kind::Half, 3) => core::CV_16FC3,
            (tch::Kind::Half, 4) => core::CV_16FC4,
            (tch::Kind::Int, 1) => core::CV_32SC1,
            (tch::Kind::Int, 2) => core::CV_32SC2,
            (tch::Kind::Int, 3) => core::CV_32SC3,
            (tch::Kind::Int, 4) => core::CV_32SC4,
            (tch::Kind::Float, 1) => core::CV_32FC1,
            (tch::Kind::Float, 2) => core::CV_32FC2,
            (tch::Kind::Float, 3) => core::CV_32FC3,
            (tch::Kind::Float, 4) => core::CV_32FC4,
            (tch::Kind::Double, 1) => core::CV_64FC1,
            (tch::Kind::Double, 2) => core::CV_64FC2,
            (tch::Kind::Double, 3) => core::CV_64FC3,
            (tch::Kind::Double, 4) => core::CV_64FC4,
            (kind, channels) => bail!(
                "unsupported tensor kind {:?} and channels {}",
                kind,
                channels
            ),
        };

        let mat = unsafe {
            core::Mat::new_rows_cols_with_data(
                rows as i32,
                cols as i32,
                typ,
                tensor.data_ptr(),
                (cols as usize * kind.elt_size_in_bytes()) as platform_types::size_t,
            )?
        };

        Ok(mat)
    }
}

impl<T> TryFromCv<TensorAsImage<T>> for core::Mat
where
    T: Borrow<tch::Tensor>,
{
    type Error = Error;

    fn try_from_cv(from: TensorAsImage<T>) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

impl TryFromCv<&tch::Tensor> for core::Mat {
    type Error = Error;

    fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
        let tensor = from.f_contiguous()?.f_to_device(tch::Device::Cpu)?;
        let size: Vec<_> = tensor.size().into_iter().map(|dim| dim as i32).collect();
        let typ = match tensor.f_kind()? {
            tch::Kind::Uint8 => core::CV_8U,
            tch::Kind::Int8 => core::CV_8S,
            tch::Kind::Int16 => core::CV_16S,
            tch::Kind::Half => core::CV_16F,
            tch::Kind::Int => core::CV_32S,
            tch::Kind::Float => core::CV_32F,
            tch::Kind::Double => core::CV_64F,
            kind => bail!("unsupported tensor kind {:?}", kind),
        };

        let mat = unsafe { core::Mat::new_nd_with_data(&size, typ, tensor.data_ptr(), 0)? };
        Ok(mat)
    }
}

impl TryFromCv<tch::Tensor> for core::Mat {
    type Error = Error;

    fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
        (&from).try_into_cv()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_ref() -> Result<()> {
        Ok(())
    }
}
