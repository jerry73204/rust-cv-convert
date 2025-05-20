use crate::{FromCv, TryAsRefCv, TryFromCv};
use anyhow::{ensure, Error, Result};
use slice_of_array::prelude::*;
use std::{mem::ManuallyDrop, ops::Deref, slice};

// Helper macros for implementing conversions between tensors and different dimensioned arrays
macro_rules! impl_from_array {
    ($elem:ty, 1) => {
        // Borrowed tensor to borrowed array
        impl<'a, const N: usize> TryAsRefCv<'a, TensorAsArray<'a, [$elem; N]>> for tch::Tensor {
            type Error = Error;

            fn try_as_ref_cv(&'a self) -> Result<TensorAsArray<'a, [$elem; N]>, Self::Error> {
                ensure!(self.device() == tch::Device::Cpu);
                ensure!(self.kind() == <$elem as tch::kind::Element>::KIND);
                ensure!(self.size() == &[N as i64]);

                let slice: &[$elem] =
                    unsafe { slice::from_raw_parts(self.data_ptr() as *mut $elem, N) };
                #[allow(unstable_name_collisions)]
                let array = slice.as_array();

                Ok(TensorAsArray {
                    data: ManuallyDrop::new(*array),
                    _tensor: self,
                })
            }
        }

        // Borrowed tensor to owned array
        impl<const N: usize> TryFromCv<tch::Tensor> for [$elem; N] {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.size() == &[N as i64]);
                let mut array = [Default::default(); N];
                from.f_copy_data(array.as_mut(), N)?;
                Ok(array)
            }
        }

        // Borrowed array to tensor
        impl<const N: usize> FromCv<[$elem; N]> for tch::Tensor {
            fn from_cv(from: &[$elem; N]) -> Self {
                Self::from_slice(from.as_ref())
            }
        }
    };

    ($elem:ty, 2) => {
        // Borrowed tensor to borrowed array
        impl<'a, const N1: usize, const N2: usize> TryAsRefCv<'a, TensorAsArray<'a, [[$elem; N2]; N1]>>
            for tch::Tensor
        {
            type Error = Error;

            fn try_as_ref_cv(&'a self) -> Result<TensorAsArray<'a, [[$elem; N2]; N1]>, Self::Error> {
                ensure!(self.device() == tch::Device::Cpu);
                ensure!(self.kind() == <$elem as tch::kind::Element>::KIND);
                ensure!(self.size() == &[N1 as i64, N2 as i64]);

                let slice: &[$elem] =
                    unsafe { slice::from_raw_parts(self.data_ptr() as *mut $elem, N1 * N2) };
                #[allow(unstable_name_collisions)]
                let array = slice.nest().as_array();

                Ok(TensorAsArray {
                    data: ManuallyDrop::new(*array),
                    _tensor: self,
                })
            }
        }

        // Borrowed tensor to owned array
        impl<const N1: usize, const N2: usize> TryFromCv<tch::Tensor> for [[$elem; N2]; N1] {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.size() == &[N1 as i64, N2 as i64]);
                let mut array = [[Default::default(); N2]; N1];
                from.f_copy_data(array.flat_mut(), N1 * N2)?;
                Ok(array)
            }
        }

        // Borrowed array to tensor
        impl<const N1: usize, const N2: usize> FromCv<[[$elem; N2]; N1]> for tch::Tensor {
            fn from_cv(from: &[[$elem; N2]; N1]) -> Self {
                Self::from_slice(from.flat()).view([N1 as i64, N2 as i64])
            }
        }
    };

    ($elem:ty, 3) => {
        // Borrowed tensor to borrowed array
        impl<'a, const N1: usize, const N2: usize, const N3: usize>
            TryAsRefCv<'a, TensorAsArray<'a, [[[$elem; N3]; N2]; N1]>> for tch::Tensor
        {
            type Error = Error;

            fn try_as_ref_cv(&'a self) -> Result<TensorAsArray<'a, [[[$elem; N3]; N2]; N1]>, Self::Error> {
                ensure!(self.device() == tch::Device::Cpu);
                ensure!(self.kind() == <$elem as tch::kind::Element>::KIND);
                ensure!(self.size() == &[N1 as i64, N2 as i64, N3 as i64]);

                let slice: &[$elem] =
                    unsafe { slice::from_raw_parts(self.data_ptr() as *mut $elem, N1 * N2 * N3) };
                #[allow(unstable_name_collisions)]
                let array = slice.nest().nest().as_array();

                Ok(TensorAsArray {
                    data: ManuallyDrop::new(*array),
                    _tensor: self,
                })
            }
        }

        // Borrowed tensor to owned array
        impl<const N1: usize, const N2: usize, const N3: usize> TryFromCv<tch::Tensor>
            for [[[$elem; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.size() == &[N1 as i64, N2 as i64, N3 as i64]);
                let mut array = [[[Default::default(); N3]; N2]; N1];
                from.f_copy_data(array.flat_mut().flat_mut(), N1 * N2 * N3)?;
                Ok(array)
            }
        }

        // Borrowed array to tensor
        impl<const N1: usize, const N2: usize, const N3: usize> FromCv<[[[$elem; N3]; N2]; N1]>
            for tch::Tensor
        {
            fn from_cv(from: &[[[$elem; N3]; N2]; N1]) -> Self {
                Self::from_slice(from.flat().flat()).view([N1 as i64, N2 as i64, N3 as i64])
            }
        }
    };

    ($elem:ty, 4) => {
        // Borrowed tensor to borrowed array
        impl<'a, const N1: usize, const N2: usize, const N3: usize, const N4: usize>
            TryAsRefCv<'a, TensorAsArray<'a, [[[[$elem; N4]; N3]; N2]; N1]>> for tch::Tensor
        {
            type Error = Error;

            fn try_as_ref_cv(&'a self) -> Result<TensorAsArray<'a, [[[[$elem; N4]; N3]; N2]; N1]>, Self::Error> {
                ensure!(self.device() == tch::Device::Cpu);
                ensure!(self.kind() == <$elem as tch::kind::Element>::KIND);
                ensure!(self.size() == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64]);

                let slice: &[$elem] = unsafe {
                    slice::from_raw_parts(self.data_ptr() as *mut $elem, N1 * N2 * N3 * N4)
                };
                #[allow(unstable_name_collisions)]
                let array = slice.nest().nest().nest().as_array();

                Ok(TensorAsArray {
                    data: ManuallyDrop::new(*array),
                    _tensor: self,
                })
            }
        }

        // Borrowed tensor to owned array
        impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize>
            TryFromCv<tch::Tensor> for [[[[$elem; N4]; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.size() == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64]);
                let mut array = [[[[Default::default(); N4]; N3]; N2]; N1];
                from.f_copy_data(array.flat_mut().flat_mut().flat_mut(), N1 * N2 * N3 * N4)?;
                Ok(array)
            }
        }

        // Borrowed array to tensor
        impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize>
            FromCv<[[[[$elem; N4]; N3]; N2]; N1]> for tch::Tensor
        {
            fn from_cv(from: &[[[[$elem; N4]; N3]; N2]; N1]) -> Self {
                Self::from_slice(from.flat().flat().flat())
                    .view([N1 as i64, N2 as i64, N3 as i64, N4 as i64])
            }
        }
    };

    ($elem:ty, 5) => {
        // Borrowed tensor to borrowed array
        impl<
                'a,
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
            > TryAsRefCv<'a, TensorAsArray<'a, [[[[[$elem; N5]; N4]; N3]; N2]; N1]>> for tch::Tensor
        {
            type Error = Error;

            fn try_as_ref_cv(&'a self) -> Result<TensorAsArray<'a, [[[[[$elem; N5]; N4]; N3]; N2]; N1]>, Self::Error> {
                ensure!(self.device() == tch::Device::Cpu);
                ensure!(self.kind() == <$elem as tch::kind::Element>::KIND);
                ensure!(self.size() == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64]);

                let slice: &[$elem] = unsafe {
                    slice::from_raw_parts(self.data_ptr() as *mut $elem, N1 * N2 * N3 * N4 * N5)
                };
                #[allow(unstable_name_collisions)]
                let array = slice.nest().nest().nest().nest().as_array();

                Ok(TensorAsArray {
                    data: ManuallyDrop::new(*array),
                    _tensor: self,
                })
            }
        }

        // Borrowed tensor to owned array
        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
            > TryFromCv<tch::Tensor> for [[[[[$elem; N5]; N4]; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.size() == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64]);
                let mut array = [[[[[Default::default(); N5]; N4]; N3]; N2]; N1];
                from.f_copy_data(
                    array.flat_mut().flat_mut().flat_mut().flat_mut(),
                    N1 * N2 * N3 * N4 * N5,
                )?;
                Ok(array)
            }
        }

        // Borrowed array to tensor
        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
            > FromCv<[[[[[$elem; N5]; N4]; N3]; N2]; N1]> for tch::Tensor
        {
            fn from_cv(from: &[[[[[$elem; N5]; N4]; N3]; N2]; N1]) -> Self {
                Self::from_slice(from.flat().flat().flat().flat())
                    .view([N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64])
            }
        }
    };

    ($elem:ty, 6) => {
        // Borrowed tensor to borrowed array
        impl<
                'a,
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
                const N6: usize,
            > TryAsRefCv<'a, TensorAsArray<'a, [[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]>> for tch::Tensor
        {
            type Error = Error;

            fn try_as_ref_cv(&'a self) -> Result<TensorAsArray<'a, [[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]>, Self::Error> {
                ensure!(self.device() == tch::Device::Cpu);
                ensure!(self.kind() == <$elem as tch::kind::Element>::KIND);
                ensure!(
                    self.size()
                        == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64, N6 as i64]
                );

                let slice: &[$elem] = unsafe {
                    slice::from_raw_parts(
                        self.data_ptr() as *mut $elem,
                        N1 * N2 * N3 * N4 * N5 * N6,
                    )
                };
                #[allow(unstable_name_collisions)]
                let array = slice.nest().nest().nest().nest().nest().as_array();

                Ok(TensorAsArray {
                    data: ManuallyDrop::new(*array),
                    _tensor: self,
                })
            }
        }

        // Borrowed tensor to owned array
        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
                const N6: usize,
            > TryFromCv<tch::Tensor> for [[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(
                    from.size()
                        == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64, N6 as i64]
                );
                let mut array = [[[[[[Default::default(); N6]; N5]; N4]; N3]; N2]; N1];
                from.f_copy_data(
                    array.flat_mut().flat_mut().flat_mut().flat_mut().flat_mut(),
                    N1 * N2 * N3 * N4 * N5 * N6,
                )?;
                Ok(array)
            }
        }

        // Borrowed array to tensor
        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
                const N6: usize,
            > FromCv<[[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]> for tch::Tensor
        {
            fn from_cv(from: &[[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]) -> Self {
                Self::from_slice(from.flat().flat().flat().flat().flat()).view([
                    N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64, N6 as i64,
                ])
            }
        }
    };
}

// Generate implementations for each element type and dimension
impl_from_array!(u8, 1);
impl_from_array!(u8, 2);
impl_from_array!(u8, 3);
impl_from_array!(u8, 4);
impl_from_array!(u8, 5);
impl_from_array!(u8, 6);

impl_from_array!(i8, 1);
impl_from_array!(i8, 2);
impl_from_array!(i8, 3);
impl_from_array!(i8, 4);
impl_from_array!(i8, 5);
impl_from_array!(i8, 6);

impl_from_array!(i16, 1);
impl_from_array!(i16, 2);
impl_from_array!(i16, 3);
impl_from_array!(i16, 4);
impl_from_array!(i16, 5);
impl_from_array!(i16, 6);

impl_from_array!(i32, 1);
impl_from_array!(i32, 2);
impl_from_array!(i32, 3);
impl_from_array!(i32, 4);
impl_from_array!(i32, 5);
impl_from_array!(i32, 6);

impl_from_array!(i64, 1);
impl_from_array!(i64, 2);
impl_from_array!(i64, 3);
impl_from_array!(i64, 4);
impl_from_array!(i64, 5);
impl_from_array!(i64, 6);

impl_from_array!(half::f16, 1);
impl_from_array!(half::f16, 2);
impl_from_array!(half::f16, 3);
impl_from_array!(half::f16, 4);
impl_from_array!(half::f16, 5);
impl_from_array!(half::f16, 6);

impl_from_array!(f32, 1);
impl_from_array!(f32, 2);
impl_from_array!(f32, 3);
impl_from_array!(f32, 4);
impl_from_array!(f32, 5);
impl_from_array!(f32, 6);

impl_from_array!(f64, 1);
impl_from_array!(f64, 2);
impl_from_array!(f64, 3);
impl_from_array!(f64, 4);
impl_from_array!(f64, 5);
impl_from_array!(f64, 6);

impl_from_array!(bool, 1);
impl_from_array!(bool, 2);
impl_from_array!(bool, 3);
impl_from_array!(bool, 4);
impl_from_array!(bool, 5);
impl_from_array!(bool, 6);

pub use tensors::*;
mod tensors {
    use super::*;

    /// A wrapper for a borrowed array reference from a tensor.
    #[derive(Debug)]
    pub struct TensorAsArray<'a, T> {
        pub(crate) data: ManuallyDrop<T>,
        pub(crate) _tensor: &'a tch::Tensor,
    }

    impl<'a, T> Drop for TensorAsArray<'a, T> {
        fn drop(&mut self) {
            unsafe {
                ManuallyDrop::drop(&mut self.data);
            }
        }
    }

    impl<'a, T> AsRef<T> for TensorAsArray<'a, T> {
        fn as_ref(&self) -> &T {
            &self.data
        }
    }

    impl<'a, T> Deref for TensorAsArray<'a, T> {
        type Target = T;

        fn deref(&self) -> &Self::Target {
            &self.data
        }
    }

    /// An 2D image [Tensor](tch::Tensor) with dimension order.
    #[derive(Debug)]
    pub struct TchTensorAsImage {
        pub(crate) tensor: tch::Tensor,
        pub(crate) kind: TchTensorImageShape,
    }

    /// Describes the image channel order of a [Tensor](tch::Tensor).
    #[derive(Debug, Clone, Copy)]
    pub enum TchTensorImageShape {
        Whc,
        Hwc,
        Chw,
        Cwh,
    }

    impl TchTensorAsImage {
        pub fn new(tensor: tch::Tensor, kind: TchTensorImageShape) -> Result<Self> {
            let ndim = tensor.dim();
            ensure!(
                ndim == 3,
                "the tensor must have 3 dimensions, but get {}",
                ndim
            );
            Ok(Self { tensor, kind })
        }

        pub fn into_inner(self) -> tch::Tensor {
            self.tensor
        }

        pub fn kind(&self) -> TchTensorImageShape {
            self.kind
        }

        pub fn try_into_cv<T>(&self) -> Result<T, T::Error>
        where
            T: TryFromCv<Self>,
        {
            T::try_from_cv(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TryAsRefCv, TryToCv};
    use rand::prelude::*;

    #[test]
    fn tensor_to_array_ref() {
        let mut rng = rand::thread_rng();

        // 1 dim
        {
            type T = [f32; 3];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_to_cv().unwrap();
            assert!(array == input);

            let array_wrapper: TensorAsArray<T> = (&tensor).try_as_ref_cv().unwrap();
            assert!(*array_wrapper == input);
        }

        // 2 dim
        {
            type T = [[f32; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_to_cv().unwrap();
            assert!(array == input);

            let array_wrapper: TensorAsArray<T> = (&tensor).try_as_ref_cv().unwrap();
            assert!(*array_wrapper == input);
        }

        // 3 dim
        {
            type T = [[[f32; 4]; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_to_cv().unwrap();
            assert!(array == input);

            let array_wrapper: TensorAsArray<T> = (&tensor).try_as_ref_cv().unwrap();
            assert!(*array_wrapper == input);
        }

        // 4 dim
        {
            type T = [[[[f32; 2]; 4]; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_to_cv().unwrap();
            assert!(array == input);

            let array_wrapper: TensorAsArray<T> = (&tensor).try_as_ref_cv().unwrap();
            assert!(*array_wrapper == input);
        }

        // 5 dim
        {
            type T = [[[[[f32; 3]; 2]; 4]; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_to_cv().unwrap();
            assert!(array == input);

            let array_wrapper: TensorAsArray<T> = (&tensor).try_as_ref_cv().unwrap();
            assert!(*array_wrapper == input);
        }

        // 6 dim
        {
            type T = [[[[[[f32; 2]; 3]; 2]; 4]; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_to_cv().unwrap();
            assert!(array == input);

            let array_wrapper: TensorAsArray<T> = (&tensor).try_as_ref_cv().unwrap();
            assert!(*array_wrapper == input);
        }
    }
}
