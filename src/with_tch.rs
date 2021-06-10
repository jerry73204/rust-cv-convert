use crate::{common::*, FromCv, TryFromCv};
use slice_of_array::prelude::*;

macro_rules! impl_from_array {
    ($elem:ty) => {
        // borrowed tensor to array reference

        impl<'a, const N: usize> TryFromCv<&'a tch::Tensor> for &'a [$elem; N] {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.device() == tch::Device::Cpu);
                ensure!(from.size() == &[N as i64]);
                let slice: &[$elem] =
                    unsafe { slice::from_raw_parts(from.data_ptr() as *mut $elem, N) };
                Ok(slice.as_array())
            }
        }

        impl<'a, const N1: usize, const N2: usize> TryFromCv<&'a tch::Tensor>
            for &'a [[$elem; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.device() == tch::Device::Cpu);
                ensure!(from.size() == &[N1 as i64, N2 as i64]);
                let slice: &[$elem] =
                    unsafe { slice::from_raw_parts(from.data_ptr() as *mut $elem, N1 * N2) };
                Ok(slice.nest().as_array())
            }
        }

        impl<'a, const N1: usize, const N2: usize, const N3: usize> TryFromCv<&'a tch::Tensor>
            for &'a [[[$elem; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.device() == tch::Device::Cpu);
                ensure!(from.size() == &[N1 as i64, N2 as i64, N3 as i64]);
                let slice: &[$elem] =
                    unsafe { slice::from_raw_parts(from.data_ptr() as *mut $elem, N1 * N2 * N3) };
                Ok(slice.nest().nest().as_array())
            }
        }

        impl<'a, const N1: usize, const N2: usize, const N3: usize, const N4: usize>
            TryFromCv<&'a tch::Tensor> for &'a [[[[$elem; N4]; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.device() == tch::Device::Cpu);
                ensure!(from.size() == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64]);
                let slice: &[$elem] = unsafe {
                    slice::from_raw_parts(from.data_ptr() as *mut $elem, N1 * N2 * N3 * N4)
                };
                Ok(slice.nest().nest().nest().as_array())
            }
        }

        impl<
                'a,
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
            > TryFromCv<&'a tch::Tensor> for &'a [[[[[$elem; N5]; N4]; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.device() == tch::Device::Cpu);
                ensure!(from.size() == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64]);
                let slice: &[$elem] = unsafe {
                    slice::from_raw_parts(from.data_ptr() as *mut $elem, N1 * N2 * N3 * N4 * N5)
                };
                Ok(slice.nest().nest().nest().nest().as_array())
            }
        }

        impl<
                'a,
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
                const N6: usize,
            > TryFromCv<&'a tch::Tensor> for &'a [[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.device() == tch::Device::Cpu);
                ensure!(
                    from.size()
                        == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64, N6 as i64]
                );
                let slice: &[$elem] = unsafe {
                    slice::from_raw_parts(
                        from.data_ptr() as *mut $elem,
                        N1 * N2 * N3 * N4 * N5 * N6,
                    )
                };
                Ok(slice.nest().nest().nest().nest().nest().as_array())
            }
        }

        // borrowed tensor to array

        impl<const N: usize> TryFromCv<&tch::Tensor> for [$elem; N] {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.size() == &[N as i64]);
                let mut array = [Default::default(); N];
                from.f_copy_data(array.as_mut(), N)?;
                Ok(array)
            }
        }

        impl<const N1: usize, const N2: usize> TryFromCv<&tch::Tensor> for [[$elem; N2]; N1] {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.size() == &[N1 as i64, N2 as i64]);
                let mut array = [[Default::default(); N2]; N1];
                from.f_copy_data(array.flat_mut(), N1 * N2)?;
                Ok(array)
            }
        }

        impl<const N1: usize, const N2: usize, const N3: usize> TryFromCv<&tch::Tensor>
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

        impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize>
            TryFromCv<&tch::Tensor> for [[[[$elem; N4]; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
                ensure!(from.size() == &[N1 as i64, N2 as i64, N3 as i64, N4 as i64]);
                let mut array = [[[[Default::default(); N4]; N3]; N2]; N1];
                from.f_copy_data(array.flat_mut().flat_mut().flat_mut(), N1 * N2 * N3 * N4)?;
                Ok(array)
            }
        }

        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
            > TryFromCv<&tch::Tensor> for [[[[[$elem; N5]; N4]; N3]; N2]; N1]
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

        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
                const N6: usize,
            > TryFromCv<&tch::Tensor> for [[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]
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

        // owned tensor to array

        impl<const N: usize> TryFromCv<tch::Tensor> for [$elem; N] {
            type Error = Error;

            fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
                Self::try_from_cv(&from)
            }
        }

        impl<const N1: usize, const N2: usize> TryFromCv<tch::Tensor> for [[$elem; N2]; N1] {
            type Error = Error;

            fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
                Self::try_from_cv(&from)
            }
        }

        impl<const N1: usize, const N2: usize, const N3: usize> TryFromCv<tch::Tensor>
            for [[[$elem; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
                Self::try_from_cv(&from)
            }
        }

        impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize>
            TryFromCv<tch::Tensor> for [[[[$elem; N4]; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
                Self::try_from_cv(&from)
            }
        }

        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
            > TryFromCv<tch::Tensor> for [[[[[$elem; N5]; N4]; N3]; N2]; N1]
        {
            type Error = Error;

            fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
                Self::try_from_cv(&from)
            }
        }

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

            fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
                Self::try_from_cv(&from)
            }
        }

        // borrowed array to tensor

        impl<const N: usize> FromCv<&[$elem; N]> for tch::Tensor {
            fn from_cv(from: &[$elem; N]) -> Self {
                Self::of_slice(from.as_ref())
            }
        }

        impl<const N1: usize, const N2: usize> FromCv<&[[$elem; N2]; N1]> for tch::Tensor {
            fn from_cv(from: &[[$elem; N2]; N1]) -> Self {
                Self::of_slice(from.flat()).view([N1 as i64, N2 as i64])
            }
        }

        impl<const N1: usize, const N2: usize, const N3: usize> FromCv<&[[[$elem; N3]; N2]; N1]>
            for tch::Tensor
        {
            fn from_cv(from: &[[[$elem; N3]; N2]; N1]) -> Self {
                Self::of_slice(from.flat().flat()).view([N1 as i64, N2 as i64, N3 as i64])
            }
        }

        impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize>
            FromCv<&[[[[$elem; N4]; N3]; N2]; N1]> for tch::Tensor
        {
            fn from_cv(from: &[[[[$elem; N4]; N3]; N2]; N1]) -> Self {
                Self::of_slice(from.flat().flat().flat())
                    .view([N1 as i64, N2 as i64, N3 as i64, N4 as i64])
            }
        }

        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
            > FromCv<&[[[[[$elem; N5]; N4]; N3]; N2]; N1]> for tch::Tensor
        {
            fn from_cv(from: &[[[[[$elem; N5]; N4]; N3]; N2]; N1]) -> Self {
                Self::of_slice(from.flat().flat().flat().flat())
                    .view([N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64])
            }
        }

        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
                const N6: usize,
            > FromCv<&[[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]> for tch::Tensor
        {
            fn from_cv(from: &[[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]) -> Self {
                Self::of_slice(from.flat().flat().flat().flat().flat()).view([
                    N1 as i64, N2 as i64, N3 as i64, N4 as i64, N5 as i64, N6 as i64,
                ])
            }
        }

        // owned array to tensor

        impl<const N: usize> FromCv<[$elem; N]> for tch::Tensor {
            fn from_cv(from: [$elem; N]) -> Self {
                Self::from_cv(&from)
            }
        }

        impl<const N1: usize, const N2: usize> FromCv<[[$elem; N2]; N1]> for tch::Tensor {
            fn from_cv(from: [[$elem; N2]; N1]) -> Self {
                Self::from_cv(&from)
            }
        }

        impl<const N1: usize, const N2: usize, const N3: usize> FromCv<[[[$elem; N3]; N2]; N1]>
            for tch::Tensor
        {
            fn from_cv(from: [[[$elem; N3]; N2]; N1]) -> Self {
                Self::from_cv(&from)
            }
        }

        impl<const N1: usize, const N2: usize, const N3: usize, const N4: usize>
            FromCv<[[[[$elem; N4]; N3]; N2]; N1]> for tch::Tensor
        {
            fn from_cv(from: [[[[$elem; N4]; N3]; N2]; N1]) -> Self {
                Self::from_cv(&from)
            }
        }

        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
            > FromCv<[[[[[$elem; N5]; N4]; N3]; N2]; N1]> for tch::Tensor
        {
            fn from_cv(from: [[[[[$elem; N5]; N4]; N3]; N2]; N1]) -> Self {
                Self::from_cv(&from)
            }
        }

        impl<
                const N1: usize,
                const N2: usize,
                const N3: usize,
                const N4: usize,
                const N5: usize,
                const N6: usize,
            > FromCv<[[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]> for tch::Tensor
        {
            fn from_cv(from: [[[[[[$elem; N6]; N5]; N4]; N3]; N2]; N1]) -> Self {
                Self::from_cv(&from)
            }
        }
    };
}

impl_from_array!(u8);
impl_from_array!(i8);
impl_from_array!(i16);
impl_from_array!(i32);
impl_from_array!(i64);
impl_from_array!(half::f16);
impl_from_array!(f32);
impl_from_array!(f64);
impl_from_array!(bool);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TryIntoCv;
    use rand::prelude::*;

    #[test]
    fn tensor_to_array_ref() {
        let mut rng = rand::thread_rng();

        // 1 dim
        {
            type T = [f32; 3];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_into_cv().unwrap();
            assert!(array == input);

            let array_ref: &T = (&tensor).try_into_cv().unwrap();
            assert!(array_ref == input.as_ref());
        }

        // 2 dim
        {
            type T = [[f32; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_into_cv().unwrap();
            assert!(array == input);

            let array_ref: &T = (&tensor).try_into_cv().unwrap();
            assert!(array_ref == input.as_ref());
        }

        // 3 dim
        {
            type T = [[[f32; 4]; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_into_cv().unwrap();
            assert!(array == input);

            let array_ref: &T = (&tensor).try_into_cv().unwrap();
            assert!(array_ref == input.as_ref());
        }

        // 4 dim
        {
            type T = [[[[f32; 2]; 4]; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_into_cv().unwrap();
            assert!(array == input);

            let array_ref: &T = (&tensor).try_into_cv().unwrap();
            assert!(array_ref == input.as_ref());
        }

        // 4 dim
        {
            type T = [[[[[f32; 3]; 2]; 4]; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_into_cv().unwrap();
            assert!(array == input);

            let array_ref: &T = (&tensor).try_into_cv().unwrap();
            assert!(array_ref == input.as_ref());
        }

        // 5 dim
        {
            type T = [[[[[[f32; 2]; 3]; 2]; 4]; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_into_cv().unwrap();
            assert!(array == input);

            let array_ref: &T = (&tensor).try_into_cv().unwrap();
            assert!(array_ref == input.as_ref());
        }

        // 6 dim
        {
            type T = [[[[[[f32; 2]; 3]; 2]; 4]; 3]; 2];

            let input: T = rng.gen();
            let tensor = tch::Tensor::from_cv(&input);

            let array: T = (&tensor).try_into_cv().unwrap();
            assert!(array == input);

            let array_ref: &T = (&tensor).try_into_cv().unwrap();
            assert!(array_ref == input.as_ref());
        }
    }
}
