use crate::{common::*, TryFromCv};
use ndarray as nd;

use to_ndarray_shape::*;

mod to_ndarray_shape {
    use super::*;

    pub trait ToNdArrayShape<D>
    where
        Self::Output: Sized + Into<nd::StrideShape<D>>,
    {
        type Output;
        type Error;

        fn to_ndarray_shape(&self) -> Result<Self::Output, Self::Error>;
    }

    impl ToNdArrayShape<nd::IxDyn> for Vec<i64> {
        type Output = Vec<usize>;
        type Error = Error;

        fn to_ndarray_shape(&self) -> Result<Self::Output, Self::Error> {
            let size: Vec<_> = self.iter().map(|&dim| dim as usize).collect();
            Ok(size)
        }
    }

    impl ToNdArrayShape<nd::Ix0> for Vec<i64> {
        type Output = [usize; 0];
        type Error = Error;

        fn to_ndarray_shape(&self) -> Result<Self::Output, Self::Error> {
            ensure!(
                self.len() == 0,
                "empty empty tensor dimension, but get {:?}",
                self
            );
            Ok([])
        }
    }

    impl ToNdArrayShape<nd::Ix1> for Vec<i64> {
        type Output = [usize; 1];
        type Error = Error;

        fn to_ndarray_shape(&self) -> Result<Self::Output, Self::Error> {
            let shape = match self.as_slice() {
                &[s0] => [s0 as usize],
                other => bail!("expect one dimension, but get {:?}", other),
            };
            Ok(shape)
        }
    }

    impl ToNdArrayShape<nd::Ix2> for Vec<i64> {
        type Output = [usize; 2];
        type Error = Error;

        fn to_ndarray_shape(&self) -> Result<Self::Output, Self::Error> {
            let shape = match self.as_slice() {
                &[s0, s1] => [s0 as usize, s1 as usize],
                other => bail!("expect one dimension, but get {:?}", other),
            };
            Ok(shape)
        }
    }

    impl ToNdArrayShape<nd::Ix3> for Vec<i64> {
        type Output = [usize; 3];
        type Error = Error;

        fn to_ndarray_shape(&self) -> Result<Self::Output, Self::Error> {
            let shape = match self.as_slice() {
                &[s0, s1, s2] => [s0 as usize, s1 as usize, s2 as usize],
                other => bail!("expect one dimension, but get {:?}", other),
            };
            Ok(shape)
        }
    }

    impl ToNdArrayShape<nd::Ix4> for Vec<i64> {
        type Output = [usize; 4];
        type Error = Error;

        fn to_ndarray_shape(&self) -> Result<Self::Output, Self::Error> {
            let shape = match self.as_slice() {
                &[s0, s1, s2, s3] => [s0 as usize, s1 as usize, s2 as usize, s3 as usize],
                other => bail!("expect one dimension, but get {:?}", other),
            };
            Ok(shape)
        }
    }

    impl ToNdArrayShape<nd::Ix5> for Vec<i64> {
        type Output = [usize; 5];
        type Error = Error;

        fn to_ndarray_shape(&self) -> Result<Self::Output, Self::Error> {
            let shape = match self.as_slice() {
                &[s0, s1, s2, s3, s4] => [
                    s0 as usize,
                    s1 as usize,
                    s2 as usize,
                    s3 as usize,
                    s4 as usize,
                ],
                other => bail!("expect one dimension, but get {:?}", other),
            };
            Ok(shape)
        }
    }

    impl ToNdArrayShape<nd::Ix6> for Vec<i64> {
        type Output = [usize; 6];
        type Error = Error;

        fn to_ndarray_shape(&self) -> Result<Self::Output, Self::Error> {
            let shape = match self.as_slice() {
                &[s0, s1, s2, s3, s4, s5] => [
                    s0 as usize,
                    s1 as usize,
                    s2 as usize,
                    s3 as usize,
                    s4 as usize,
                    s5 as usize,
                ],
                other => bail!("expect one dimension, but get {:?}", other),
            };
            Ok(shape)
        }
    }
}

impl<A, D> TryFromCv<tch::Tensor> for nd::Array<A, D>
where
    D: nd::Dimension,
    A: tch::kind::Element,
    Vec<A>: From<tch::Tensor>,
    Vec<i64>: ToNdArrayShape<D, Error = Error>,
{
    type Error = Error;

    fn try_from_cv(from: tch::Tensor) -> Result<Self, Self::Error> {
        // check element type consistency
        ensure!(
            from.kind() == A::KIND,
            "tensor with kind {:?} cannot converted to array with type {:?}",
            from.kind(),
            A::KIND
        );

        let shape = from.size();
        let elems = Vec::<A>::from(from);
        let array_shape = shape.to_ndarray_shape()?;
        let array = Self::from_shape_vec(array_shape, elems)?;
        Ok(array)
    }
}

impl<A, D> TryFromCv<&tch::Tensor> for nd::Array<A, D>
where
    D: nd::Dimension,
    A: tch::kind::Element,
    Vec<A>: From<tch::Tensor>,
    Vec<i64>: ToNdArrayShape<D, Error = Error>,
{
    type Error = Error;

    fn try_from_cv(from: &tch::Tensor) -> Result<Self, Self::Error> {
        Self::try_from_cv(from.shallow_clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TryIntoCv;
    use tch::IndexOp;

    #[test]
    fn tch_ndarray_conv() -> Result<()> {
        // ArrayD
        {
            let s0 = 3;
            let s1 = 4;
            let s2 = 5;

            let tensor = tch::Tensor::randn(&[s0, s1, s2], tch::kind::FLOAT_CPU);
            let array: nd::ArrayD<f32> = (&tensor).try_into_cv()?;

            let is_correct = itertools::iproduct!(0..s0, 0..s1, 0..s2).all(|(i0, i1, i2)| {
                let lhs = f32::from(tensor.i((i0, i1, i2)));
                let rhs = array[[i0 as usize, i1 as usize, i2 as usize]];
                lhs == rhs
            });

            ensure!(is_correct, "value mismatch");
        }

        // Array0
        {
            let tensor = tch::Tensor::randn(&[], tch::kind::FLOAT_CPU);
            let array: nd::Array0<f32> = (&tensor).try_into_cv()?;
            let lhs = f32::from(tensor);
            let rhs = array[()];
            ensure!(lhs == rhs, "value mismatch");
        }

        // Array1
        {
            let s0 = 10;
            let tensor = tch::Tensor::randn(&[s0], tch::kind::FLOAT_CPU);
            let array: nd::Array1<f32> = (&tensor).try_into_cv()?;

            let is_correct = (0..s0).all(|ind| {
                let lhs = f32::from(tensor.i((ind,)));
                let rhs = array[ind as usize];
                lhs == rhs
            });

            ensure!(is_correct, "value mismatch");
        }

        // Array2
        {
            let s0 = 3;
            let s1 = 5;

            let tensor = tch::Tensor::randn(&[s0, s1], tch::kind::FLOAT_CPU);
            let array: nd::Array2<f32> = (&tensor).try_into_cv()?;

            let is_correct = itertools::iproduct!(0..s0, 0..s1).all(|(i0, i1)| {
                let lhs = f32::from(tensor.i((i0, i1)));
                let rhs = array[[i0 as usize, i1 as usize]];
                lhs == rhs
            });

            ensure!(is_correct, "value mismatch");
        }

        // Array3
        {
            let s0 = 3;
            let s1 = 5;
            let s2 = 7;

            let tensor = tch::Tensor::randn(&[s0, s1, s2], tch::kind::FLOAT_CPU);
            let array: nd::Array3<f32> = (&tensor).try_into_cv()?;

            let is_correct = itertools::iproduct!(0..s0, 0..s1, 0..s2).all(|(i0, i1, i2)| {
                let lhs = f32::from(tensor.i((i0, i1, i2)));
                let rhs = array[[i0 as usize, i1 as usize, i2 as usize]];
                lhs == rhs
            });

            ensure!(is_correct, "value mismatch");
        }

        // Array4
        {
            let s0 = 3;
            let s1 = 5;
            let s2 = 7;
            let s3 = 11;

            let tensor = tch::Tensor::randn(&[s0, s1, s2, s3], tch::kind::FLOAT_CPU);
            let array: nd::Array4<f32> = (&tensor).try_into_cv()?;

            let is_correct =
                itertools::iproduct!(0..s0, 0..s1, 0..s2, 0..s3).all(|(i0, i1, i2, i3)| {
                    let lhs = f32::from(tensor.i((i0, i1, i2, i3)));
                    let rhs = array[[i0 as usize, i1 as usize, i2 as usize, i3 as usize]];
                    lhs == rhs
                });

            ensure!(is_correct, "value mismatch");
        }

        // Array5
        {
            let s0 = 3;
            let s1 = 5;
            let s2 = 7;
            let s3 = 11;
            let s4 = 13;

            let tensor = tch::Tensor::randn(&[s0, s1, s2, s3, s4], tch::kind::FLOAT_CPU);
            let array: nd::Array5<f32> = (&tensor).try_into_cv()?;

            let is_correct = itertools::iproduct!(0..s0, 0..s1, 0..s2, 0..s3, 0..s4).all(
                |(i0, i1, i2, i3, i4)| {
                    let lhs = f32::from(tensor.i((i0, i1, i2, i3, i4)));
                    let rhs = array[[
                        i0 as usize,
                        i1 as usize,
                        i2 as usize,
                        i3 as usize,
                        i4 as usize,
                    ]];
                    lhs == rhs
                },
            );

            ensure!(is_correct, "value mismatch");
        }

        // Array4
        {
            let s0 = 3;
            let s1 = 5;
            let s2 = 7;
            let s3 = 11;
            let s4 = 13;
            let s5 = 17;

            let tensor = tch::Tensor::randn(&[s0, s1, s2, s3, s4, s5], tch::kind::FLOAT_CPU);
            let array: nd::Array6<f32> = (&tensor).try_into_cv()?;

            let is_correct = itertools::iproduct!(0..s0, 0..s1, 0..s2, 0..s3, 0..s4, 0..s5).all(
                |(i0, i1, i2, i3, i4, i5)| {
                    let lhs = f32::from(tensor.i((i0, i1, i2, i3, i4, i5)));
                    let rhs = array[[
                        i0 as usize,
                        i1 as usize,
                        i2 as usize,
                        i3 as usize,
                        i4 as usize,
                        i5 as usize,
                    ]];
                    lhs == rhs
                },
            );

            ensure!(is_correct, "value mismatch");
        }

        Ok(())
    }
}
