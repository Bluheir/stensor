#![feature(min_specialization)]
#![feature(negative_impls)]
#![feature(auto_traits)]
#![feature(allocator_api)]
#![feature(ptr_internals)]
#![feature(box_syntax)]
#![feature(specialization)]

pub mod tensor;
mod stridespair;
mod helpers;
mod ops;

#[cfg(test)]
mod tests {
    use std::ops::Deref;

    use crate::{
        tensor::{calculate_strides, indices_to_ind, num_to_indices, DTensor, DimsHelper},
    };
    use crate::ops::*;

    #[test]
    fn test2() {
        let b = (1..7).into_iter().collect::<Vec<i32>>();
        let tensor = DTensor::new(b.clone(), vec![2, 3]);

        let mut tensor2 = tensor.clone();
        tensor2.reshape(vec![6]);

        let mut t = tensor.clone().kron(tensor2.clone());
        t.fix();

        let c = (1..5).into_iter().collect::<Vec<i32>>();
        let tensor = DTensor::new(c.clone(), vec![2, 2]);

        println!("{:?}", tensor.clone().kron(tensor));


    }
}

pub trait ZeroTensor {
    fn zero_tensor(dims : &[usize]) -> Self;
}