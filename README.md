# stensor
A library that provides traits that relate to tensors, and the generic type `DTensor<T>`.

## Example
```rust
use crate::prelude::*;

// Create a 4 by 4 matrix
let a = (1..17).into_iter().shape(vec![4, 4]);

// Transpose the matrix
let c = (&a).transpose();

let b = [
    1, 5 , 9 , 13,
    2, 6 , 10, 14,
    3, 7 , 11, 15,
    4, 8 , 12, 16
].into_iter().shape(vec![4, 4]);

assert_eq!(b, c);

// You can also add tensors...
let c = a + c;

let d = [
    2 , 7 , 12, 17,
    7 , 12, 17, 22,
    12, 17, 22, 27,
    17, 22, 27, 32 
].into_iter().shape(vec![4, 4]);

assert_eq!(c, d);

// You can also do kronecker products...

// Create 2 by 2 matrix
let mut b = (1..5).into_iter().collect::<DTensor<i32>>().shape(vec![2, 2]);

// Do kronecker product on self
b = b.clone().kron(b);

let c = [
    1, 2 , 2 , 4,
    3, 4 , 6 , 8,
    3, 6 , 4 , 8,
    9, 12, 12, 16
].into_iter().shape(vec![4, 4]);

assert_eq!(b, c);

// Dot products...

let c = vec![
    1, 2, 3, 4, 5, 6
].to_tensor();

let d = vec![
    2, 8, 12, 13, 100, 36
].to_tensor();

assert_eq!(c * d, DTensor::scalar(822))
```
