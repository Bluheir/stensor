use std::{
    alloc::Allocator,
    collections::HashSet,
    default,
    fmt::{Debug, Display, Formatter, Write},
    ops::{Add, AddAssign, Deref, DerefMut, Index, IndexMut, Mul, MulAssign, Sub, SubAssign}, iter::Take,
};

use num_traits::{Zero, One};

use crate::{ZeroTensor, helpers::{Chunks, ChunksExt, ToTensor, VecTensorHelpers, ToParallel}};
use crate::ops::*;

pub use crate::stridespair::*;

// Helper Methods
pub trait DimsHelper {
    fn mul_all(&self) -> usize;
    fn mul_from(&self, ind: usize) -> usize;
    fn to_vec(&self) -> Vec<usize>;
}
impl DimsHelper for [usize] {
    fn mul_all(&self) -> usize {
        self.iter().fold(1, |c, a| c * a)
    }
    fn mul_from(&self, ind: usize) -> usize {
        self[ind..].iter().fold(1, |c, a| c * a)
    }
    fn to_vec(&self) -> Vec<usize> {
        let mut ret = Vec::with_capacity(self.len());
        ret.extend_from_slice(self);
        ret
    }
}

pub fn indices_to_ind(strides: &[usize], indices: &[usize]) -> usize {
    strides
        .iter()
        .zip(indices.iter())
        .fold(0usize, |c, (a, b)| c + a * b)
}
pub fn indices_to_ind_rev(strides: &[usize], indices: &[usize]) -> usize {
    strides
        .iter()
        .enumerate()
        .map(|(i, elem)| (elem, indices[indices.len() - i - 1]))
        .fold(0usize, |c, (a, b)| c + a * b)
}
pub fn calculate_strides(dims: &[usize]) -> Vec<usize> {
    let mut ret = Vec::with_capacity(dims.len());
    let mut last = dims.mul_from(0);

    dims.iter().for_each(|i| {
        last /= i;
        ret.push(last);
    });

    ret
}
pub fn num_to_indices(mut num: usize, dims: &[usize], indices: &mut [usize]) {
    for i in 0..dims.len() {
        let index = dims.len() - i - 1;
        let d = dims[index];

        let modulo = num % d;
        indices[index] = modulo;

        num = (num - modulo) / d;
    }
}
pub fn num_to_indices_rev(mut num: usize, dims: &[usize], indices: &mut [usize]) {
    for i in 0..dims.len() {
        let index = dims.len() - i - 1;
        let d = dims[index];

        let modulo = num % d;
        indices[i] = modulo;

        num = (num - modulo) / d;
    }
}

#[derive(Debug, Clone)]
pub enum DynArray<T> {
    Item(T),
    Array(Vec<DynArray<T>>),
}

impl<T> DynArray<T> {
    pub fn from_strides(mut arr: Vec<T>, strides: &[usize]) -> Self {
        if strides.len() == 0 {
            return Self::Item(arr.remove(0));
        }

        let mul = strides.mul_from(0);

        Self::from_strides_arr_internal(
            arr.into_iter().map(|a| Self::Item(a)).collect(),
            strides,
            mul,
        )
    }
    fn from_strides_arr_internal(mut arr: Vec<DynArray<T>>, strides: &[usize], mul: usize) -> Self {
        if strides.len() == 1 {
            return Self::Array(arr);
        }

        let stride = strides[0];
        let mut ret = Vec::with_capacity(stride);
        let next_mul = mul / stride;

        for _ in 0..stride {
            let v = arr.drain(0..mul).collect();
            ret.push(Self::from_strides_arr_internal(v, &strides[1..], next_mul))
        }

        Self::Array(ret)
    }
}

impl<T> From<T> for DynArray<T> {
    fn from(value: T) -> Self {
        Self::Item(value)
    }
}

#[derive(Debug, Clone)]
pub struct DTensor<T> {
    strides: Vec<usize>,
    dims: Vec<usize>,
    contents: Vec<T>,
}

impl<T> DTensor<T> {
    pub fn new_unchecked(contents: Vec<T>, dims: Vec<usize>, strides: Vec<usize>) -> Self {
        Self {
            strides,
            dims,
            contents,
        }
    }
    pub fn new(contents: Vec<T>, dims: Vec<usize>) -> Self {
        let strides = calculate_strides(&dims);
        Self::new_unchecked(contents, dims, strides)
    }

    pub fn scalar(v: T) -> Self {
        Self::new_unchecked(vec![v], vec![], vec![])
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn reshape(&mut self, dims: Vec<usize>) {
        self.dims = dims;
        self.strides = calculate_strides(&self.dims);
    }

    pub fn pair_mut(&mut self) -> StridePairMut<'_, '_, T> {
        let s = unsafe { &*(self.strides() as *const [usize]) };
        let d = unsafe { &*(self.dims() as *const [usize]) };
        StridePairMut::new(self, s, d)
    }
    pub fn pair(&self) -> StridePair<'_, '_, T> {
        StridePair::new(&self, self.strides(), self.dims())
    }

    pub fn dyn_mut(&mut self) -> DynArray<&mut T> {
        let a = self.strides.clone();
        return DynArray::from_strides(self.iter_mut().collect(), &a);
    }

    pub fn get(&self, indices: &[usize]) -> &T {
        &self[indices_to_ind(self.strides(), indices)]
    }
    pub fn get_mut(&mut self, indices: &[usize]) -> &mut T {
        let index = indices_to_ind(&self.strides, indices);
        &mut self[index]
    }

    pub fn iter_mut(&mut self) -> DTensorIteratorMut<'_, T> {
        self.into_iter()
    }
    pub fn iter(&self) -> DTensorIteratorRef<'_, T> {
        self.into_iter()
    }
    pub fn into_iter2(self) -> DTensorIterator2<T> {
        DTensorIterator2::new(self)
    }

    /// Returns the first element of this tensor. Useful for getting scalar values out.
    /// 
    pub fn pop(mut self) -> T {
        self.contents.remove(0)
    }

    pub fn make_dims_amount(&mut self, amount : usize) {
        let diff = (self.dims.len() as isize - amount as isize).abs() as usize;

        if diff == 0 {
            return;
        }

        for _ in 0..diff {
            self.dims.insert(0, 1);
            self.strides.insert(0, 1);
        }
    }

    pub fn fix(&mut self) {
        let new_strides = calculate_strides(self.dims());

        if new_strides == self.strides {
            return;
        }

        let mut indices = Vec::with_capacity(self.dims.len());
        for _ in 0..self.dims.len() {
            indices.push(0);
        }

        let mut new_contents = Vec::with_capacity(self.contents.len());

        for i in 0..self.contents.len() {
            num_to_indices(i, self.dims(), &mut indices);
            let index = indices_to_ind(&self.strides, &indices);

            new_contents.push(unsafe { std::ptr::read(&self.contents[index] as *const T) });
        }

        unsafe {
            self.contents.set_len(0);
        }
        self.contents = new_contents;
        self.strides = new_strides
    }
}
impl<T> DTensor<DTensor<T>> {
    /// Returns the flattened version of self
    /// Assumes all tensors in this tensor are of the same shape
    pub fn flatten(self) -> DTensor<T> {
        if self.contents.len() == 0 {
            return self.pop()
        }

        let mut next_dims = self.dims.clone();
        next_dims.extend(self.contents[0].dims.clone());

        let mut vec = Vec::with_capacity(next_dims.mul_all());

        self.into_iter().for_each(|f| f.into_iter().for_each(|a| vec.push(a)));
        
        DTensor::new(vec, next_dims)
    }

    pub fn flatten_to_dims(self, dims : Vec<usize>, jump : usize) -> DTensor<T> {
        if dims.len() == 0 {
            return self.pop()
        }

        let mut vec = Vec::with_capacity(dims.mul_all());

        self.into_iter().map(|a| a.into_iter2()).chunks(jump).for_each(|a| {

            a.to_parallel().for_each(|a| {
                vec.extend(a.into_iter())
            })
        });

        DTensor::new(vec, dims)
    }
}
impl<T : Zero> ZeroTensor for DTensor<T> {
    fn zero_tensor(dims : &[usize]) -> Self {
        let len = dims.mul_all();
        let mut v = Vec::with_capacity(len);
        v.extend(std::iter::repeat_with(T::zero).take(len));

        let mut dims2 = Vec::with_capacity(dims.len());
        dims2.extend(dims);

        Self::new(v, dims2)
    }
}
impl<T : One + Zero> DTensor<T> {
    pub fn eye(columns : usize, rows : usize) -> Self {
        let len = rows * columns;
        let mut contents = Vec::with_capacity(len);

        let dims = vec![columns, rows];
        let mut indices = vec![0, 0];

        for i in 0..len {
            num_to_indices(i, &dims, &mut indices);

            if indices[0] == indices[1] {
                contents.push(T::one());
            } else {
                contents.push(T::zero());
            }
        }

        Self::new(contents, dims)
    }
}

// Conversion Impls
impl<T> From<Vec<T>> for DTensor<T> {
    fn from(value: Vec<T>) -> Self {
        let len = value.len();
        Self::new_unchecked(value, vec![len], vec![1])
    }
}

// Iterator Implementations
pub struct DTensorIterator2<T> {
    inner_iter : Chunks<DTensorIterator<T>>,
}
impl<T> DTensorIterator2<T> {
    pub fn new(t : DTensor<T>) -> Self {
        let jump = t.dims.mul_from(1);
        Self { inner_iter: t.into_iter().chunks(jump) }
    }
}
impl<T> Iterator for DTensorIterator2<T> {
    type Item = DTensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner_iter.next() {
            Some(v) => {
                Some(DTensor::new(v, self.inner_iter.iter.tensor.dims[1..].to_vec()))
            },
            None => {
                None
            },
        }
    }
}

pub struct DTensorIterator<T> {
    tensor: DTensor<T>,
    cur: usize,
    rem_indices: Option<(HashSet<usize>, Vec<usize>)>,
    len : usize
}
impl<T> DTensorIterator<T> {
    pub fn new(tensor: DTensor<T>) -> Self {
        if tensor.strides() != calculate_strides(tensor.dims()) {
            Self {
                cur: 0,
                len : tensor.dims.mul_all(),
                rem_indices: Some((
                    HashSet::from_iter((0..tensor.contents.len()).into_iter()),
                    vec![0; tensor.dims.len()],
                )),
                tensor,
            }
        } else {
            Self {
                cur: 0,
                len : tensor.dims.mul_all(),
                tensor,
                rem_indices: None,
            }
        }
    }
}
impl<T> Iterator for DTensorIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.len {
            return None;
        }

        let ret = match &mut self.rem_indices {
            Some(v) => {
                num_to_indices(self.cur, self.tensor.dims(), &mut v.1);
                let index = indices_to_ind(self.tensor.strides(), &v.1);
                v.0.remove(&index);

                Some(unsafe { std::ptr::read(&self.tensor[index] as *const T) })
            }
            None => Some(unsafe { std::ptr::read(&self.tensor[self.cur] as *const T) }),
        };

        self.cur += 1;
        return ret;
    }
}
impl<T> Drop for DTensorIterator<T> {
    fn drop(&mut self) {
        match &mut self.rem_indices {
            Some(v) => {
                unsafe {
                    v.0.iter().for_each(|a| {
                        // Drop all items which were not yielded in the iterator
                        std::ptr::drop_in_place(&mut self.tensor[*a] as *mut T)
                    });
                    // Set the length to zero to prevent drop being called on memory
                    self.tensor.contents.set_len(0);
                }
            }
            None => {
                unsafe {
                    // Drop the items from the current position onward
                    std::ptr::drop_in_place(
                        &mut (&mut self.tensor as &mut [T])[self.cur..] as *mut [T],
                    );
                    // Set the length to zero to prevent drop being called on memory
                    self.tensor.contents.set_len(0);
                }
            }
        }
    }
}
impl<T> IntoIterator for DTensor<T> {
    type Item = T;

    type IntoIter = DTensorIterator<T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

pub struct DTensorIteratorRef<'a, T> {
    tensor: &'a DTensor<T>,
    ind: Option<Vec<usize>>,
    cur: usize,
    len : usize
}
impl<'a, T> DTensorIteratorRef<'a, T> {
    pub fn new(tensor: &'a DTensor<T>) -> Self {
        if tensor.strides() != calculate_strides(tensor.dims()) {
            Self {
                cur: 0,
                len : tensor.dims().mul_all(),
                ind: Some(vec![0; tensor.dims().len()]),
                tensor,
            }
        } else {
            Self {
                cur: 0,
                len : tensor.dims().mul_all(),
                tensor,
                ind: None,
            }
        }
    }
}
impl<'a, T> Iterator for DTensorIteratorRef<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.len {
            return None;
        }

        let index = match &mut self.ind {
            None => self.cur,
            Some(indices) => {
                num_to_indices(self.cur, self.tensor.dims(), indices);
                indices_to_ind(self.tensor.strides(), &indices)
            }
        };

        self.cur += 1;
        return Some(&self.tensor.contents[index]);
    }
}
impl<'a, T> IntoIterator for &'a DTensor<T> {
    type Item = &'a T;

    type IntoIter = DTensorIteratorRef<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

pub struct DTensorIteratorMut<'a, T> {
    tensor: &'a mut DTensor<T>,
    ind: Option<Vec<usize>>,
    cur: usize,
    len: usize
}
impl<'a, T> DTensorIteratorMut<'a, T> {
    pub fn new(tensor: &'a mut DTensor<T>) -> Self {
        if tensor.strides() != calculate_strides(tensor.dims()) {
            Self {
                cur: 0,
                len : tensor.dims.mul_all(),
                ind: Some(vec![0; tensor.dims().len()]),
                tensor,
            }
        } else {
            Self {
                cur: 0,
                len: tensor.dims.mul_all(),
                tensor,
                ind: None,
            }
        }
    }
}
impl<'a, T> Iterator for DTensorIteratorMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.len {
            return None;
        }

        let index = match &mut self.ind {
            None => self.cur,
            Some(indices) => {
                num_to_indices(self.cur, self.tensor.dims(), indices);
                indices_to_ind(self.tensor.strides(), &indices)
            }
        };

        self.cur += 1;
        return Some(unsafe { &mut *(&mut self.tensor.contents[index] as *mut T) });
    }
}
impl<'a, T> IntoIterator for &'a mut DTensor<T> {
    type Item = &'a mut T;

    type IntoIter = DTensorIteratorMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

impl<A> FromIterator<A> for DTensor<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<A>>().into()
    }
}

// Operator Implementations
impl<T: PartialEq> PartialEq for DTensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dims == other.dims && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}
impl<T: Eq> Eq for DTensor<T> {}

impl<T> Deref for DTensor<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.contents
    }
}
impl<T> DerefMut for DTensor<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.contents
    }
}

impl<T> TransposeAssign for DTensor<T> {
    fn transpose_assign(&mut self) {
        self.permutate_dims(|f| f.reverse())
    }
}
impl<T> Transpose for DTensor<T> {
    type Output = Self;

    fn transpose(mut self) -> Self::Output {
        self.transpose_assign();
        self
    }
}
impl<T : Clone> Transpose for &DTensor<T> {
    type Output = DTensor<T>;

    fn transpose(self) -> Self::Output {
        self.clone().transpose()
    }
}

impl<T> PermutateDims for DTensor<T> {
    fn permutate_dims<F: FnMut(&mut [usize])>(&mut self, mut f: F) {
        f(&mut self.dims);
        f(&mut self.strides)
    }
}
impl<T: Clone> PermutatedDims for &DTensor<T> {
    type Output = DTensor<T>;

    fn permutated_dims<F: FnMut(&mut [usize])>(self, f: F) -> Self::Output {
        self.clone().permutated_dims(f)
    }
}

// Add implementation
impl<T: AddAssign> AddAssign for DTensor<T> {
    fn add_assign(&mut self, rhs: Self) {
        assert_eq!(self.dims(), rhs.dims());

        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(a, b)| *a += b);
    }
}
impl<T> AddAssign<&DTensor<T>> for DTensor<T> 
    where for<'a> T : AddAssign<&'a T>
{
    fn add_assign(&mut self, rhs: &DTensor<T>) {
        assert_eq!(self.dims(), rhs.dims());

        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(a, b)| *a += b);
    }
}

impl<T> Add for DTensor<T>
    where T : Add<T, Output = T>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims());

        let dims = self.dims.clone();
        
        Self::Output::new(self.into_iter().zip(rhs.into_iter()).map(|(a, b)| a + b).collect(), dims)
    }
}
impl<T> Add for &DTensor<T>
    where for<'a> &'a T : Add<&'a T, Output = T> 
{
    type Output = DTensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims());

        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().zip(rhs.into_iter()).map(|(a, b)| a + b).collect(), dims)
    }
}
impl<T> Add<DTensor<T>> for &DTensor<T> 
    where for<'a> &'a T : Add<T, Output = T>
{
    type Output = DTensor<T>;

    fn add(self, rhs: DTensor<T>) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims());

        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().zip(rhs.into_iter()).map(|(a, b)| a + b).collect(), dims)
    }
}
impl<T> Add<&DTensor<T>> for DTensor<T> 
    where for<'a> T : Add<&'a T, Output = T> 
{
    type Output = DTensor<T>;

    fn add(self, rhs: &DTensor<T>) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims());

        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().zip(rhs.into_iter()).map(|(a, b)| a + b).collect(), dims)
    }
}

// Scalar Addition
impl<T : Clone + AddAssign> AddAssign<T> for DTensor<T> 
{
    fn add_assign(&mut self, rhs: T) {
        self.iter_mut()
            .for_each(|a| *a += rhs.clone())
    }
}
impl<T> AddAssign<&T> for DTensor<T>
    where for<'a> T : AddAssign<&'a T>
{
    fn add_assign(&mut self, rhs: &T) {
        self.iter_mut()
        .for_each(|a| *a += rhs)
    }
}

impl<T> Add<T> for DTensor<T> 
    where T : Clone + Add<T, Output = T>
{
    type Output = DTensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().map(|a| a + rhs.clone()).collect(), dims)
    }
}
impl<T : Clone> Add<T> for &DTensor<T> 
    where for<'a> &'a T : Add<T, Output = T>
{
    type Output = DTensor<T>;

    fn add(self, rhs: T) -> Self::Output {
        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().map(|a| a + rhs.clone()).collect(), dims)
    }
}
impl<T> Add<&T> for DTensor<T> 
    where for<'a> T : Add<&'a T, Output = T>
{
    type Output = DTensor<T>;

    fn add(self, rhs: &T) -> Self::Output {
        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().map(|a| a + rhs).collect(), dims)
    }
}
impl<T> Add<&T> for &DTensor<T> 
    where for<'a> &'a T : Add<&'a T, Output = T>
{
    type Output = DTensor<T>;

    fn add(self, rhs: &T) -> Self::Output {
        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().map(|a| a + rhs).collect(), dims)
    }
}

// Sub implementation
impl<T : SubAssign> SubAssign for DTensor<T> {
    fn sub_assign(&mut self, rhs: Self) {
        assert_eq!(self.dims(), rhs.dims());

        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(a, b)| *a -= b);
    }
}
impl<T> SubAssign<&DTensor<T>> for DTensor<T> 
    where for<'a> T : SubAssign<&'a T>
{
    fn sub_assign(&mut self, rhs: &DTensor<T>) {
        assert_eq!(self.dims(), rhs.dims());

        self.iter_mut()
            .zip(rhs.into_iter())
            .for_each(|(a, b)| *a -= b);
    }
}

impl<T> Sub for DTensor<T>
    where T : Sub<T, Output = T>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims());

        let dims = self.dims.clone();
        
        Self::Output::new(self.into_iter().zip(rhs.into_iter()).map(|(a, b)| a - b).collect(), dims)
    }
}
impl<T> Sub for &DTensor<T>
    where for<'a> &'a T : Sub<&'a T, Output = T> 
{
    type Output = DTensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims());

        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().zip(rhs.into_iter()).map(|(a, b)| a - b).collect(), dims)
    }
}
impl<T> Sub<DTensor<T>> for &DTensor<T> 
    where for<'a> &'a T : Sub<T, Output = T>
{
    type Output = DTensor<T>;

    fn sub(self, rhs: DTensor<T>) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims());

        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().zip(rhs.into_iter()).map(|(a, b)| a - b).collect(), dims)
    }
}
impl<T> Sub<&DTensor<T>> for DTensor<T> 
    where for<'a> T : Sub<&'a T, Output = T> 
{
    type Output = DTensor<T>;

    fn sub(self, rhs: &DTensor<T>) -> Self::Output {
        assert_eq!(self.dims(), rhs.dims());

        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().zip(rhs.into_iter()).map(|(a, b)| a - b).collect(), dims)
    }
}

// Scalar subtraction
impl<T : Clone + SubAssign> SubAssign<T> for DTensor<T> 
{
    fn sub_assign(&mut self, rhs: T) {
        self.iter_mut()
            .for_each(|a| *a -= rhs.clone())
    }
}
impl<T> SubAssign<&T> for DTensor<T>
    where for<'a> T : SubAssign<&'a T>
{
    fn sub_assign(&mut self, rhs: &T) {
        self.iter_mut()
        .for_each(|a| *a -= rhs)
    }
}

impl<T> Sub<T> for DTensor<T> 
    where T : Clone + Sub<T, Output = T>
{
    type Output = DTensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().map(|a| a - rhs.clone()).collect(), dims)
    }
}
impl<T : Clone> Sub<T> for &DTensor<T> 
    where for<'a> &'a T : Sub<T, Output = T>
{
    type Output = DTensor<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().map(|a| a - rhs.clone()).collect(), dims)
    }
}
impl<T> Sub<&T> for DTensor<T> 
    where for<'a> T : Sub<&'a T, Output = T>
{
    type Output = DTensor<T>;

    fn sub(self, rhs: &T) -> Self::Output {
        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().map(|a| a - rhs).collect(), dims)
    }
}
impl<T> Sub<&T> for &DTensor<T> 
    where for<'a> &'a T : Sub<&'a T, Output = T>
{
    type Output = DTensor<T>;

    fn sub(self, rhs: &T) -> Self::Output {
        let dims = self.dims.clone();

        Self::Output::new(self.into_iter().map(|a| a - rhs).collect(), dims)
    }
}

// Mul implementation
impl<T> Mul<T> for DTensor<T> 
    where T : Clone + Mul<Output = T> + MulAssign
{
    type Output = DTensor<T>;

    fn mul(mut self, rhs: T) -> Self::Output {
        self.deref_mut().iter_mut().for_each(|a| {
            *a *= rhs.clone()
        });

        self
    }
}

impl<T : Debug> Mul for DTensor<T> 
    where T : Mul<Output = T> + Clone + Add<Output = T> + MulAssign + Zero + AddAssign
{
    type Output = Self;

    fn mul(mut self, mut rhs: Self) -> Self::Output {
        if self.dims().len() == 0 {
            // Scalar multiplication
            return rhs * self.pop()
        } else if rhs.dims().len() == 0 {
           return self * rhs.pop()
        } else if rhs.dims().len() == 1 && self.dims.len() == 1 {
            // Dot product
            return Self::Output::scalar(self.into_iter().zip(rhs.into_iter()).fold(None, |c, (a, b)| {
                match c {
                    Some(v) => {
                        Some(v + a * b)
                    }
                    None => {
                        Some(a * b)
                    }
                }
            }).unwrap())
        }

        // Transpose to be able to multiply rows by columns
        rhs.transpose_assign();

        if rhs.dims().len() == 1 {
            self.transpose_assign();
            let init = ZeroTensor::zero_tensor(rhs.dims());

            return self.into_iter2().zip(rhs.into_iter()).map(|(a, b)| {
                a * b
            }).fold(init, |c : DTensor<T>, a| { c + a })
        } else if rhs.dims.len() < self.dims.len() {
            todo!()
        }

        DTensor::<DTensor<T>>::from(self.into_iter2().map(|a| {
            DTensor::<DTensor<T>>::from(rhs.clone().into_iter2().map(|b| {

                a.clone() * b
            }).collect::<Vec<DTensor<T>>>()).flatten()
        }).collect::<Vec<DTensor<T>>>()).flatten()
    }
}

impl<T> KronProduct for DTensor<T> 
    where T : Clone + Mul<Output = T>,
    DTensor<T> : Mul<T, Output = DTensor<T>>
{
    type Output = DTensor<T>;

    fn kron(mut self, mut rhs : Self) -> Self::Output {
        if self.dims().len() < rhs.dims().len() {
            self.make_dims_amount(rhs.dims().len());
        } else {
            rhs.make_dims_amount(self.dims().len());
            println!("{:?}", rhs.dims());
        }

        let new_dims = self.dims().iter().zip(rhs.dims().iter()).map(|(a, b)| a * b).collect::<Vec<usize>>();

        let jump = rhs.dims()[rhs.dims.len() - 1];

        self.into_iter().map(|a| rhs.clone() * a).collect::<Vec<DTensor<T>>>().to_tensor().flatten_to_dims(new_dims, jump)
    }
}