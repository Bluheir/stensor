use crate::tensor::DTensor;


#[derive(Debug, Clone, Copy, Default)]
pub struct StridePair<'a, 'b, T> {
    data: &'a [T],
    strides: &'b [usize],
    dims : &'b [usize]
}

impl<'a, 'b, T> StridePair<'a, 'b, T> {
    pub fn new(data: &'a [T], strides: &'b [usize], dims : &'b [usize]) -> Self {
        Self { data, strides, dims }
    }

    pub fn next(&self, ind: usize) -> Self {
        let stride = self.strides[0];

        Self {
            data: &self.data[ind * stride..((ind + 1) * stride)],
            strides: &self.strides[1..],
            dims: &self.dims[1..]
        }
    }
}

// Iterator Implementations
#[derive(Clone, Copy)]
pub struct StridesPairIter<'a, 'b, T> {
    pair : StridePair<'a, 'b, T>,
    cur : usize,
    len : usize
}

impl<'a, 'b, T> StridesPairIter<'a, 'b, T> {
    pub fn new(pair : StridePair<'a ,'b, T>) -> Self {
        Self {
            len : pair.dims[0],
            cur: 0,
            pair,
        }
    }
}

impl<'a, 'b, T> Iterator for StridesPairIter<'a ,'b, T> {
    type Item = StridePair<'a, 'b, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == self.cur {
            return None;
        }

        let ret =  Some(self.pair.next(self.cur));
        self.cur += 1;
        ret
    }
}

impl<'a, 'b, T> IntoIterator for StridePair<'a ,'b, T> {
    type Item = StridePair<'a, 'b, T>;

    type IntoIter = StridesPairIter<'a, 'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self)
    }
}

// Operators
impl<'a, 'b, T> Into<DTensor<T>> for StridePair<'a, 'b, T> 
    where T : Clone
{
    fn into(self) -> DTensor<T> {
        let dims = self.dims.to_vec();
        let strides = self.strides.to_vec();

        let mut elems = Vec::with_capacity(self.data.len());
        elems.extend_from_slice(self.data);

        DTensor::new_unchecked(elems, dims, strides)
    }
}

#[derive(Debug, Default)]
pub struct StridePairMut<'a, 'b, T> {
    data: &'a mut [T],
    strides: &'b [usize],
    dims : &'b [usize]
}

impl<'a, 'b, T> StridePairMut<'a, 'b, T> {
    pub fn new(data: &'a mut [T], strides: &'b [usize], dims : &'b [usize]) -> Self {
        Self { data, strides, dims }
    }

    pub fn next(&'a self, ind: usize) -> StridePair<'a, 'b, T> {
        let stride = self.strides[0];

        StridePair {
            data: &self.data[ind * stride..((ind + 1) * stride)],
            strides: &self.strides[1..],
            dims : &self.dims[1..]
        }
    }

    pub fn next_mut(&'a mut self, ind: usize) -> Self {
        let stride = self.strides[0];

        Self {
            data: &mut self.data[ind * stride..((ind + 1) * stride)],
            strides: &self.strides[1..],
            dims: &self.dims[1..],
        }
    }
}
impl<'a, 'b, T> IntoIterator for StridePairMut<'a ,'b, T> {
    type Item = StridePair<'a, 'b, T>;

    type IntoIter = StridesPairIter<'a, 'b, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::new(self.into())
    }
}

impl<'a, 'b, T> Into<StridePair<'a, 'b, T>> for StridePairMut<'a, 'b, T> {
    fn into(self) -> StridePair<'a, 'b, T> {
        StridePair::new(self.data, self.strides, self.dims)
    }
}