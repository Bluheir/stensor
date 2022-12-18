use crate::tensor::{DTensor, DimsHelper};

pub struct Chunks<I> {
    pub(crate) iter : I,
    jump : usize
}

impl<I> Chunks<I> {
    pub fn new(iter : I, jump : usize) -> Self {
        Self { iter, jump }
    }
    pub fn jump(&self) -> usize {
        self.jump
    }
}

impl<I : Iterator> Iterator for Chunks<I> {
    type Item = Vec<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut ret = Vec::with_capacity(self.jump);

        let first_jump = self.iter.next();

        match first_jump {
            None => {
                None
            }
            Some(v) => {
                ret.push(v);

                for _ in 1..(self.jump) {
                    match self.iter.next() {
                        Some(v) => {
                            ret.push(v)
                        }
                        None => {
                            break
                        }
                    }
                }

                Some(ret)
            }
        }
    }
}

pub trait ChunksExt : Sized {
    fn chunks(self, jump : usize) -> Chunks<Self>;
}

impl<I : Iterator> ChunksExt for I {
    fn chunks(self, jump : usize) -> Chunks<Self> {
        Chunks::new(self, jump)
    }
}

pub trait ToTensor<T> {
    fn to_tensor(self) -> DTensor<T>;
}

impl<T> ToTensor<T> for Vec<T> {
    fn to_tensor(self) -> DTensor<T> {
        self.into()
    }
}

pub trait VecTensorHelpers<T> {
    fn flatten(self) -> DTensor<T>;
    fn flatten_to_dims(self) -> DTensor<T>;
}

impl<T> VecTensorHelpers<T> for Vec<DTensor<T>> {
    fn flatten(self) -> DTensor<T> {

        let mut next_dims = vec![self.len()];
        next_dims.extend_from_slice(self[0].dims());

        let mut vec = Vec::with_capacity(next_dims.mul_all());

        self.into_iter().for_each(|f| f.into_iter().for_each(|a| vec.push(a)));
        
        DTensor::new(vec, next_dims)
    }

    fn flatten_to_dims(self) -> DTensor<T> {
        todo!()
    }
}

pub struct ParallelVecIterator<I> {
    iterators : Vec<I>,
    cur : usize
}

impl<I : Iterator> Iterator for ParallelVecIterator<I> {
    type Item = <I as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let iter = &mut self.iterators[self.cur];

        let ret = iter.next();
        if self.cur == self.iterators.len() - 1 {
            self.cur = 0;
        } else {
            self.cur += 1;
        }

        return ret
    }
}

pub trait ToParallel<I> {
    fn to_parallel(self) -> ParallelVecIterator<I>;
}

impl<I> ToParallel<I> for Vec<I> {
    fn to_parallel(self) -> ParallelVecIterator<I> {
        ParallelVecIterator { iterators: self, cur: 0 }
    }
}