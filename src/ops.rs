
pub trait KronProduct {
    type Output;

    fn kron(self, rhs : Self) -> Self::Output;
}


pub trait TransposeAssign {
    fn transpose_assign(&mut self);
}
pub trait Transpose {
    type Output;

    fn transpose(self) -> Self::Output;
}

pub trait PermutateDims {
    fn permutate_dims<F: FnMut(&mut [usize])>(&mut self, f: F);
}

pub trait PermutatedDims {
    type Output;

    fn permutated_dims<F: FnMut(&mut [usize])>(self, f: F) -> Self::Output;
}

impl<T: PermutateDims> PermutatedDims for T {
    fn permutated_dims<F: FnMut(&mut [usize])>(mut self, f: F) -> Self {
        self.permutate_dims(f);
        self
    }

    type Output = Self;
}

pub trait Fix {
    fn fix(&mut self);
}
pub trait Fixed {
    type Output;

    fn fixed(self) -> Self::Output;
}

pub trait Reshape {
    fn reshape(&mut self, shape : Vec<usize>);
}
pub trait Shape {
    type Output;

    fn shape(shape : Vec<usize>) -> Self::Output;
}