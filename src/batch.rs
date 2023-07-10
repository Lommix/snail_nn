use crate::mat::MatF64;

pub struct TrainingBatch {
    pub input: MatF64,
    pub expected: MatF64,
}

impl TrainingBatch {
    pub fn new(input: MatF64, expected: MatF64) -> TrainingBatch {
        assert_eq!(input.rows(), expected.rows());
        TrainingBatch { input, expected }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[f64], &[f64])> {
        self.input.iter_rows().zip(self.expected.iter_rows())
    }

    pub fn len(&self) -> usize {
        self.input.len()
    }
}
