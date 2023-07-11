use crate::mat::MatF64;

pub struct TrainingBatch {
    pub input: MatF64,
    pub expected: MatF64,
    index : usize,
}

impl TrainingBatch {
    pub fn new(input: MatF64, expected: MatF64) -> TrainingBatch {
        assert_eq!(input.rows(), expected.rows());
        TrainingBatch { input, expected, index: 0 }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[f64], &[f64])> {
        self.input.iter_rows().zip(self.expected.iter_rows())
    }

    pub fn len(&self) -> usize {
        self.input.len()
    }

    pub fn next_chunk(&mut self, size: usize) -> Self {

        let mut input = MatF64::empty(0, self.input.cols());
        let mut expected = MatF64::empty(0, self.expected.cols());

        for i in 0..size {

            let offset = rand::random::<usize>() % 3;
            let index = ( self.index + i + offset) % self.input.rows();

            input.add_row(self.input.get_row(index));
            expected.add_row(self.expected.get_row(index));

            self.index += 1;
        }

        TrainingBatch::new(input, expected)
    }

    pub fn random_chunk(&self, size: usize) -> Self {

        let mut input = MatF64::empty(0, self.input.cols());
        let mut expected = MatF64::empty(0, self.expected.cols());

        let offset = rand::random::<usize>();


        for i in 0..size {

            let index = ( offset + i ) % self.input.rows();

            input.add_row(self.input.get_row(index));
            expected.add_row(self.expected.get_row(index));

        }

        TrainingBatch::new(input, expected)
    }
}
