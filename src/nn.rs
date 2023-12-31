use crate::{act::Activation, batch::TrainingBatch, mat::MatF64};
use rayon::prelude::*;

pub struct Model {
    weights: Vec<MatF64>,
    biases: Vec<MatF64>,
    activation: Activation,
}

impl Model {
    pub fn new(arch: &[usize]) -> Model {
        let mut weights: Vec<MatF64> = Vec::with_capacity(arch.len());
        let mut biases: Vec<MatF64> = Vec::with_capacity(arch.len());

        for i in 0..arch.len() - 1 {
            weights.push(MatF64::rand(arch[i], arch[i + 1]));
            biases.push(MatF64::zeros_row(arch[i + 1]));
        }

        Model {
            weights,
            biases,
            activation: Activation::Sigmoid,
        }
    }

    pub fn set_activation(&mut self, activation: Activation) {
        self.activation = activation;
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let output = self.activate(&MatF64::row_from_slice(input));
        output.last().unwrap().to_vec()
    }

    pub fn activate(&self, input: &MatF64) -> Vec<MatF64> {
        assert_eq!(input.len(), self.weights[0].rows());

        let mut output: Vec<MatF64> = Vec::new();
        output.push(input.clone());

        for i in 0..self.weights.len() {
            let mut next = output
                .last()
                .unwrap()
                .dot(&self.weights[i])
                .add(&self.biases[i]);
            next.iter_mut()
                .for_each(|v| *v = self.activation.forward(*v));
            output.push(next);
        }
        output
    }

    pub fn cost(&self, batch: &TrainingBatch) -> f64 {
        let mut cost = 0.0;

        batch.iter().for_each(|(i, e)| {
            let activation = self.activate(&MatF64::row_from_slice(i));
            let output = activation.last().unwrap();
            let expected = MatF64::row_from_slice(e);

            cost += output
                .iter()
                .zip(expected.iter())
                .map(|(x, y)| {
                    let d = x - y;
                    d * d
                })
                .sum::<f64>();
        });

        cost / batch.len() as f64
    }

    pub fn gradient(&self, batch: &TrainingBatch) -> (Vec<MatF64>, Vec<MatF64>) {
        let mut weight_gradient: Vec<MatF64> =
            self.weights.iter().map(|m| MatF64::clone_zero(m)).collect();

        let mut bias_gradient: Vec<MatF64> =
            self.biases.iter().map(|m| MatF64::clone_zero(m)).collect();

        let o = batch
            .iter()
            .par_bridge()
            .map(|(i, e)| {
                let mut wout: Vec<MatF64> = Vec::new();
                let mut bout: Vec<MatF64> = Vec::new();

                let input = MatF64::row_from_slice(i);
                let expected = MatF64::row_from_slice(e);

                let activation = self.activate(&input);
                let output = activation.last().unwrap().clone();
                // output.iter_mut().for_each(|x| *x *= 2.0);

                let error = &output - &expected;

                let mut current_error = error;

                for l in (1..activation.len()).rev() {
                    let mut delta = activation[l].clone();

                    delta
                        .iter_mut()
                        .for_each(|x| *x = self.activation.derivative(*x));

                    delta *= current_error;

                    let mut prev_weights = self.weights[l - 1].clone();
                    prev_weights.transpose();

                    let prev_error = delta.dot(&prev_weights);

                    let mut prev_activation = activation[l - 1].clone();
                    prev_activation.transpose();

                    let wdelta = prev_activation.dot(&delta);

                    current_error = prev_error;

                    wout.push(wdelta);
                    bout.push(delta);
                }

                wout.reverse();
                bout.reverse();

                (wout, bout)
            })
            .collect::<Vec<(Vec<MatF64>, Vec<MatF64>)>>();

        // --- sum gradient ---
        o.iter().for_each(|(w, b)| {
            weight_gradient
                .iter_mut()
                .zip(w.iter())
                .for_each(|(a, b)| *a += b);
            bias_gradient
                .iter_mut()
                .zip(b.iter())
                .for_each(|(a, b)| *a += b);
        });

        // --- avarage gradient ---
        for i in 0..self.weights.len() {
            weight_gradient[i]
                .iter_mut()
                .for_each(|x| *x /= batch.len() as f64);
            bias_gradient[i]
                .iter_mut()
                .for_each(|x| *x /= batch.len() as f64);
        }

        (weight_gradient, bias_gradient)
    }

    pub fn learn(&mut self, weight_gradiant: Vec<MatF64>, bias_gradiant: Vec<MatF64>, rate: f64) {
        assert_eq!(weight_gradiant.len(), self.weights.len());
        assert_eq!(bias_gradiant.len(), self.biases.len());

        for i in 0..self.weights.len() {
            assert_eq!(weight_gradiant[i].len(), self.weights[i].len());
            assert_eq!(bias_gradiant[i].len(), self.biases[i].len());

            self.weights[i]
                .iter_mut()
                .zip(weight_gradiant[i].iter())
                .for_each(|(a, b)| *a -= *b * rate);

            self.biases[i]
                .iter_mut()
                .zip(bias_gradiant[i].iter())
                .for_each(|(a, b)| *a -= *b * rate);
        }
    }
}

#[test]
fn test_forward() {
    let model = Model::new(&[2, 3, 1]);
    let input = MatF64::random_rows(2);
    let output = model.activate(&input);
    assert_eq!(output.len(), 3);
}

// #[bench]
// fn bench_forward(b: &mut test::Bencher) {
//     let model = Model::new(&[2, 9, 9, 1]);
//     let input = MatF64::random_rows(2);
//     b.iter(|| {
//         model.activate(&input);
//     })
// }
//
// #[bench]
// fn bench_dot(b: &mut test::Bencher) {
//     let m1 = MatF64::rand(3, 1);
//     let m2 = MatF64::random_rows(3);
//     b.iter(|| {
//         m1.dot(&m2);
//     })
// }

#[test]
fn test_gradiant() {
    let mut model = Model::new(&[2, 3, 5, 1]);
    let input = MatF64::random_rows(2);
    let expected = MatF64::random_rows(1);

    let train = TrainingBatch::new(input, expected);
    let (w, b) = model.gradient(&train);

    assert_eq!(w.len(), 3);
    assert_eq!(b.len(), 3);

    model.learn(w, b, 1.0);
}
