use crate::{act::Activation, batch::TrainingBatch, mat::MatF64};

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

    pub fn forward(&self, input: &MatF64) -> Vec<MatF64> {
        assert_eq!(input.len(), self.weights[0].rows());
        let mut output: Vec<MatF64> = Vec::new();

        let mut first = input.dot(&self.weights[0]).add(&self.biases[0]);
        first
            .iter_mut()
            .for_each(|x| *x = self.activation.forward(*x));

        output.push(first);

        for i in 1..self.weights.len() {
            output.push(output[i - 1].dot(&self.weights[i]).add(&self.biases[i]));
        }

        output
    }

    pub fn cost(&self, batch: &TrainingBatch) -> f64 {
        let mut cost = 0.0;

        batch.iter().for_each(|(i, e)| {
            let activation = self.forward(&MatF64::row_from_slice(i));
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

    pub fn error_gradiant(&self, batch: &TrainingBatch) -> (Vec<MatF64>, Vec<MatF64>) {
        let mut weight_gradiant: Vec<MatF64> =
            self.weights.iter().map(|m| MatF64::clone_zero(m)).collect();

        let mut bias_gradiant: Vec<MatF64> =
            self.biases.iter().map(|m| MatF64::clone_zero(m)).collect();

        batch.iter().for_each(|(i, e)| {

            let input = MatF64::row_from_slice(i);
            let expected = MatF64::row_from_slice(e);

            let activation = self.forward(&input);
            let output = activation.last().unwrap();
            let error = &expected - output;

            let mut current_error = error;

            for l in (1..self.weights.len()).rev() {
                let mut delta = activation[l].clone();

                delta
                    .iter_mut()
                    .for_each(|x| *x = self.activation.derivative(*x));

                delta *= current_error;

                // make it without copy
                let mut prev_weights = self.weights[l].clone();
                prev_weights.transpose();

                let prev_delta = delta.dot(&prev_weights);

                let mut prev = activation[l - 1].clone();
                prev.transpose();

                let wdelta = prev.dot(&prev_delta);

                current_error = prev_delta;

                // -----------------------------------------

                weight_gradiant[l - 1]
                    .iter_mut()
                    .zip(wdelta.iter())
                    .for_each(|(a, b)| *a += *b);

                bias_gradiant[l - 1]
                    .iter_mut()
                    .zip(delta.iter())
                    .for_each(|(a, b)| *a += *b);

                // weight_gradiant[l - 1].add(&wdelta);

                // let current_deriv =
                //     self.activation[l].clone_apply(|a| self.func.calc_derivative(a.clone()));
                //
                // let current_delta = &gradiant_error[l] * &current_deriv;
                // // update weights gradiant
                // let mut prev_activation = self.activation[l - 1].clone();
                // prev_activation.transpos();
                // let wdelta = prev_activation.dot(&current_delta);
                //
                // gradiant_weights[l - 1].add(&wdelta);
                // gradiant_bias[l - 1].add(&current_delta);
            }
        });

        for i in 0..self.weights.len() {
            weight_gradiant[i]
                .iter_mut()
                .for_each(|x| *x /= batch.len() as f64);
            bias_gradiant[i]
                .iter_mut()
                .for_each(|x| *x /= batch.len() as f64);
        }

        (weight_gradiant, bias_gradiant)
        // -------------------------------------------------
        // let activation = self.forward(input);
        // let output = activation.last().unwrap();
        // let error = expected - output;
        // gradiant.push(error);

        // train_data.iter().for_each(|t|{
        //     let input = t.iter().take(self.weights[0].rows());
        //     let expected = t.iter().skip(self.weights[0].rows());
        //
        // });
        //
        // for l in (0..self.weights.len()).rev() {
        //     let mut delta = activation[l].clone();
        //
        //     delta
        //         .iter_mut()
        //         .for_each(|x| *x = self.activation.derivative(*x));
        //     delta *= gradiant.last().unwrap();
        //
        //     // make it without copy
        //     let mut prev_weights = self.weights[l].clone();
        //     prev_weights.transpose();
        //
        //     gradiant.push(delta.dot(&prev_weights));
        // }
        // -------------------------------------------------
    }

    pub fn learn(&mut self, weight_gradiant: Vec<MatF64>, bias_gradiant: Vec<MatF64>, rate: f64) {

        assert_eq!(weight_gradiant.len(), self.weights.len());
        assert_eq!(bias_gradiant.len(), self.biases.len());

        for i in 0..self.weights.len() {
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
    let output = model.forward(&input);
    // println!("{:?}", output);
}

#[test]
fn test_gradiant() {
    let mut model = Model::new(&[2, 3, 5, 1]);
    let input = MatF64::random_rows(2);
    let expected = MatF64::random_rows(1);

    let train = TrainingBatch::new(input, expected);
    let (w, b) = model.error_gradiant(&train);

    assert_eq!(w.len(), 3);
    assert_eq!(b.len(), 3);

    model.learn(w, b, &1.0);
}
