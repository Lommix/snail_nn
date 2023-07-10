pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
}


impl Activation {
    pub fn forward(&self, f: f64) -> f64 {
        match self {
            Activation::Sigmoid => sigmoid(f),
            Activation::Tanh => tanh(f),
            Activation::ReLU => relu(f),
        }
    }

    pub fn derivative(&self, f: f64) -> f64 {
        match self {
            Activation::Sigmoid => sigmoid_derivative(f),
            Activation::Tanh => tanh_derivative(f),
            Activation::ReLU => relu_derivative(f),
        }
    }
}


fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn sigmoid_derivative(x: f64) -> f64 {
    x * (1.0 - x)
}
fn tanh(x: f64) -> f64 {
    (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())
}
fn tanh_derivative(x: f64) -> f64 {
    1.0 - x.tanh().powi(2)
}
