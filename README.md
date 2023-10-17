# [WIP] Snail NN - smol neural network library

Minimalistic CPU based neural network library with backpropagation and parallelized stochastic gradient descent.

## Examples

Storing images inside the neural network, upscaling and interpolate between them.

```bash
cargo run --example imagepol --release
```

![image](docs/example_interpolation.png)

---

The mandatory xor example

```bash
cargo run --example xor --release
```

![image](docs/xor.png)

---

Example Code:

```rust
use snail_nn::prelude::*;

fn main(){
    let mut nn = Model::new(&[2, 3, 1]);
    nn.set_activation(Activation::Sigmoid)

    let mut batch = TrainingBatch::empty(2, 1);
    let rate = 1.0;

    // AND - training data
    batch.add(&[0.0, 0.0], &[0.0]);
    batch.add(&[1.0, 0.0], &[0.0]);
    batch.add(&[0.0, 1.0], &[0.0]);
    batch.add(&[1.0, 1.0], &[1.0]);

    for _ in 0..10000 {
        let (w_gradient, b_gradient) = nn.gradient(&batch.random_chunk(2));
        nn.learn(w_gradient, b_gradient, rate);
    }

    println!("ouput {:?} expected: 0.0", nn.forward(&[0.0, 0.0]));
    println!("ouput {:?} expected: 0.0", nn.forward(&[1.0, 0.0]));
    println!("ouput {:?} expected: 0.0", nn.forward(&[0.0, 1.0]));
    println!("ouput {:?} expected: 1.0", nn.forward(&[1.0, 1.0]));
}
```

## Features

-   Sigmoid, Tanh & Relu activation functions
-   Parallelized stochastic gradient descent

## Todos

-   Wgpu compute shaders
