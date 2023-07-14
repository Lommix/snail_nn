# [WIP] Snail NN - smol neural network library

fully functional neural network libary with backpropagation and parallelized stochastic gradient descent implementation.


## Examles

Storing images inside the neural network and interpolate between them. Rendering done with egui.


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

let mut nn = Model::new(&[2, 3, 1]);
let mut batch = TrainingBatch::empty(2, 1);
let rate = 1.0;
nn.set_activation(Activation::Sigmoid)

batch.add(&[0.0, 0.0], &[0.0]);
batch.add(&[1.0, 0.0], &[0.0]);
batch.add(&[0.0, 1.0], &[0.0]);
batch.add(&[1.0, 1.0], &[1.0]);

for _ in 0..10000 {
    let (w_gradient, b_gradient) = nn.gradient(&batch);
    nn.learn(w_gradient, b_gradient, rate);
}

println!("{:?}", nn.forward(&[0.0, 0.0]));
```

##  Freatures

- Sigmoid, Tanh & Relu activation functions
- Parallelized stochastic gradient descent
- It works on my machine ¯\\_(ツ)_/¯
- Will gobble up most of your cpu


## Todo

- more examples
- better documentation
- compute shaders with wgpu
