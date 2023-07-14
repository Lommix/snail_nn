#![allow(unused)]
use snail_nn::{mat, prelude::*};

fn main() {
    let mut nn = Model::new(&[2, 3, 3, 1]);

    let input = mat!((0, 1), (1, 0), (1, 1), (0, 0));
    let output = mat!((1), (1), (0), (0));

    let batch = TrainingBatch::new(input.clone(), output.clone());

    let mut timer: u128 = 0;
    let mut epoch: u128 = 0;

    loop {
        timer += 1;
        let (w, b) = nn.gradient(&batch);
        nn.learn(w, b, 1.0);
        epoch += 1;

        if timer % 100 == 0 {
            //clear terminal
            print!("{esc}c", esc = 27 as char);
            println!("epoch: {}",epoch);
            println!("cost: {}", nn.cost(&batch));

            input.iter_rows().zip(output.iter_rows()).for_each(|x| {
                let i = MatF64::row_from_slice(x.0);
                let o = MatF64::row_from_slice(x.1);
                let out = nn.forward(&i);
                print!("in:[{}] --> ", i);
                print!("out:[{}] ", out.last().unwrap());
                print!("expected: {}", o);
                print!("\n");
            });
        }
    }
}
