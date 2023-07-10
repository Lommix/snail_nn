use snail_nn::{mat, prelude::*};

fn main() {
    let mut nn = Model::new(&[2, 3, 3, 1]);

    let input = mat!((0, 1), (1, 0), (1, 1), (0, 0));
    let output = mat!((1), (1), (0), (0));

    let batch = TrainingBatch::new(input.clone(), output.clone());

    let mut timer: u128 = 0;

    loop {
        timer += 1;
        let (w, b) = nn.error_gradiant(&batch);

        nn.learn(w, b, 1.0);

        if timer % 10 == 0 {
            //clear terminal
            print!("{esc}c", esc = 27 as char);
            println!("cost: {}", nn.cost(&batch));

            input.iter_rows().for_each(|x| {
                let i = MatF64::row_from_slice(x);
                let out = nn.forward(&i);
                println!("in: {:?} out: {:?}", i, out.last().unwrap());
            });
        }
    }
}
