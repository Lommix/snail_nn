use snail_nn::prelude::*;

const ascii_grayscale: &str = " ´.,+=*#$@";

fn load_image(path: &str) -> (usize, usize, image::RgbImage) {
    let image = image::open(path).unwrap();
    (
        image.width() as usize,
        image.height() as usize,
        image.grayscale().to_rgb8(),
    )
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let (h, w, img) = load_image("./assets/3.png");

    let img_data = img
        .chunks(3)
        .map(|x| (x.iter().map(|n| *n as f64).sum::<f64>() / 3.0) / 255.0)
        .collect::<Vec<f64>>();

    let mut input = MatF64::empty(0, 2);
    let mut expected = MatF64::empty(0, 1);

    img_data.iter().enumerate().for_each(|(i, d)| {
        let x = ((i % w) as f64) / (w - 1) as f64;
        let y = ((i / w) as f64) / (h - 1) as f64;

        input.add_row(&[x, y]);
        expected.add_row(&[d.clone()]);
    });

    let batch = TrainingBatch::new(input, expected);

    let mut nn = Model::new(&[2, 5, 5, 1]);
    let mut epoch: u128 = 0;

    nn.set_activation(Activation::Sigmoid);
    let learning_rate = 0.1;

    loop {
        epoch += 1;

        let (wg, bg) = nn.gradient(&batch);
        nn.learn(wg, bg, learning_rate);

        if epoch % 100 == 0 {
            let original = image(&img_data, w, h);
            let imagined = imagine(&mut nn, 50, 50);
            let cost = nn.cost(&batch);

            print!("{esc}c", esc = 27 as char);
            print!("epoch: {} cost: {} \n", epoch, cost);
            println!("-----------------");
            print!("{}", original);
            print!("{}", imagined);
        }
    }
}

fn image(data: &Vec<f64>, width: usize, height: usize) -> String {
    let mut text = String::new();
    for y in 0..height {
        for x in 0..width {
            let s = data[y * width + x];
            let grayscale = (ascii_grayscale.len() - 2) as f64 * s;
            text += &format!(
                " {}",
                ascii_grayscale.chars().nth(grayscale as usize).unwrap()
            );
        }
        text += &format!("\n");
    }
    text
}
fn imagine(nn: &mut Model, width: usize, height: usize) -> String {
    let mut text = String::new();
    for y in 0..height {
        for x in 0..width {
            let xf = x as f64 / width as f64;
            let yf = y as f64 / height as f64;

            let input = MatF64::row_from_slice(&[xf, yf]);
            let out = nn.forward(&input).last().unwrap()[(0, 0)];
            let grayscale = (ascii_grayscale.len() - 2) as f64 * out.abs().clamp(0.0, 1.0);
            text += &format!(
                " {}",
                ascii_grayscale.chars().nth(grayscale as usize).unwrap()
            );
        }
        text += &format!("\n");
    }
    text
}
