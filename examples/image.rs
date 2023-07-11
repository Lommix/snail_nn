#![allow(unused)]

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use eframe::egui;
use egui::TextureId;
use egui_extras::RetainedImage;
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

#[derive(Default, Debug, Clone)]
pub struct NContext {
    image_out: image::GrayImage,
    cost: f64,
    epoch: u128,
}

unsafe impl Sync for NContext {}
unsafe impl Send for NContext {}

#[derive(Default)]
struct App {
    context: Arc<Mutex<NContext>>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ctx.request_repaint_after(Duration::from_millis(100));

            let mut img: image::GrayImage;
            let mut cost: f64;
            let mut epoch: u128;

            {
                let c = self.context.lock().unwrap();
                img = c.image_out.clone();
                cost = c.cost;
                epoch = c.epoch;
            }

            let mut i = Vec::new();
            img.pixels().for_each(|p| {
                i.push(p.0[0]);
                i.push(p.0[0]);
                i.push(p.0[0]);
            });

            let color_image = egui::ColorImage::from_rgb(
                [img.width() as usize, img.height() as usize],
                i.as_slice(),
            );

            let mut img = RetainedImage::from_color_image("lol", color_image);
            let size = egui::Vec2::new(img.width() as f32, img.height() as f32);

            ui.label("Epoch: ");
            ui.text_edit_singleline(&mut epoch.to_string());
            ui.label("Cost: ");
            ui.text_edit_singleline(&mut cost.to_string());

            ui.image(img.texture_id(ctx), size * 5.0);
        });
    }
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let (h, w, img) = load_image("./assets/marc.png");

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

    let mut batch = TrainingBatch::new(input, expected);
    let mut nn = Model::new(&[2, 24, 10, 1]);
    let mut epoch: u128 = 0;

    nn.set_activation(Activation::Sigmoid);
    let learning_rate = 1.0;

    let context = Arc::new(Mutex::new(NContext::default()));

    let ctx = context.clone();

    let handle = std::thread::spawn(move || {
        loop {
            epoch += 1;
            let (wg, bg) = nn.gradient(&batch.random_chunk(16));
            nn.learn(wg, bg, learning_rate);

            if epoch % 100 == 0 {
                // let original = image(&img_data, w, h);
                let mut imagined = imagine_img(&mut nn, 100, 100);
                let cost = nn.cost(&batch);

                let b = imagined
                    .iter()
                    .map(|x| (x.clone() * 255.0) as u8)
                    .collect::<Vec<u8>>();
                let image = image::GrayImage::from_vec(100, 100, b).unwrap();

                {
                    let mut c = ctx.lock().unwrap();
                    c.cost = cost;
                    c.epoch = epoch;
                    c.image_out = image;
                }

                // print!("{esc}c", esc = 27 as char);
                // print!("epoch: {} cost: {} \n", epoch, cost);
                // println!("-----------------");
                // // print!("{}", original);
                // print!("{}", imagined);
            }
        }
    });

    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Snail nn Demo",
        options,
        Box::new(|cc| Box::new(App { context: context })),
    );

    // ---------------------------------------
    // loop {
    //     epoch += 1;
    //     let (wg, bg) = nn.gradient(&batch.random_chunk(16));
    //     nn.learn(wg, bg, learning_rate);
    //
    //     if epoch % 100 == 0 {
    //         let original = image(&img_data, w, h);
    //         let imagined = imagine(&mut nn, 60, 60);
    //         let cost = nn.cost(&batch);
    //
    //         print!("{esc}c", esc = 27 as char);
    //         print!("epoch: {} cost: {} \n", epoch, cost);
    //         println!("-----------------");
    //         // print!("{}", original);
    //         print!("{}", imagined);
    //     }
    // }
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

fn imagine_img(nn: &Model, width: usize, height: usize) -> Vec<f64> {
    let mut output = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let xf = x as f64 / width as f64;
            let yf = y as f64 / height as f64;
            let input = MatF64::row_from_slice(&[xf, yf]);
            let out = nn.forward(&input).last().unwrap()[(0, 0)];
            output.push(out);
        }
    }
    output
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
