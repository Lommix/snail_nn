#![allow(unused)]

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use eframe::egui;
use egui::{plot::Plot, TextureId};
use egui_extras::RetainedImage;
use snail_nn::prelude::*;

const ASCII_GRAYSCALE: &str = " ´.,+=*#$@";

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
    original: image::GrayImage,
    image_out: image::GrayImage,
    learning_rate: f64,
    cost: f64,
    epoch: u128,
}

unsafe impl Sync for NContext {}
unsafe impl Send for NContext {}

#[derive(Default)]
struct App {
    context: Arc<Mutex<NContext>>,
    cost: Vec<f64>,
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ctx.request_repaint_after(Duration::from_secs_f32(1.0 / 60.0));

            let mut org: image::GrayImage;
            let mut img: image::GrayImage;
            let mut cost: f64;
            let mut epoch: u128;

            {
                let c = self.context.lock().unwrap();
                img = c.image_out.clone();
                cost = c.cost;
                epoch = c.epoch;
                org = c.original.clone();

                self.cost.push(c.cost.clone());
            }

            let mut i = Vec::new();
            let mut j = Vec::new();

            img.pixels().for_each(|p| {
                i.push(p.0[0]);
                i.push(p.0[0]);
                i.push(p.0[0]);
            });

            org.pixels().for_each(|p| {
                j.push(p.0[0]);
                j.push(p.0[0]);
                j.push(p.0[0]);
            });

            // -------- imagined image
            let color_image = egui::ColorImage::from_rgb(
                [img.width() as usize, img.height() as usize],
                i.as_slice(),
            );
            let mut img = RetainedImage::from_color_image("lol", color_image);
            let size = egui::Vec2::new(img.width() as f32, img.height() as f32);

            // -------- original image
            let o = egui::ColorImage::from_rgb(
                [org.width() as usize, org.height() as usize],
                j.as_slice(),
            );
            let mut original = RetainedImage::from_color_image("lol", o);
            let osize = egui::Vec2::new(org.width() as f32, org.height() as f32);

            ui.label(format!("Epoch: {}  Cost: {}", epoch, cost));
            ui.add(
                egui::Slider::new(&mut self.context.lock().unwrap().learning_rate, 0.00..=5.0)
                    .text("Learning rate"),
            );
            let line = egui::plot::Line::new(
                self.cost
                    .iter()
                    .enumerate()
                    .map(|(i, p)| [i as f64, p.clone()])
                    .collect::<egui::plot::PlotPoints>(),
            );

            Plot::new("cost")
                .view_aspect(4.0)
                .width(size.x * 5.0)
                .show(ui, |p| p.line(line));

            ui.image(img.texture_id(ctx), size * 5.0);
            ui.image(original.texture_id(ctx), osize);
        });
    }
}

fn main() {
    std::env::set_var("RUST_BACKTRACE", "1");

    let (h, w, img) = load_image("./assets/patrick.png");

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
    let mut nn = Model::new(&[2, 12, 6, 1]);
    let mut epoch: u128 = 0;

    nn.set_activation(Activation::Sigmoid);
    let mut learning_rate = 6.0;

    let context = Arc::new(Mutex::new(NContext {
        learning_rate,
        original: image::GrayImage::from_vec(
            h as u32,
            w as u32,
            img.chunks(3)
                .map(|x| ((x.iter().map(|n| *n as u32).sum::<u32>() / 3) as u8))
                .collect::<Vec<u8>>(),
        )
        .unwrap(),
        ..Default::default()
    }));

    let ctx = context.clone();

    let handle = std::thread::spawn(move || {
        loop {
            epoch += 1;
            let (wg, bg) = nn.gradient(&batch.random_chunk(32));

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
                    learning_rate = c.learning_rate;
                }
            }
        }
    });

    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "Snail-NN Demo",
        options,
        Box::new(|cc| {
            Box::new(App {
                context,
                cost: Vec::new(),
            })
        }),
    );
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
