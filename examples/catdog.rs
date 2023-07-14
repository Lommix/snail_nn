use egui::plot::Plot;
use egui_extras::RetainedImage;
use snail_nn::prelude::*;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

// -------------------------------------
// train resources
const FIRST_IMAGE_PATH: &str = "./assets/3.png";
const SECOND_IMAGE_PATH: &str = "./assets/4.png";
// -------------------------------------
fn main() {
    let context = Arc::new(Mutex::new(CatDogContext {
        learning_rate: 0.01,
        lerp: 0.0,
        ..Default::default()
    }));

    let ctx = context.clone();
    // --- nn thread
    let h = std::thread::spawn(move || {
        let mut epoch = 0;
        let mut model = Model::new(&[3, 15, 9, 1]);
        let mut batch = load_taining_data();
        let mut learning_rate = 0.0;
        let mut lerp = 0.0;

        loop {
            let (w, b) = model.gradient(&batch.next_chunk(32));
            model.learn(w, b, learning_rate);
            epoch += 1;

            if epoch % 100 == 0 {
                let output = imagine(100, 100, lerp, &model);
                let cost = model.cost(&batch);
                let mut c = ctx.lock().unwrap();
                c.epoch = epoch;
                c.out = output;
                c.cost = cost;
                learning_rate = c.learning_rate;
                lerp = c.lerp;
            }
        }
    });

    // --- egui thread
    eframe::run_native(
        "Ai Image Interpolation",
        eframe::NativeOptions::default(),
        Box::new(|_cc| {
            Box::new(CatDogApp {
                context,
                cost: Vec::new(),
                ref_1: RetainedImage::from_image_bytes("one", include_bytes!("../assets/3.png"))
                    .unwrap(),
                ref_2: RetainedImage::from_image_bytes("one", include_bytes!("../assets/4.png"))
                    .unwrap(),
            })
        }),
    );
}
// -------------------------------------
// egui
struct CatDogApp {
    context: Arc<Mutex<CatDogContext>>,
    cost: Vec<f64>,
    ref_1: RetainedImage,
    ref_2: RetainedImage,
}
impl eframe::App for CatDogApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ctx.request_repaint_after(Duration::from_secs_f32(1.0 / 60.0));

            let mut epoch: u128;

            {
                let c = self.context.lock().unwrap();
                self.cost.push(c.cost);
                epoch = c.epoch;
            }

            ui.colored_label(
                egui::Color32::WHITE,
                format!(
                    "Epoch: {} Cost: {}",
                    epoch,
                    self.cost.last().unwrap_or(&0.0)
                ),
            );

            ui.add(
                egui::Slider::new(&mut self.context.lock().unwrap().learning_rate, 0.0..=5.0)
                    .text("Learning Rate"),
            );
            ui.add(
                egui::Slider::new(&mut self.context.lock().unwrap().lerp, 0.0..=1.0)
                    .text("Interpolation"),
            );

            Plot::new("Costgraph")
                .view_aspect(4.0)
                .width(600.0)
                .show(ui, |p| p.line(line_from_vec(&self.cost)));

            {
                let image = gray_to_egui(&self.context.lock().unwrap().out);
                ui.image(image.texture_id(ctx), egui::Vec2::new(600.0, 600.0));
            }

            ui.horizontal(|ui| {
                ui.image(self.ref_1.texture_id(ctx), egui::Vec2::new(100.0, 100.0));
                ui.image(self.ref_2.texture_id(ctx), egui::Vec2::new(100.0, 100.0));
            });
        });
    }
}
// -------------------------------------
// context
#[derive(Debug, Clone, Default)]
struct CatDogContext {
    learning_rate: f64,
    lerp: f64,
    cost: f64,
    epoch: u128,
    out: image::GrayImage,
}
unsafe impl Sync for CatDogContext {}
unsafe impl Send for CatDogContext {}
// -------------------------------------
// network stuff
fn load_taining_data() -> TrainingBatch {
    let (w_a, h_a, img_a) = load_image(FIRST_IMAGE_PATH);
    let (w_b, h_b, img_b) = load_image(SECOND_IMAGE_PATH);
    assert_eq!(w_a, w_b);
    assert_eq!(h_a, h_b);
    let mut batch = TrainingBatch::empty(3, 1);
    for x in 0..w_a {
        for y in 0..h_a {
            let xf = (x as f64) / (w_a as f64);
            let yf = (y as f64) / (h_a as f64);
            batch.add(&[xf, yf, 0.0], &[img_a[(y * w_a + x) as usize]]);
            batch.add(&[xf, yf, 1.0], &[img_b[(y * w_a + x) as usize]]);
        }
    }
    batch
}
// -------------------------------------
// utility functions
fn load_image(path: &str) -> (u32, u32, Vec<f64>) {
    let image = image::open(path).unwrap().to_rgb8();
    let img_data = image
        .chunks(3)
        .map(|x| (x.iter().map(|n| *n as f64).sum::<f64>() / 3.0) / 255.0)
        .collect::<Vec<f64>>();
    (image.width(), image.height(), img_data)
}
fn vec_to_image(height: u32, width: u32, data: Vec<f64>) -> image::GrayImage {
    image::GrayImage::from_vec(
        height,
        width,
        data.iter().map(|x| (*x * 255.0) as u8).collect(),
    )
    .unwrap()
}
fn imagine(width: u32, height: u32, lerp: f64, nn: &Model) -> image::GrayImage {
    let mut output: Vec<f64> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let xf = x as f64 / width as f64;
            let yf = y as f64 / height as f64;
            let input = MatF64::row_from_slice(&[xf, yf, lerp]);
            let out = nn.forward(&input).last().unwrap()[(0, 0)];
            output.push(out);
        }
    }
    vec_to_image(height, width, output)
}
fn gray_to_egui(img: &image::GrayImage) -> RetainedImage {
    let mut pixels = Vec::new();
    img.pixels().for_each(|p| {
        pixels.push(p.0[0]);
        pixels.push(p.0[0]);
        pixels.push(p.0[0]);
    });
    let color_image = egui::ColorImage::from_rgb(
        [img.width() as usize, img.height() as usize],
        pixels.as_slice(),
    );
    RetainedImage::from_color_image("", color_image)
}
fn line_from_vec(vec: &Vec<f64>) -> egui::plot::Line {
    egui::plot::Line::new(
        vec.iter()
            .enumerate()
            .map(|(i, p)| [i as f64, p.clone()])
            .collect::<egui::plot::PlotPoints>(),
    )
}
