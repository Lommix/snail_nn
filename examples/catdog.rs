use egui_extras::RetainedImage;
use snail_nn::prelude::*;
use std::sync::{Arc, Mutex};

// -------------------------------------
// init
const FIRST_IMAGE_PATH: &str = "./assets/patrick.png";
const SECOND_IMAGE_PATH: &str = "./assets/marc.png";
// -------------------------------------
fn main() {
    let context = Arc::new(Mutex::new(CatDogContext {
        learning_rate: 0.1,
        lerp: 0.0,
        ..Default::default()
    }));

    // --- nn thread
    let h = std::thread::spawn(move || {
        let mut epoch = 0;
        let mut model = Model::new(&[3, 16, 8, 1]);
        loop {
            epoch += 1;
            if epoch % 100 == 0 {}
        }
    });

    // --- egui thread
    eframe::run_native(
        "Ai Image Interpolation",
        eframe::NativeOptions::default(),
        Box::new(|_cc| Box::new(CatDogApp { context: context })),
    );
}
// -------------------------------------
// egui
struct CatDogApp {
    context: Arc<Mutex<CatDogContext>>,
}
impl eframe::App for CatDogApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                egui::Slider::new(&mut self.context.lock().unwrap().learning_rate, 0.0..=5.0)
                    .text("Learning Rate"),
            );
            ui.add(
                egui::Slider::new(&mut self.context.lock().unwrap().lerp, 0.0..=1.0)
                    .text("Interpolation"),
            );
        });
    }
}
// -------------------------------------
// context
#[derive(Default, Debug, Clone)]
struct CatDogContext {
    learning_rate: f64,
    lerp: f64,
    cost: f64,
    epoch: u128,
    in_one: image::GrayImage,
    in_two: image::GrayImage,
    out: image::GrayImage,
}
unsafe impl Sync for CatDogContext {}
unsafe impl Send for CatDogContext {}
// -------------------------------------
// network stuff
fn init_networt() {}
fn load_taining_data() -> TrainingBatch {
    let img_one = load_image(FIRST_IMAGE_PATH);
    let img_two = load_image(SECOND_IMAGE_PATH);

    TrainingBatch::new(MatF64::empty(0, 1), MatF64::empty(0, 1))
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
    image::GrayImage::from_vec(height, width, data.iter().map(|x| *x as u8).collect()).unwrap()
}
