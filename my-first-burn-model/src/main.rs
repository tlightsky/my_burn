use std::env;
use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::AutoGraphicsApi;
use burn::data::dataloader::Dataset;
use burn::optim::AdamConfig;
use crate::model::ModelConfig;

mod model;
mod data;
mod training;
mod infer;

fn run(index: usize) {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";

    if index == 0 {
        // crate::training::train::<MyAutodiffBackend>(
        //     artifact_dir,
        //     crate::training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        //     device,
        // );
    } else {
        crate::infer::infer::<MyBackend>(
            artifact_dir,
            device,
            burn::data::dataset::vision::MnistDataset::test()
                .get(index)
                .unwrap(),
        )
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Check if at least one argument is passed (excluding the program name)
    if args.len() > 1 {
        // Attempt to parse the first argument as a number
        match args[1].parse::<i32>() {
            Ok(num) => {
                println!("The first numeric argument is: {}", num);
                run(num as usize);
            },
            Err(_) => println!("The first argument is not a valid number."),
        }
    } else {
        println!("No numeric argument provided.");
    }
}
