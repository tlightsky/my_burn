use burn::backend::{Autodiff, Wgpu};
use burn::backend::wgpu::AutoGraphicsApi;
use burn::optim::AdamConfig;
use crate::model::ModelConfig;

mod model;
mod data;
mod training;


fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    crate::training::train::<MyAutodiffBackend>(
        "/tmp/guide",
        crate::training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}
