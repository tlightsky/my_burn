use burn::tensor::Tensor;
use burn::backend::Wgpu;

type Backend = Wgpu;

fn main() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);
    println!("{}", tensor_1 + tensor_2);
}
