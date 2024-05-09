# Following [The Burn Book](https://burn.dev/book/basic-workflow/index.html)

Building by uncomment build block and pass 0 as argument

Also can be validated from [MINIST Viewer](https://observablehq.com/@davidalber/mnist-viewer)

# DataLoader
研究了下Batcher的类型系统是如何工作的，

```rust
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());
```

这一段中，`DataLoaderBuilder` 约束了new只接受`DynBatcher<I, O> + 'static`，
而这里的`I`需要和`MnistDataset::train()`的类型对应上
```rust
    pub fn new<B>(batcher: B) -> Self
    where
        B: DynBatcher<I, O> + 'static,
    {
        Self {
            batcher: Box::new(batcher),
            strategy: None,
            num_threads: None,
            shuffle: None,
        }
    }
```

再看一下什么是`DynBatcher<I, O>`，在类型上做了个Trick，为所有的普通Batcher做了个实现，包了一个Box
```rust
pub trait DynBatcher<I, O>: Send + Batcher<I, O> {
    /// Clone the batcher and returns a new one.
    fn clone_dyn(&self) -> Box<dyn DynBatcher<I, O>>;
}

impl<B, I, O> DynBatcher<I, O> for B
where
    B: Batcher<I, O> + Clone + 'static,
{
    fn clone_dyn(&self) -> Box<dyn DynBatcher<I, O>> {
        Box::new(self.clone())
    }
}

```