pub mod tensor;
pub mod layer;
pub mod loss;
pub mod optimizer;

use tensor::Tensor;
use layer::{Layer, Dense, Softmax};
use loss::{Loss};
use optimizer::{Optimizer};

use std::cmp::min;
use rand::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct Sequential {
    pub layers: Vec<Box<dyn Layer>>,
    pub loss: Box<dyn Loss>,
    pub optimizer: Box<dyn Optimizer>
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>, loss: Box<dyn Loss>, optimizer: Box<dyn Optimizer>) -> Self {
        Self {
            layers,
            loss,
            optimizer
        }
    }

    pub fn predict(&mut self, input: &Tensor) -> Tensor {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn fit(&mut self, x_train: &Tensor, y_train: &Tensor, epochs: usize, batch_size: usize) {
        let num_samples = x_train.shape[0];

        for epoch in 0..epochs {
            let mut indices: Vec<usize> = (0..num_samples).collect();
            indices.shuffle(&mut rand::rng());

            let mut total_loss = 0.0;
            let mut num_batches = 0;

            for i in (0..num_samples).step_by(batch_size) {
                let end = min(i + batch_size, num_samples);
                let batch_indices = &indices[i..end];

                let x_batch = x_train.gather_rows(batch_indices);
                let y_batch = y_train.gather_rows(batch_indices);

                let logits = self.predict(&x_batch);

                let mut softmax_layer = Softmax::new();
                let y_pred = softmax_layer.forward(&logits);

                let loss = self.loss.calculate(&y_pred, &y_batch);
                total_loss += loss;
                num_batches += 1;

                let mut d_output = self.loss.gradient(&y_pred, &y_batch);
                for layer in self.layers.iter_mut().rev() {
                    d_output = layer.backward(&d_output);
                }

                self.optimizer.step(&mut self.layers);
            }

            println!("Epoch: {}, Loss: {}", epoch + 1, total_loss / num_batches as f32);
        }
    }

    pub fn train_on_batch(&mut self, x_batch: &Tensor, y_batch: &Tensor) {
        let y_pred = self.predict(x_batch);
        let mut d_output = self.loss.gradient(&y_pred, y_batch);
        for layer in self.layers.iter_mut().rev() {
            d_output = layer.backward(&d_output);
        }
        self.optimizer.step(&mut self.layers);
    }

    pub fn copy_weights_from(&mut self, other: &Self) {
        for (self_layer, other_layer) in self.layers.iter_mut().zip(other.layers.iter()) {
            if let (Some(self_dense), Some(other_dense)) = (self_layer.as_any_mut().downcast_mut::<Dense>(), other_layer.as_any().downcast_ref::<Dense>()) {
                self_dense.weights = other_dense.weights.deep_clone();
                self_dense.biases = other_dense.biases.deep_clone();
            }
        }
    }
}

impl Clone for Sequential {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|layer| layer.clone_box()).collect(),
            loss: self.loss.clone_box(),
            optimizer: self.optimizer.clone_box()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequential::loss::MeanSquaredError;
    use crate::sequential::optimizer::SGD;
    use std::sync::Arc;

    #[test]
    fn test_sequential_fit() {
        let mut dense_layer = Dense::new(2, 1);
        dense_layer.weights = Tensor::from_vec(vec![0.5, -0.5], vec![2, 1]);
        dense_layer.biases = Tensor::from_vec(vec![0.1], vec![1, 1]);
        
        let initial_weights = dense_layer.weights.read().clone();

        let layers: Vec<Box<dyn Layer>> = vec![Box::new(dense_layer)];

        let loss = Box::new(MeanSquaredError);
        let optimizer = Box::new(SGD::new(0.1));

        let mut model = Sequential::new(layers, loss, optimizer);

        let x_train = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let y_train = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]); 

        model.fit(&x_train, &y_train, 1, 1);

        let final_layer = model.layers[0].as_any_mut().downcast_mut::<Dense>().unwrap();
        let final_weights = final_layer.weights.read();

        assert_ne!(initial_weights[0], final_weights[0], "weights did not update");
        assert_ne!(initial_weights[1], final_weights[1], "weights did not update");
    }

    fn assert_vec_approx_eq(a: &[f32], b: &[f32]) {
        let tolerance = 1e-6;
        assert_eq!(a.len(), b.len(), "vectors have different lengths");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tolerance, "mismatch at index {}: {} vs {}", i, x, y);
        }
    }

    #[test]
    fn test_copy_weights_from() {
        let mut dense1 = Dense::new(2, 2);
        dense1.weights = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        dense1.biases = Tensor::from_vec(vec![1.0, 1.0], vec![1, 2]);
        let model1 = Sequential::new(vec![Box::new(dense1)], Box::new(MeanSquaredError), Box::new(SGD::new(0.1)));

        let mut dense2 = Dense::new(2, 2);
        dense2.weights = Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
        dense2.biases = Tensor::from_vec(vec![0.0, 0.0], vec![1, 2]);
        let mut model2 = Sequential::new(vec![Box::new(dense2)], Box::new(MeanSquaredError), Box::new(SGD::new(0.1)));

        model2.copy_weights_from(&model1);

        let dense1_final = model1.layers[0].as_any().downcast_ref::<Dense>().unwrap();
        let dense2_final = model2.layers[0].as_any().downcast_ref::<Dense>().unwrap();

        assert_vec_approx_eq(&dense1_final.weights.read(), &dense2_final.weights.read());
        assert_vec_approx_eq(&dense1_final.biases.read(), &dense2_final.biases.read());
        assert!(!Arc::ptr_eq(&dense1_final.weights.data, &dense2_final.weights.data));
    }

    #[test]
    fn test_train_on_batch() {
        let mut dense_layer = Dense::new(2, 1);
        dense_layer.weights = Tensor::from_vec(vec![0.5, -0.5], vec![2, 1]);
        dense_layer.biases = Tensor::from_vec(vec![0.1], vec![1, 1]);
        let initial_weights = dense_layer.weights.deep_clone();

        let layers: Vec<Box<dyn Layer>> = vec![Box::new(dense_layer)];
        let loss = Box::new(MeanSquaredError);
        let optimizer = Box::new(SGD::new(0.1));
        let mut model = Sequential::new(layers, loss, optimizer);

        let x_batch = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let y_batch = Tensor::from_vec(vec![1.0], vec![1, 1]);

        model.train_on_batch(&x_batch, &y_batch);

        let final_layer = model.layers[0].as_any().downcast_ref::<Dense>().unwrap();
        let final_weights = final_layer.weights.read();

        assert_ne!(*initial_weights.read(), *final_weights, "Weights did not update after a training step");
    }
}