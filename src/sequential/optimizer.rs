use super::layer::{Layer, Dense};
use serde::{Serialize, Deserialize};

#[typetag::serde(tag = "type")]
pub trait Optimizer {
    fn step(&self, layers: &mut [Box<dyn Layer>]);
    fn clone_box(&self) -> Box<dyn Optimizer>;
}


// SGD

#[derive(Serialize, Deserialize, Clone)]
pub struct SGD {
    learning_rate: f32
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate
        }
    }
}

#[typetag::serde]
impl Optimizer for SGD {
    fn step(&self, layers: &mut [Box<dyn Layer>]) {
        for layer in layers {
            if let Some(dense_layer) = layer.as_any_mut().downcast_mut::<Dense>() {
                if let (Some(d_weights), Some(d_biases)) = (&dense_layer.d_weights, &dense_layer.d_biases) {
                    let new_weights = dense_layer.weights.map2(d_weights, |w, dw| {
                        w - self.learning_rate * dw
                    });
                    
                    let new_biases = dense_layer.biases.map2(d_biases, |b, db| {
                        b - self.learning_rate * db
                    });

                    dense_layer.weights = new_weights;
                    dense_layer.biases = new_biases;
                }
            }
        }
    }

    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequential::tensor::Tensor;

    fn assert_vec_approx_eq(a: &[f32], b: &[f32]) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sgd_optimizer_step() {
        let mut dense_layer = Dense::new(2, 2);
        dense_layer.weights = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        dense_layer.biases = Tensor::from_vec(vec![5.0, 6.0], vec![1, 2]);
        dense_layer.d_weights = Some(Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]));
        dense_layer.d_biases = Some(Tensor::from_vec(vec![0.5, 1.5], vec![1, 2]));

        let optimizer = SGD::new(0.1);
        let mut layers: Vec<Box<dyn Layer>> = vec![Box::new(dense_layer)];
        optimizer.step(&mut layers);
        let updated_layer = layers[0].as_any_mut().downcast_mut::<Dense>().unwrap();
        
        // new_w = old_w - lr * d_w
        // new_w[0,0] = 10.0 - 0.1 * 2.0 = 9.8
        // new_w[0,1] = 20.0 - 0.1 * 3.0 = 19.7
        // new_w[1,0] = 30.0 - 0.1 * 4.0 = 29.6
        // new_w[1,1] = 40.0 - 0.1 * 5.0 = 39.5
        let expected_weights = vec![9.8, 19.7, 29.6, 39.5];
        assert_vec_approx_eq(&updated_layer.weights.read(), &expected_weights);

        // new_b = old_b - lr * d_b
        // new_b[0] = 5.0 - 0.1 * 0.5 = 4.95
        // new_b[1] = 6.0 - 0.1 * 1.5 = 5.85
        let expected_biases = vec![4.95, 5.85];
        assert_vec_approx_eq(&updated_layer.biases.read(), &expected_biases);
    }
}