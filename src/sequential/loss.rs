use super::tensor::Tensor;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

#[typetag::serde(tag = "type")]
pub trait Loss {
    fn calculate(&self, y_pred: &Tensor, y_true: &Tensor) -> f32;
    fn gradient(&self, y_pred: &Tensor, y_true: &Tensor) -> Tensor;
    fn clone_box(&self) -> Box<dyn Loss>;
}


// cross entropy loss

#[derive(Serialize, Deserialize, Clone)]
pub struct CategoricalCrossEntropy;

impl CategoricalCrossEntropy {
    pub fn new() -> Self {
        Self
    }
}

#[typetag::serde]
impl Loss for CategoricalCrossEntropy {
    fn calculate(&self, y_pred: &Tensor, y_true: &Tensor) -> f32 {
        assert_eq!(y_pred.shape, y_true.shape, "prediction and true labels must have the same shape.");

        // L = -sum(y_true * log(y_pred))
        let batch_size = y_pred.shape[0];
        if batch_size == 0 {
            return 0.0;
        }
        let num_classes = y_pred.shape[1];

        let y_pred_data = y_pred.read();
        let y_true_data = y_true.read();

        let total_loss: f32 = y_pred_data
            .par_chunks(num_classes)
            .zip(y_true_data.par_chunks(num_classes))
            .map(|(pred_row, true_row)| {
                let mut sample_loss = 0.0;
                const EPSILON: f32 = 1e-9;
                for j in 0..num_classes {
                    if true_row[j] == 1.0 {
                        sample_loss = -(pred_row[j] + EPSILON).ln();
                        break;
                    }
                }
                sample_loss
            }).sum();

        total_loss / batch_size as f32
    }

    fn gradient(&self, y_pred: &Tensor, y_true: &Tensor) -> Tensor {
        y_pred.map2(y_true, |pred_x, true_x| {
            pred_x - true_x
        })
    }

    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(self.clone())
    }
}


// mean squared error

#[derive(Serialize, Deserialize, Clone)]
pub struct MeanSquaredError;

#[typetag::serde]
impl Loss for MeanSquaredError {
    fn calculate(&self, y_pred: &Tensor, y_true: &Tensor) -> f32 {
        let diff = y_pred.map2(y_true, |pred_x, pred_y| pred_x - pred_y);
        let squared_errors = diff.map(|x| x * x);
        squared_errors.read().iter().sum::<f32>() / y_pred.shape[0] as f32
    }

    fn gradient(&self, y_pred: &Tensor, y_true: &Tensor) -> Tensor {
        y_pred.map2(y_true, |pred_x, pred_y| pred_x - pred_y).map(|x| 2.0 * x / y_pred.shape[0] as f32)
    }

    fn clone_box(&self) -> Box<dyn Loss> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx_eq(a: f32, b: f32) {
        let tolerance = 1e-6;
        assert!((a - b).abs() < tolerance, "mismatch: {} vs {}", a, b);
    }
    
    fn assert_vec_approx_eq(a: &[f32], b: &[f32]) {
        let tolerance = 1e-6;
        assert_eq!(a.len(), b.len(), "vectors have different lengths");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tolerance, "mismatch at index {}: {} vs {}", i, x, y);
        }
    }

    #[test]
    fn test_cce_loss_calculation() {
        let loss_fn = CategoricalCrossEntropy::new();

        let y_pred = Tensor::from_vec(vec![0.1, 0.8, 0.1, 0.7, 0.2, 0.1], vec![2, 3]);
        let y_true = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0], vec![2, 3]);

        let loss = loss_fn.calculate(&y_pred, &y_true);
        let expected_loss = 0.28990924;
        
        assert_approx_eq(loss, expected_loss);
    }

    #[test]
    fn test_cce_gradient_calculation() {
        let loss_fn = CategoricalCrossEntropy::new();
        let y_pred = Tensor::from_vec(vec![0.1, 0.8, 0.1], vec![1, 3]);
        let y_true = Tensor::from_vec(vec![0.0, 1.0, 0.0], vec![1, 3]);

        let gradient = loss_fn.gradient(&y_pred, &y_true);
        
        let expected_gradient = vec![0.1, -0.2, 0.1];

        assert_eq!(gradient.shape, vec![1, 3]);
        assert_vec_approx_eq(&gradient.read(), &expected_gradient);
    }
}