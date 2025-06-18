use super::tensor::Tensor;
use std::any::Any;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

#[typetag::serde(tag = "type")]
pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, d_output: &Tensor) -> Tensor;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn as_any(&self) -> &dyn Any;
    fn clone_box(&self) -> Box<dyn Layer>;
}


// dense layer

#[derive(Serialize, Deserialize, Clone)]
pub struct Dense {
    pub weights: Tensor,
    pub biases: Tensor,
    cached_input: Option<Tensor>, // for back propagation
    pub d_weights: Option<Tensor>,
    pub d_biases: Option<Tensor>
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Tensor::random(vec![input_size, output_size]);
        let biases = Tensor::zeros(vec![1, output_size]);
        Self {
            weights,
            biases,
            cached_input: None,
            d_weights: None,
            d_biases: None
        }
    }
}

#[typetag::serde]
impl Layer for Dense {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.cached_input = Some(input.clone());

        let output = input.matmul(&self.weights);

        // add biases
        {
            let mut output_data = output.write();
            let biases_data = self.biases.read();
            let output_size = output.shape[1];

            output_data.par_chunks_mut(output_size).for_each(|row_chunk| {
                for j in 0..output_size {
                    row_chunk[j] += biases_data[j];
                }
            });
        }

        output
    }

    fn backward(&mut self, d_output: &Tensor) -> Tensor {
        if let Some(cached_input) = &self.cached_input {
            // dL/dW = input.T @ dL/dY
            let d_weights = cached_input.transpose().matmul(d_output);
            self.d_weights = Some(d_weights);

            // dL/db = dL/dY.sum(axis=0)
            let d_biases = d_output.sum(0);
            self.d_biases = Some(d_biases);

            // dL/dX = dL/dY @ weights.T
            d_output.matmul(&self.weights.transpose())
        } else {
            panic!("complete forward pass first.");
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}


// relu layer

#[derive(Serialize, Deserialize, Clone)]
pub struct ReLU {
    cached_input: Option<Tensor>
}

impl ReLU {
    pub fn new() -> Self {
        Self {
            cached_input: None
        }
    }
}

#[typetag::serde]
impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.cached_input = Some(input.clone());
        input.map(|x| x.max(0.0))
    }

    fn backward(&mut self, d_output: &Tensor) -> Tensor {
        if let Some(cached_input) = &self.cached_input {
            cached_input.map2(d_output, |input_val, output_val| {
                if input_val > 0.0 {
                    output_val
                } else {
                    0.0
                }
            })
        } else {
            panic!("complete forward pass first.");
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}


// softmax layer

#[derive(Serialize, Deserialize, Clone)]
pub struct Softmax {
    cached_output: Option<Tensor>
}

impl Softmax {
    pub fn new() -> Self {
        Self {
            cached_output: None
        }
    }
}

#[typetag::serde]
impl Layer for Softmax {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let output = Tensor::zeros(input.shape.clone());
        let input_data = input.read();
        let num_classes = input.shape[1];

        {
            let mut output_data = output.write();
            output_data.par_chunks_mut(num_classes).zip(input_data.par_chunks(num_classes)).for_each(|(output_row, input_row)| {
                let max_val = input_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                let mut sum = 0.0;
                for i in 0..num_classes {
                    let exp_val = (input_row[i] - max_val).exp();
                    output_row[i] = exp_val;
                    sum += exp_val;
                }

                for val in output_row.iter_mut() {
                    *val /= sum;
                }
            });
        }

        self.cached_output = Some(output.clone());
        output
    }

    fn backward(&mut self, d_output: &Tensor) -> Tensor {
        if self.cached_output.is_some() {
            d_output.clone()
        } else {
            panic!("complete forward pass first.");
        }
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec_approx_eq(a: &[f32], b: &[f32]) {
        let tolerance = 1e-6;
        assert_eq!(a.len(), b.len(), "vectors have different lengths");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < tolerance, "mismatch at index {}: {} vs {}", i, x, y);
        }
    }

    #[test]
    fn test_dense_forward() {
        let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let mut layer = Dense::new(2, 2);

        layer.weights = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        layer.biases = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        
        // matmul: [1, 2] @ [[10, 20], [30, 40]] = [70, 100]
        // + bias: [70, 100] + [1, 2] = [71, 102]
        let output = layer.forward(&input);
        let expected = vec![71.0, 102.0];
        
        assert_eq!(output.shape, vec![1, 2]);
        assert_vec_approx_eq(&output.read(), &expected);
    }
    
    #[test]
    fn test_dense_backward() {
        let input = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let d_output = Tensor::from_vec(vec![5.0, 8.0], vec![1, 2]);
        let mut layer = Dense::new(2, 2);
        layer.weights = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);

        layer.forward(&input);
        
        // d_input = d_output @ weights.T
        // d_output: [5, 8] @ [[10, 30], [20, 40]] = [5*10+8*20, 5*30+8*40] = [210, 470]
        let d_input = layer.backward(&d_output);
        let expected_d_input = vec![210.0, 470.0];
        assert_eq!(d_input.shape, vec![1, 2]);
        assert_vec_approx_eq(&d_input.read(), &expected_d_input);

        // d_weights = input.T @ d_output
        // [[1], [2]] @ [[5, 8]] = [[1*5, 1*8], [2*5, 2*8]] = [[5, 8], [10, 16]]
        let expected_d_weights = vec![5.0, 8.0, 10.0, 16.0];
        assert_vec_approx_eq(&layer.d_weights.unwrap().read(), &expected_d_weights);

        // d_biases = d_output.sum(axis=0)
        let expected_d_biases = vec![5.0, 8.0];
        assert_vec_approx_eq(&layer.d_biases.unwrap().read(), &expected_d_biases);
    }

    #[test]
    fn test_relu_forward() {
        let input = Tensor::from_vec(vec![-10.0, -0.5, 0.0, 0.5, 10.0], vec![1, 5]);
        let mut layer = ReLU::new();
        let output = layer.forward(&input);
        let expected = vec![0.0, 0.0, 0.0, 0.5, 10.0];
        assert_vec_approx_eq(&output.read(), &expected);
    }

    #[test]
    fn test_relu_backward() {
        let input = Tensor::from_vec(vec![-10.0, -0.5, 0.0, 0.5, 10.0], vec![1, 5]);
        let d_output = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0], vec![1, 5]);
        let mut layer = ReLU::new();

        layer.forward(&input);
        let d_input = layer.backward(&d_output);
        
        let expected = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        assert_vec_approx_eq(&d_input.read(), &expected);
    }

    #[test]
    fn test_softmax_forward() {
        let input = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![1, 3]);
        let mut layer = Softmax::new();
        let output = layer.forward(&input);
        
        for &val in output.read().iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }

        let sum: f32 = output.read().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // e^0/sum, e^1/sum, e^2/sum
        // sum = e^0 + e^1 + e^2 = 1 + 2.718 + 7.389 = 11.107
        // probs = [1/sum, 2.718/sum, 7.389/sum] = [0.090, 0.244, 0.665]
        let expected = vec![0.09003057, 0.24472847, 0.66524094];
        assert_vec_approx_eq(&output.read(), &expected);
    }
}