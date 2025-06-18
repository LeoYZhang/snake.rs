use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;
use std::fmt;
use serde::{Serialize, Deserialize, Serializer, Deserializer};

pub struct Tensor {
    pub data: Arc<RwLock<Vec<f32>>>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let data: Vec<f32> = vec![0.0; shape.iter().product()];
        Self {
            data: Arc::new(RwLock::new(data)),
            strides: Tensor::calc_strides(&shape),
            shape
        }
    }

    pub fn random(shape: Vec<usize>) -> Self {
        let normal = Normal::new(0.0, 1.0).unwrap();
        let data: Vec<f32> = (0..shape.iter().product()).map(|_| normal.sample(&mut rand::rng())).collect();

        Self {
            data: Arc::new(RwLock::new(data)),
            strides: Tensor::calc_strides(&shape),
            shape
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
            strides: Tensor::calc_strides(&shape),
            shape
        }
    }

    pub fn read(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.data.read().unwrap()
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.data.write().unwrap()
    }

    pub fn transpose(&self) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.reverse();
        let mut new_strides = self.strides.clone();
        new_strides.reverse();

        Self {
            data: Arc::clone(&self.data),
            shape: new_shape,
            strides: new_strides
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "self must be a 2D tensor.");
        assert_eq!(other.shape.len(), 2, "other must be a 2D tensor.");
        assert_eq!(self.shape[1], other.shape[0], "self columns must equal other rows");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let c = Tensor::zeros(vec![m, n]);

        // const BLOCK_SIZE: usize = 32;

        // get read locks
        let a_data = self.read();
        let b_data = other.read();

        {
            // get write lock for result
            let mut c_data_guard = c.write();
            let c_slice: &mut [f32] = &mut c_data_guard;

            // parallelize
            c_slice.par_chunks_mut(n).enumerate().for_each(|(m_idx, c_row)| {
                for k_idx in 0..k {
                    let a_val = a_data[m_idx * self.strides[0] + k_idx * self.strides[1]];
                    for n_idx in 0..n {
                        let b_val = b_data[k_idx * other.strides[0] + n_idx * other.strides[1]];
                        c_row[n_idx] += a_val * b_val;
                    }
                }
            });
        }

        // {
        //     // get write lock for result
        //     let mut c_data_guard = c.write();
        //     let c_slice: &mut [f32] = &mut c_data_guard;

        //     c_slice.par_chunks_mut(n * BLOCK_SIZE).enumerate().for_each(|(m_block_idx, c_chunk)| {
        //         // block matrices for cache efficiency
        //         let m_block = m_block_idx * BLOCK_SIZE;
        //         for k_block in (0..k).step_by(BLOCK_SIZE) {
        //             for n_block in (0..n).step_by(BLOCK_SIZE) {
        //                 for m_idx in m_block..min(m_block + BLOCK_SIZE, m) {
        //                     for k_idx in k_block..min(k_block + BLOCK_SIZE, k) {
        //                         let a_val = a_data[m_idx * self.strides[0] + k_idx * self.strides[1]];
        //                         for n_idx in n_block..min(n_block + BLOCK_SIZE, n) {
        //                             let b_val = b_data[k_idx * other.strides[0] + n_idx * other.strides[1]];
        //                             c_chunk[(m_idx - m_block) * n + n_idx] += a_val * b_val; 
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     });
        // }

        c
    }

    pub fn sum(&self, axis: usize) -> Tensor {
        assert!(axis < self.shape.len(), "axis out of bounds");
        assert_eq!(self.shape.len(), 2, "sum only works for 2D tensors");

        let data = self.read();
        let m = self.shape[0];
        let n = self.shape[1];

        if axis == 0 {
            let partial_sum = data.par_chunks(n).map(|row_slice| {
                row_slice.to_vec()
            }).reduce(
                || vec![0.0; n],
                |mut acc, row| {
                    for i in 0..n {
                        acc[i] += row[i];
                    }
                    acc
                }
            );

            Tensor::from_vec(partial_sum, vec![1, n])
        } else if axis == 1 {
            let partial_sum = data.par_chunks(n).map(|row_slice| {
                row_slice.iter().sum()
            }).collect();

            Tensor::from_vec(partial_sum, vec![m, 1])
        } else {
            unimplemented!("axis must be 0 or 1");
        }
    }

    pub fn map<F>(&self, f: F) -> Tensor 
    where F: Fn(f32) -> f32 + Sync + Send {
        let input_data = self.read();
        let new_data: Vec<f32> = input_data.par_iter().map(|&x| f(x)).collect();
        Tensor::from_vec(new_data, self.shape.clone())
    }

    // map through self allowing access to second tensor
    pub fn map2<F>(&self, other: &Tensor, f: F) -> Tensor
    where F: Fn(f32, f32) -> f32 + Sync + Send {
        assert_eq!(self.shape, other.shape, "tensors must have the same shape");

        let data1 = self.read();
        let data2 = other.read();
        let new_data: Vec<f32> = data1.par_iter().zip(data2.par_iter()).map(|(&x1, &x2)| f(x1, x2)).collect();
        Tensor::from_vec(new_data, self.shape.clone())
    }

    pub fn gather_rows(&self, indices: &[usize]) -> Tensor {
        assert_eq!(self.shape.len(), 2, "gather_rows only works for 2D tensors");

        let num_cols = self.shape[1];
        let new_num_rows = indices.len();
        let mut new_data = Vec::with_capacity(new_num_rows * num_cols);
        let data = self.read();

        for &row_idx in indices {
            assert!(row_idx < self.shape[0], "row index out of bounds");

            let start = row_idx * self.strides[0];
            let end = start + num_cols;
            new_data.extend_from_slice(&data[start..end]);
        }

        Tensor::from_vec(new_data, vec![new_num_rows, num_cols])
    }

    pub fn deep_clone(&self) -> Tensor {
        let data_clone = self.read().clone();
        Tensor::from_vec(data_clone, self.shape.clone())
    }

    fn calc_strides(shape: &Vec<usize>) -> Vec<usize> {
        let mut strides: Vec<usize> = vec![1; shape.len()];
        for i in (0..strides.len() - 1).rev() {
            strides[i] = strides[i+1] * shape[i+1];
        }
        strides
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            shape: self.shape.clone(),
            strides: self.strides.clone()
        }
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        
        let self_data = self.data.read().unwrap();
        let other_data = other.data.read().unwrap();
        
        *self_data == *other_data
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // When printing, we lock the data and format it nicely.
        f.debug_struct("Tensor")
         .field("shape", &self.shape)
         .field("data", &self.data.read().unwrap())
         .finish()
    }
}

#[derive(Serialize, Deserialize)]
struct SerializableTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: Serializer {
        let s_tensor = SerializableTensor {
            shape: self.shape.clone(),
            data: self.read().clone()
        };
        s_tensor.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: Deserializer<'de> {
        let s_tensor = SerializableTensor::deserialize(deserializer)?;
        Ok(Tensor::from_vec(s_tensor.data, s_tensor.shape))
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
    
    fn reference_matmul(a: &Tensor, b: &Tensor) -> Vec<f32> {
        let m = a.shape[0];
        let k = a.shape[1];
        let n = b.shape[1];
        let mut result = vec![0.0; m * n];
        let a_data = a.read();
        let b_data = b.read();

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    let a_idx = i * a.strides[0] + l * a.strides[1];
                    let b_idx = l * b.strides[0] + j * b.strides[1];
                    sum += a_data[a_idx] * b_data[b_idx];
                }
                result[i * n + j] = sum;
            }
        }
        result
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(*t.read(), vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_random() {
        let t = Tensor::random(vec![100, 2, 3, 4]);
        assert_eq!((*t.read()).len(), 2400);
    }

    #[test]
    fn test_from_vec() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        assert_eq!(*t.read(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_strides() {
        let v = Tensor::calc_strides(&vec![1usize, 2usize, 3usize, 4usize]);
        assert_eq!(v, vec![24, 12, 4, 1]);
    }

    #[test]
    fn test_transpose_shape() {
        let t = Tensor::zeros(vec![2, 3, 4]).transpose();
        assert_eq!(t.shape, vec![4, 3, 2]);
    }

    #[test]
    fn test_transpose_strides() {
        let t = Tensor::zeros(vec![2, 3, 4]).transpose();
        assert_eq!(t.strides, vec![1, 4, 12]);
    }

    #[test]
    fn test_matmul_simple() {
        // A: [[1, 2, 3], [4, 5, 6]]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // B: [[7, 8], [9, 10], [11, 12]]
        let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        
        let c = a.matmul(&b);
        let expected = vec![58.0, 64.0, 139.0, 154.0];

        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(*c.read(), expected);
    }

    #[test]
    fn test_matmul_transpose() {
        // A: [[1, 2], [3, 4]]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // B = A.T: [[1, 3], [2, 4]]
        let b = a.transpose();

        let c = a.matmul(&b);

        let expected = vec![5.0, 11.0, 11.0, 25.0];

        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(*c.read(), expected);
    }
    
    #[test]
    fn test_matmul_against_reference() {
        let a = Tensor::random(vec![64, 32]);
        let b = Tensor::random(vec![32, 70]);

        let result_fast = a.matmul(&b);
        let result_ref = reference_matmul(&a, &b);

        assert_eq!(result_fast.shape, vec![64, 70]);
        assert_vec_approx_eq(&result_fast.read(), &result_ref);
    }

    #[test]
    fn test_matmul_transpose_against_reference() {
        let a = Tensor::random(vec![50, 60]);
        let b = a.transpose();

        let result_fast = a.matmul(&b);
        let result_ref = reference_matmul(&a, &b);
        
        assert_eq!(result_fast.shape, vec![50, 50]);
        assert_vec_approx_eq(&result_fast.read(), &result_ref);
    }

    // #[test]
    // fn test_matmul_speed() {
    //     let a = Tensor::random(vec![784, 256]);
    //     let b = a.transpose();
    //     let result = a.matmul(&b);
    // }

    #[test]
    fn test_sum_axis_0() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = t.sum(0);
        let expected = vec![5.0, 7.0, 9.0];

        assert_eq!(s.shape, vec![1, 3]);
        assert_vec_approx_eq(&s.read(), &expected);
    }

    #[test]
    fn test_sum_axis_1() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = t.sum(1);
        let expected = vec![6.0, 15.0];

        assert_eq!(s.shape, vec![2, 1]);
        assert_vec_approx_eq(&s.read(), &expected);
    }

    #[test]
    fn test_map_simple_multiply() {
        let t = Tensor::from_vec(vec![1.0, 2.0, -3.0], vec![1, 3]);
        let result = t.map(|x| x * 2.0);
        let expected = vec![2.0, 4.0, -6.0];

        assert_eq!(result.shape, vec![1, 3]);
        assert_vec_approx_eq(&result.read(), &expected);
    }
    
    #[test]
    fn test_map_empty_tensor() {
        let t = Tensor::zeros(vec![0, 5]);
        let result = t.map(|x| x + 1.0);
        
        assert_eq!(result.shape, vec![0, 5]);
        assert!(result.read().is_empty());
    }

    #[test]
    fn test_map2_simple_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![1, 3]);
        let result = a.map2(&b, |x, y| x + y);
        let expected = vec![11.0, 22.0, 33.0];

        assert_eq!(result.shape, vec![1, 3]);
        assert_vec_approx_eq(&result.read(), &expected);
    }

    #[test]
    #[should_panic]
    fn test_map2_shape_mismatch() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let b = Tensor::from_vec(vec![10.0, 20.0], vec![1, 2]);
        a.map2(&b, |x, y| x + y);
    }
}