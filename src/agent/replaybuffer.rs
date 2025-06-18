use crate::sequential::tensor::Tensor;

use std::collections::VecDeque;

type Experience = (Tensor, usize, f32, Tensor, bool);

pub struct ReplayBuffer {
    pub buffer: VecDeque<Experience>,
    pub capacity: usize
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity
        }
    }

    pub fn add(&mut self, experience: Experience) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    // returns (states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)
    pub fn sample(&self, batch_size: usize) -> Option<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        if self.buffer.len() < batch_size {
            return None;
        }

        let indices = rand::seq::index::sample(&mut rand::rng(), self.buffer.len(), batch_size);

        let mut states = Vec::with_capacity(batch_size);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut next_states = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);

        for index in indices.iter() {
            let (state, action, reward, next_state, done) = self.buffer.get(index).unwrap();
            states.push(state.clone());
            actions.push(*action as f32);
            rewards.push(*reward);
            next_states.push(next_state.clone());
            dones.push(if *done { 1.0 } else { 0.0 });
        }

        // flatten states to convert into tensor
        let state_shape = states[0].shape.clone();
        let state_features = state_shape[1];
        let states_flat: Vec<f32> = states.into_iter().flat_map(|t| t.data.read().unwrap().clone()).collect();
        let next_states_flat: Vec<f32> = next_states.into_iter().flat_map(|t| t.data.read().unwrap().clone()).collect();

        let states_tensor = Tensor::from_vec(states_flat, vec![batch_size, state_features]);
        let actions_tensor = Tensor::from_vec(actions, vec![batch_size, 1]);
        let rewards_tensor = Tensor::from_vec(rewards, vec![batch_size, 1]);
        let next_states_tensor = Tensor::from_vec(next_states_flat, vec![batch_size, state_features]);
        let dones_tensor = Tensor::from_vec(dones, vec![batch_size, 1]);

        Some((states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);

        for i in 0..10 {
            let state = Tensor::from_vec(vec![i as f32; 4], vec![1, 4]);
            let action = i;
            let reward = i as f32 * 10.0;
            let next_state = Tensor::from_vec(vec![(i + 1) as f32; 4], vec![1, 4]);
            let done = i == 9;
            
            buffer.add((state, action, reward, next_state, done));
        }

        assert_eq!(buffer.buffer.len(), 10);

        let sample_result = buffer.sample(5);
        assert!(sample_result.is_some());

        let (states, actions, rewards, next_states, dones) = sample_result.unwrap();

        assert_eq!(states.shape, vec![5, 4]);
        assert_eq!(actions.shape, vec![5, 1]);
        assert_eq!(rewards.shape, vec![5, 1]);
        assert_eq!(next_states.shape, vec![5, 4]);
        assert_eq!(dones.shape, vec![5, 1]);
        
        for i in 10..110 {
            let state = Tensor::from_vec(vec![i as f32; 4], vec![1, 4]);
            buffer.add((state.clone(), i, 0.0, state.clone(), false));
        }
        assert_eq!(buffer.buffer.len(), 100);

        let first_experience_state = buffer.buffer.front().unwrap().0.data.read().unwrap()[0];
        assert_eq!(first_experience_state, 10.0);
    }
}