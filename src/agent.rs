pub mod replaybuffer;

use replaybuffer::ReplayBuffer;
use crate::sequential::{
    tensor::Tensor,
    layer::Layer,
    layer::Dense,
    layer::ReLU,
    loss::MeanSquaredError,
    optimizer::SGD,
    Sequential
};
use rand::prelude::*;
use rand::rng;
use crate::game::Game;

pub struct Agent {
    q_network: Sequential,
    target_network: Sequential,
    replay_buffer: ReplayBuffer,

    batch_size: usize,
    discount_factor: f32,
    epsilon: f32,
    epsilon_decay: f32,
    min_epsilon: f32,

    target_update_counter: usize,
    target_update_frequency: usize,

    action_size: usize
}

impl Agent {
    pub fn new(state_size: usize, action_size: usize) -> Self {
        let q_layers: Vec<Box<dyn Layer>> = vec![
            Box::new(Dense::new(state_size, 256)),
            Box::new(ReLU::new()),
            Box::new(Dense::new(256, action_size))
        ];
        let loss = Box::new(MeanSquaredError);
        let optimizer = Box::new(SGD::new(0.001));

        let q_network = Sequential::new(q_layers, loss, optimizer);

        let target_network = q_network.clone();

        Self {
            q_network,
            target_network,
            replay_buffer: ReplayBuffer::new(10000),
            batch_size: 32,
            discount_factor: 0.99,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            min_epsilon: 0.01,
            target_update_counter: 0,
            target_update_frequency: 25,
            action_size
        }
    }

    pub fn train(&mut self, game: &mut Game, num_episodes: usize) {
        for episode in 0..num_episodes {
            let mut state = game.reset();
            let mut total_reward = 0.0;
            let mut done = false;

            while !done {
                let action = self.get_action_epsilon(&state);
                let (reward, next_state, d) = game.step(action);
                done = d;
                total_reward += reward;
                self.replay_buffer.add((state, action, reward, next_state.clone(), done));
                state = next_state;

                self.train_step();
            }

            println!("Episode: {}, Total Reward: {}, Epsilon: {:.4}", episode + 1, total_reward, self.epsilon);
        }
    }

    pub fn get_action(&mut self, state: &Tensor) -> usize {
        let q_values = self.q_network.predict(state);
        q_values.read().iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0)
    }

    fn get_action_epsilon(&mut self, state: &Tensor) -> usize {
        if rng().random::<f32>() < self.epsilon {
            rng().random_range(0..self.action_size)
        } else {
            self.get_action(state)
        }
    }

    fn train_step(&mut self) {
        if self.replay_buffer.len() < self.batch_size {
            return;
        }

        let (states, actions, rewards, next_states, dones) = self.replay_buffer.sample(self.batch_size).unwrap();
        let next_q_values = self.target_network.predict(&next_states);
        let next_q_data = next_q_values.read();
        let max_next_q: Vec<f32> = next_q_data.chunks(self.action_size).map(|row| row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))).collect();
        let q_targets = self.q_network.predict(&states);

        {
            let mut q_targets_data = q_targets.write();
            let rewards_data = rewards.read();
            let actions_data = actions.read();
            let dones_data = dones.read();
            for i in 0..self.batch_size {
                let action_taken = actions_data[i] as usize;
                let target = rewards_data[i] + self.discount_factor * max_next_q[i] * (1.0 - dones_data[i]);
                q_targets_data[i * self.action_size + action_taken] = target;
            }
        }

        self.q_network.train_on_batch(&states, &q_targets);

        if self.epsilon > self.min_epsilon {
            self.epsilon *= self.epsilon_decay;
        }
        self.target_update_counter += 1;
        if self.target_update_counter % self.target_update_frequency == 0 {
            self.target_network.copy_weights_from(&self.q_network);
        }
    }

    pub fn store(&self, filepath: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(filepath)?;
        bincode::serialize_into(file, &self.q_network)?;
        Ok(())
    }

    pub fn load(filepath: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(filepath)?;
        let q_network: Sequential = bincode::deserialize_from(file)?;
        let target_network = q_network.clone();

        let last_layer = q_network.layers.last().unwrap().as_any().downcast_ref::<Dense>().unwrap();
        let action_size = last_layer.weights.shape[1];

        Ok(Self {
            q_network,
            target_network,
            action_size,
            replay_buffer: ReplayBuffer::new(10000),
            batch_size: 64,
            discount_factor: 0.99,
            epsilon: 0.01,
            epsilon_decay: 0.995,
            min_epsilon: 0.01,
            target_update_counter: 0,
            target_update_frequency: 100,
        })
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
    fn test_agent_store_and_load() {
        let state_size = 4;
        let action_size = 2;

        let mut original_agent = Agent::new(state_size, action_size);
        
        let first_dense_layer = original_agent.q_network.layers[0].as_any_mut().downcast_mut::<Dense>().unwrap();
        first_dense_layer.weights = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], vec![4, 2]);

        let filepath = "test_agent_model.bin";
        
        original_agent.store(filepath).unwrap();

        let loaded_agent = Agent::load(filepath).unwrap();
        
        let original_weights = original_agent.q_network.layers[0].as_any().downcast_ref::<Dense>().unwrap().weights.read();
        let loaded_weights = loaded_agent.q_network.layers[0].as_any().downcast_ref::<Dense>().unwrap().weights.read();

        assert_vec_approx_eq(&original_weights, &loaded_weights);
        
        std::fs::remove_file(filepath).unwrap();
    }
}