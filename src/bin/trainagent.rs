use snakers::agent::Agent;
use snakers::game::Game;

fn main() {
    println!("initializing Snake game and RL Agent...");

    let mut game = Game::new(27, 21);
    let mut agent = Agent::new(7, 3);
    
    let num_episodes = 100;
    println!("starting training for {} episodes...", num_episodes);

    agent.train(&mut game, num_episodes);

    println!("\ntraining finished.");

    let model_path = "input/snake_agent.bin";
    match agent.store(model_path) {
        Ok(_) => println!("successfully saved trained model to {}", model_path),
        Err(e) => eprintln!("error saving model: {}", e),
    }
}