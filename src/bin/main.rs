use macroquad::prelude::*;
use snakers::game::Game;
use snakers::agent::Agent;
use std::time::{SystemTime, UNIX_EPOCH};

const SCORE_AREA_HEIGHT: f32 = 60.0;
const SCORE_TEXT_SIZE: f32 = 40.0;
const CELL_SIZE: i32 = 30;
const GAME_WIDTH: i32 = 27; // cells
const GAME_HEIGHT: i32 = 21; // cells
const BASE_TICK_SPEED: f32 = 0.05; //0.2; // seconds
const TICK_INCREASE_RATE: f32 = 0.05; // per score
const MIN_TICK_SPEED: f32 = 0.05; // seconds

fn window_conf() -> Conf {
    Conf {
        window_title: "snake.rs".to_owned(),
        window_width: GAME_WIDTH*CELL_SIZE,
        window_height: (SCORE_AREA_HEIGHT as i32) + GAME_HEIGHT*CELL_SIZE,
        window_resizable: false,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();
    rand::srand(seed);

    let mut curr_tick_speed: f32 = BASE_TICK_SPEED; // seconds
    let mut time_accumulator: f32 = 0.0; // seconds
    // // let mut game_ticks: i32 = 0;

    let mut game = Game::new(GAME_WIDTH, GAME_HEIGHT);
    let mut agent = Agent::load("input/snake_agent.bin").unwrap();
    // let delay = time::Duration::from_millis(100);

    loop {
        if game.alive() {
            time_accumulator += get_frame_time();
            while time_accumulator >= curr_tick_speed {
                // game_ticks += 1;
                time_accumulator -= curr_tick_speed;

                // game.tick();
                let action = agent.get_action(&game.get_state());
                match action {
                    1 => game.turn_left(),
                    2 => game.turn_right(),
                    _ => ()
                };
                game.tick();

                // increase tick speed as score increases
                curr_tick_speed = 1.0 / ((1.0 / BASE_TICK_SPEED) * (1.0 + (game.score() as f32) * TICK_INCREASE_RATE));
                curr_tick_speed = curr_tick_speed.max(MIN_TICK_SPEED);
            }

            // handle_key_inputs(&mut game);
            game.draw(SCORE_AREA_HEIGHT, SCORE_TEXT_SIZE);
        } else {
            std::process::exit(-1);
        }

        // agent.train(&mut game, 1);
        // game.draw(SCORE_AREA_HEIGHT, SCORE_TEXT_SIZE);
        // thread::sleep(delay);

        next_frame().await
    }
}

// fn handle_key_inputs(game: &mut Game) {
//     // use WASD or arrow keys for input
//     if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W) {
//         game.set_direction((0, -1));
//     } else if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S) {
//         game.set_direction((0, 1));
//     } else if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
//         game.set_direction((1, 0));
//     } else if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
//         game.set_direction((-1, 0));
//     }
// }