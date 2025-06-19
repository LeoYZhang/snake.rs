use macroquad::prelude::*;
use macroquad::ui::{hash, root_ui};
use snakers::game::Game;
use snakers::agent::Agent;
use std::time::{SystemTime, UNIX_EPOCH};

const SCORE_AREA_HEIGHT: f32 = 60.0;
const SCORE_TEXT_SIZE: f32 = 40.0;
const CELL_SIZE: i32 = 30;
const GAME_WIDTH: i32 = 27; // cells
const GAME_HEIGHT: i32 = 21; // cells
const BASE_TICK_SPEED: f32 = 0.2; // seconds
const TICK_INCREASE_RATE: f32 = 0.05; // per score
const AGENT_TICK_SPEED: f32 = 0.05; // seconds
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

    let mut game_state = 0; // 0: not started, 1: human player, 2: AI player

    let mut curr_tick_speed: f32 = BASE_TICK_SPEED; // seconds
    let mut time_accumulator: f32 = 0.0; // seconds
    let mut game_ticks: i32 = 0;

    let mut game = Game::new(GAME_WIDTH, GAME_HEIGHT);
    let mut agent = Agent::load("input/snake_agent.bin").unwrap();

    loop {
        if game_state == 0 {
            draw_menu(&mut game_state);
        } else if game_state == 1 {
            if game.alive() {
                time_accumulator += get_frame_time();
                while time_accumulator >= curr_tick_speed {
                    game_ticks += 1;
                    time_accumulator -= curr_tick_speed;

                    game.tick();

                    // increase tick speed as score increases
                    curr_tick_speed = 1.0 / ((1.0 / BASE_TICK_SPEED) * (1.0 + (game.score() as f32) * TICK_INCREASE_RATE));
                    curr_tick_speed = curr_tick_speed.max(MIN_TICK_SPEED);
                }

                handle_key_inputs(&mut game);
                game.draw(SCORE_AREA_HEIGHT, SCORE_TEXT_SIZE);
            } else {
                game.reset();
                game_state = 0;
            }
        } else {
            if game.alive() {
                time_accumulator += get_frame_time();
                while time_accumulator >= AGENT_TICK_SPEED {
                    game_ticks += 1;
                    time_accumulator -= AGENT_TICK_SPEED;

                    let action = agent.get_action(&game.get_state());
                    match action {
                        1 => game.turn_left(),
                        2 => game.turn_right(),
                        _ => ()
                    };
                    game.tick();
                }

                game.draw(SCORE_AREA_HEIGHT, SCORE_TEXT_SIZE);
            } else {
                game.reset();
                game_state = 0;
            }
        }

        next_frame().await
    }
}

struct Button {
    rect: Rect,
    text: &'static str,
}

impl Button {
    fn new(x: f32, y: f32, w: f32, h: f32, text: &'static str) -> Self {
        Self {
            rect: Rect::new(x, y, w, h),
            text,
        }
    }

    fn draw_and_check_click(&self) -> bool {
        let mouse_pos = mouse_position();
        let mouse_over = self.rect.contains(vec2(mouse_pos.0, mouse_pos.1));

        let color = if mouse_over {
            Color::from_rgba(100, 100, 120, 255) // Darker color on hover
        } else {
            Color::from_rgba(80, 80, 100, 255) // Default color
        };

        draw_rectangle(self.rect.x, self.rect.y, self.rect.w, self.rect.h, color);
        let text_dims = measure_text(self.text, None, 30, 1.0);
        let text_x = self.rect.x + (self.rect.w - text_dims.width) / 2.0;
        let text_y = self.rect.y + (self.rect.h - text_dims.height) / 2.0 + text_dims.offset_y;
        draw_text(self.text, text_x, text_y, 30.0, WHITE);
        mouse_over && is_mouse_button_pressed(MouseButton::Left)
    }
}

pub fn draw_menu(game_state: &mut i32) {
    let center_x = screen_width() / 2.0;
    let center_y = screen_height() / 2.0;
    let button_width = 250.0;
    let button_height = 60.0;
    let spacing = 20.0;

    let human_button = Button::new(
        center_x - button_width / 2.0,
        center_y - button_height - spacing / 2.0,
        button_width,
        button_height,
        "Human Player"
    );

    let agent_button = Button::new(
        center_x - button_width / 2.0,
        center_y + spacing / 2.0,
        button_width,
        button_height,
        "RL Agent"
    );
    
    let title = "Select Player";
    let title_dims = measure_text(title, None, 50, 1.0);
    draw_text(title, center_x - title_dims.width / 2.0, center_y - 100.0, 50.0, BLACK);

    if human_button.draw_and_check_click() {
        *game_state = 1;
    }

    if agent_button.draw_and_check_click() {
        *game_state = 2;
    }
}

fn handle_key_inputs(game: &mut Game) {
    // use WASD or arrow keys for input
    if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W) {
        game.set_direction((0, -1));
    } else if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S) {
        game.set_direction((0, 1));
    } else if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
        game.set_direction((1, 0));
    } else if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
        game.set_direction((-1, 0));
    }
}