mod game;

use macroquad::prelude::*;
use game::{Game};
use std::time::{SystemTime, UNIX_EPOCH};

const SCORE_AREA_HEIGHT: f32 = 60.0;
const SCORE_TEXT_SIZE: f32 = 40.0;
const CELL_SIZE: i32 = 30;
const GAME_WIDTH: i32 = 27; // cells
const GAME_HEIGHT: i32 = 21; // cells
const BASE_TICK_SPEED: f32 = 0.2; // seconds
const TICK_INCREASE_RATE: f32 = 0.05; // per score
const MIN_TICK_SPEED: f32 = 0.05; // seconds

fn window_conf() -> Conf {
    Conf {
        window_title: "Fixed Resolution (1024x768)".to_owned(),
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
    let mut game_ticks: i32 = 0;

    let mut game = Game::new(GAME_WIDTH, GAME_HEIGHT);

    loop {
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
            draw(&game);
        } else {
            std::process::exit(-1);
        }

        next_frame().await
    }
}

fn draw(game: &Game) {
    let screen_w = screen_width();
    let screen_h = screen_height();


    // score area

    let score_area = Rect::new(0.0, 0.0, screen_w, SCORE_AREA_HEIGHT);

    draw_rectangle(
        score_area.x,
        score_area.y,
        score_area.w,
        score_area.h,
        Color::new(0.1, 0.1, 0.2, 1.0),
    );
    draw_line(score_area.x, score_area.h, screen_w, score_area.h, 2.0, BLACK);

    let score_text = format!("Score: {}", game.score());
    let text_dims = measure_text(&score_text, None, SCORE_TEXT_SIZE as u16, 1.0);
    draw_text(
        &score_text,
        score_area.w / 2.0 - text_dims.width / 2.0,
        score_area.y + score_area.h / 2.0 + text_dims.height / 2.0,
        SCORE_TEXT_SIZE as f32,
        WHITE,
    );


    // game area
    
    let game_area = Rect::new(0.0, score_area.h, screen_w, screen_h - score_area.h);

    let cell_width = game_area.w / GAME_WIDTH as f32;
    let cell_height = game_area.h / GAME_HEIGHT as f32;
    let grid_line_color = Color::new(0.4, 0.4, 0.4, 0.3);

    // vertical grid lines
    for i in 1..GAME_WIDTH {
        let x = game_area.x + i as f32 * cell_width;
        draw_line(
            x,
            game_area.y,
            x,
            game_area.y + game_area.h,
            1.0,
            grid_line_color,
        );
    }

    // horizontal grid lines
    for i in 1..GAME_HEIGHT {
        let y = game_area.y + i as f32 * cell_height;
        draw_line(
            game_area.x,
            y,
            game_area.x + game_area.w,
            y,
            1.0,
            grid_line_color,
        );
    }


    let cell_fill_border: f32 = 2.0;


    // target

    let target_x = game_area.x + game.target().unwrap().0 as f32 * cell_width;
    let target_y = game_area.y + game.target().unwrap().1 as f32 * cell_height;
    draw_rectangle(
        target_x,
        target_y,
        cell_width-1.0,
        cell_height-1.0,
        Color::new(0.5, 0.0, 0.0, 1.0),
    );
    draw_rectangle(
        target_x + cell_fill_border,
        target_y + cell_fill_border,
        (cell_width-1.0) - cell_fill_border*2.0,
        (cell_height-1.0) - cell_fill_border*2.0,
        RED,
    );

    // snake

    for segment in game.snake() {
        let segment_x = game_area.x + segment.0 as f32 * cell_width;
        let segment_y = game_area.y + segment.1 as f32 * cell_height;
        draw_rectangle(
            segment_x,
            segment_y,
            cell_width-1.0,
            cell_height-1.0,
            DARKGREEN,
        );
        draw_rectangle(
            segment_x + cell_fill_border,
            segment_y + cell_fill_border,
            (cell_width-1.0) - cell_fill_border*2.0,
            (cell_height-1.0) - cell_fill_border*2.0,
            GREEN,
        );
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