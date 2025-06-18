use std::collections::{LinkedList, HashSet};
use macroquad::rand;
use macroquad::prelude::*;
// use rand::prelude::*;
use crate::sequential::tensor::Tensor;

pub type Vec2 = (i32, i32);

pub struct Game {
    target: Option<Vec2>, // food location
    snake: LinkedList<Vec2>,
    score: i32,
    alive: bool,

    width: i32,
    height: i32,
    attempted_direction: Option<Vec2>, // unit vector (N, E, S, W)
    move_direction: Vec2, // unit vector (N, E, S, W)
    segments_map: HashSet<Vec2> // for O(1) collision querying
}

impl Game {
    pub fn new(width: i32, height: i32) -> Self {
        let head = (width / 2, height / 2);

        let mut snake = LinkedList::new();
        snake.push_back(head);

        let mut segments_map = HashSet::new();
        segments_map.insert(head);

        let mut instance = Self {
            target: None,
            snake,
            score: 0,
            alive: true,
            width,
            height,
            attempted_direction: None,
            move_direction: (1, 0),
            segments_map
        };
        instance.generate_target();

        instance
    }

    pub fn target(&self) -> Option<Vec2> {self.target}
    pub fn snake(&self) -> &LinkedList<Vec2> {&self.snake}
    pub fn score(&self) -> i32 {self.score}
    pub fn alive(&self) -> bool {self.alive}

    pub fn tick(&mut self) {
        if self.attempted_direction.is_some() {
            // dot product to check for valid direction change
            if self.attempted_direction.unwrap().0*self.move_direction.0 + self.attempted_direction.unwrap().1*self.move_direction.1 == 0 {
                self.move_direction = self.attempted_direction.unwrap();
            }
        }
        self.attempted_direction = None;


        let new_head = (
            self.snake.front().unwrap().0 + self.move_direction.0, 
            self.snake.front().unwrap().1 + self.move_direction.1
        );

        if self.check_self_collision(new_head) || self.check_wall_collision(new_head) {
            self.alive = false;
            return;
        }

        self.snake.push_front(new_head);
        self.segments_map.insert(new_head);

        if self.check_target_collision(new_head) {
            self.score += 1;
            self.generate_target();
        } else {
            let old_tail = self.snake.pop_back();
            self.segments_map.remove(&old_tail.unwrap());
        }
    }

    pub fn set_direction(&mut self, new_direction: Vec2) {
        self.attempted_direction = Some(new_direction);
    }

     pub fn turn_left(&mut self) {
        let (_, _, dir_l, _) = self.get_relative_direction_vectors();
        self.set_direction(dir_l);
    }

    pub fn turn_right(&mut self) {
        let (_, _, _, dir_r) = self.get_relative_direction_vectors();
        self.set_direction(dir_r);
    }

    // generate random point, then search in a spiral for unoccupied cells
    fn generate_target(&mut self) {
        let start_x = rand::gen_range(0, self.width);
        let start_y = rand::gen_range(0, self.height);

        if !self.segments_map.contains(&(start_x, start_y)) {
            self.target = Some((start_x, start_y));
            return;
        }

        for radius in 1..self.width.max(self.height) {
            let t_edge_y = start_y - radius;
            let b_edge_y = start_y + radius;
            let r_edge_x = start_x + radius;
            let l_edge_x = start_x - radius;

            for i in -radius..=radius {
                if !self.check_wall_collision((start_x+i, t_edge_y)) && !self.segments_map.contains(&(start_x+i, t_edge_y)) {
                    self.target = Some((start_x+i, t_edge_y));
                    return;
                }
                if !self.check_wall_collision((start_x+i, b_edge_y)) && !self.segments_map.contains(&(start_x+i, b_edge_y)) {
                    self.target = Some((start_x+i, b_edge_y));
                    return;
                }
                if !self.check_wall_collision((r_edge_x, start_y+i)) && !self.segments_map.contains(&(r_edge_x, start_y+i)) {
                    self.target = Some((r_edge_x, start_y+i));
                    return;
                }
                if !self.check_wall_collision((l_edge_x, start_y+i)) && !self.segments_map.contains(&(l_edge_x, start_y+i)) {
                    self.target = Some((l_edge_x, start_y+i));
                    return;
                }
            }
        }

        // map is filled
        self.target = None;
    }

    fn check_target_collision(&self, cell: Vec2) -> bool {
        self.target.is_some() && cell.0 == self.target.unwrap().0 && cell.1 == self.target.unwrap().1
    }

    fn check_self_collision(&self, cell: Vec2) -> bool {
        self.segments_map.contains(&cell) && 
        !(cell.0 == self.snake.back().unwrap().0 && cell.1 == self.snake.back().unwrap().1)
    }

    fn check_wall_collision(&self, cell: Vec2) -> bool {
        cell.0 < 0 || cell.0 >= self.width || cell.1 < 0 || cell.1 >= self.height
    }

    pub fn draw(&self, score_area_height: f32, score_text_size: f32) {
        let screen_w = screen_width();
        let screen_h = screen_height();

        // score area

        let score_area = Rect::new(0.0, 0.0, screen_w, score_area_height);

        draw_rectangle(
            score_area.x,
            score_area.y,
            score_area.w,
            score_area.h,
            Color::new(0.1, 0.1, 0.2, 1.0),
        );
        draw_line(score_area.x, score_area.h, screen_w, score_area.h, 2.0, BLACK);

        let score_text = format!("Score: {}", self.score());
        let text_dims = measure_text(&score_text, None, score_text_size as u16, 1.0);
        draw_text(
            &score_text,
            score_area.w / 2.0 - text_dims.width / 2.0,
            score_area.y + score_area.h / 2.0 + text_dims.height / 2.0,
            score_text_size as f32,
            WHITE,
        );


        // game area
        
        let game_area = Rect::new(0.0, score_area.h, screen_w, screen_h - score_area.h);

        let cell_width = game_area.w / self.width as f32;
        let cell_height = game_area.h / self.height as f32;
        let grid_line_color = Color::new(0.4, 0.4, 0.4, 0.3);

        // vertical grid lines
        for i in 1..self.width {
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
        for i in 1..self.height {
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

        if self.target.is_some() {
            let target_x = game_area.x + self.target.unwrap().0 as f32 * cell_width;
            let target_y = game_area.y + self.target.unwrap().1 as f32 * cell_height;
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
        }

        // snake

        for segment in self.snake() {
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


    // RL training

    // returns (reward, next_state, done)
    // actions
    // 0: do nothing, 1: turn left, 2: turn right
    pub fn step(&mut self, action: usize) -> (f32, Tensor, bool) {
        let dist_before_move = self.distance_to_food();
        let score_before_move = self.score;

        let (dir_f, _, dir_l, dir_r) = self.get_relative_direction_vectors();

        let new_direction = match action {
            0 => dir_f,
            1 => dir_l,
            2 => dir_r,
            _ => panic!("action must be 0, 1, or 2."),
        };
        
        self.set_direction(new_direction);
        self.tick();

        let food_eaten = !(self.score == score_before_move);
        let dist_after_move = self.distance_to_food();

        let reward = if !self.alive {
            -20.0
        } else if food_eaten {
            10.0
        } else if dist_after_move < dist_before_move {
            1.0
        } else {
            -1.0
        };

        (reward, self.get_state(), !self.alive)
    }

    // returns state
    pub fn reset(&mut self) -> Tensor {
        let head = (self.width / 2, self.height / 2);

        let mut snake = LinkedList::new();
        snake.push_back(head);

        let mut segments_map = HashSet::new();
        segments_map.insert(head);

        self.snake = snake;
        self.score = 0;
        self.alive = true;
        self.attempted_direction = None;
        self.move_direction = (1, 0);
        self.segments_map = segments_map;
        self.generate_target();

        self.get_state()
    }

    // state: [danger f,l,r; food f,b,l,r]
    pub fn get_state(&self) -> Tensor {
        let head = self.snake.front().unwrap();
        let (dir_f, _, dir_l, dir_r) = self.get_relative_direction_vectors();

        let danger_forward = {
            let p = (head.0 + dir_f.0, head.1 + dir_f.1);
            self.check_wall_collision(p) || self.check_self_collision(p)
        };
        let danger_left = {
            let p = (head.0 + dir_l.0, head.1 + dir_l.1);
            self.check_wall_collision(p) || self.check_self_collision(p)
        };
        let danger_right = {
            let p = (head.0 + dir_r.0, head.1 + dir_r.1);
            self.check_wall_collision(p) || self.check_self_collision(p)
        };

        let mut food_forward = false;
        let mut food_backward = false;
        let mut food_left = false;
        let mut food_right = false;

        if let Some(food_pos) = self.target {
            let food_vec = (food_pos.0 - head.0, food_pos.1 - head.1);

            // use dot product to figure out relative direction

            let dot_fb = food_vec.0 * dir_f.0 + food_vec.1 * dir_f.1;
            if dot_fb > 0 {
                food_forward = true; 
            } else if dot_fb < 0 {
                food_backward = true;
            }

            let dot_lr = food_vec.0 * dir_l.0 + food_vec.1 * dir_l.1;
            if dot_lr > 0 {
                food_left = true;
            } else if dot_lr < 0 {
                food_right = true;
            }
        }

        let state_vec = vec![
            danger_forward as i32 as f32,
            danger_left as i32 as f32,
            danger_right as i32 as f32,
            food_forward as i32 as f32,
            food_backward as i32 as f32,
            food_left as i32 as f32,
            food_right as i32 as f32,
        ];
        
        Tensor::from_vec(state_vec, vec![1, 7])
    }

    // returns relative direciton: [forward, backward, left, right]
    fn get_relative_direction_vectors(&self) -> (Vec2, Vec2, Vec2, Vec2) {
         match self.move_direction {
            (1, 0) => ((1, 0), (-1, 0), (0, -1), (0, 1)),
            (-1, 0) => ((-1, 0), (1, 0), (0, 1), (0, -1)),
            (0, -1) => ((0, -1), (0, 1), (-1, 0), (1, 0)),
            (0, 1) => ((0, 1), (0, -1), (1, 0), (-1, 0)),
            _ => ((0, 0), (0, 0), (0, 0), (0, 0)),
        }
    }

    fn distance_to_food(&self) -> f32 {
        if self.target.is_none() {
            panic!("target is not set.");
        }

        let target = self.target.unwrap();
        let head = self.snake.front().unwrap();

        (((target.0 - head.0).pow(2) + (target.1 - head.1).pow(2)) as f32).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_relative_direction_vectors() {
        let mut game = Game::new(10, 10);
        
        game.move_direction = (1, 0);
        assert_eq!(game.get_relative_direction_vectors(), ((1, 0), (-1, 0), (0, -1), (0, 1)));

        game.move_direction = (0, 1);
        assert_eq!(game.get_relative_direction_vectors(), ((0, 1), (0, -1), (1, 0), (-1, 0)));
    }
    
    #[test]
    fn test_get_state_simple() {
        let mut game = Game::new(10, 10);
        game.snake = LinkedList::from_iter(vec![(5, 5)]);
        game.move_direction = (0, -1);
        game.target = Some((5, 0));
        
        let expected_state = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let state_tensor = game.get_state();

        assert_eq!(state_tensor, Tensor::from_vec(expected_state, vec![1, 7]));
    }
    
    #[test]
    fn test_get_state_with_danger() {
        let mut game = Game::new(10, 10);
        game.snake = LinkedList::from_iter(vec![(1, 1), (0, 1)]);
        game.move_direction = (1, 0);
        game.target = Some((5, 5));
        
        let expected_state = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let state_tensor = game.get_state();

        assert_eq!(state_tensor, Tensor::from_vec(expected_state, vec![1, 7]));
    }

    #[test]
    fn test_step_ate_food_reward() {
        let mut game = Game::new(10, 10);
        game.snake = LinkedList::from_iter(vec![(5, 5)]);
        game.target = Some((6, 5));
        game.move_direction = (1, 0);

        let (reward, _, done) = game.step(0);

        assert_eq!(reward, 10.0);
        assert_eq!(done, false);
        assert_eq!(game.score, 1);
    }
    
    #[test]
    fn test_step_death_penalty() {
        let mut game = Game::new(3, 3);
        game.snake = LinkedList::from_iter(vec![(2, 1)]);
        game.move_direction = (1, 0);
        game.target = Some((0,0));
        
        // go into wall
        let (reward, _, done) = game.step(0);
        
        assert_eq!(reward, -20.0);
        assert_eq!(done, true);
    }
}
