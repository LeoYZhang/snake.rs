use std::collections::{LinkedList, HashSet};
use macroquad::rand;

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
}