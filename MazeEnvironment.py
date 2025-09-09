import numpy as np
import cv2
from operator import itemgetter
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecMonitor


class MazeEnv(gym.Env):
    def __init__(self, maze_imgs, maze_arrs, seed=None, rew_weights=None, dfs_dist=False, end_on_wall=True):
        self.NORTH, self.EAST, self.WEST, self.SOUTH = [0b1000, 0b0100, 0b0010, 0b0001]
        
        if rew_weights is None:
            self.rew_weights = {
                'base': -0.01,
                'goal': 1.00,
                'wall': -0.1,
                'new_cell': 0.01,
                'distance': -0.0001
            }
        else: self.rew_weights = rew_weights
        
        self.seed(seed)
        
        self.maze_imgs = maze_imgs
        self.maze_arrs = maze_arrs
        
        self.num_mazes = len(self.maze_arrs)
        assert self.num_mazes == len(self.maze_imgs), "Maze image and array datasets don't match in length."
        
        self.maze_shape = self.maze_arrs.shape[1:3]
        self.img_shape = self.maze_imgs.shape[1:]

        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(*self.img_shape,), dtype=np.uint8),
            'telemetry': spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
        }) 
        
        self.full_state = {
            'pos': None,
            'goal': None,
            'visited_cells': None,
            'wall_north': False,
            'wall_east': False,
            'wall_west': False,
            'wall_south': False,
            'sample_idx': None,
            'maze_arr': None,
            'maze_img': None,
            'n_steps': 0,
            'step_rewards': {
                'base' : None,
                'goal': None,
                'wall': None,
                'new_cell': None,
                'distance': None,
            }
        }
        
        self.observables = ['maze_img', 'wall_north', 'wall_east', 'wall_west', 'wall_south']
        self.dirns = np.array([[-1,0], [0,1], [0,-1], [1,0]]) # NEWS
        self.dirn_masks = np.array([self.NORTH, self.EAST, self.WEST, self.SOUTH])

        self.max_steps = 100
        self.is_dfs_dist = dfs_dist
        self.end_on_wall = end_on_wall

    def seed(self, seed=None):
        np.random.random(seed)
    
    def get_cell_idx(self, pos):
        return pos[0]*self.maze_shape[0] + pos[1]

    def dfs(self, start_pos, goal_pos, depth_matrix, direction = None):
        walls = self.full_state['maze_arr'][start_pos[0], start_pos[1]]
             
        if (start_pos == goal_pos).all():
            return depth_matrix

        dirns = np.random.permutation(4)
        
        for dirn in dirns:
            if not walls & self.dirn_masks[dirn]:
                next_pos = start_pos + self.dirns[dirn]
                depth_matrix[next_pos[0], next_pos[1]] = min(depth_matrix[start_pos[0], start_pos[1]] + 1,
                                                         depth_matrix[next_pos[0], next_pos[1]])
                return self.dfs(next_pos, goal_pos, depth_matrix, direction = dirn)
            
        return depth_matrix
    
    def get_dfs_dist(self, start_pos, goal_pos):
        depth_matrix = np.zeros(self.maze_shape, dtype = np.int32) + 500
        depth_matrix[start_pos[0], start_pos[1]] = 0
        depth_matrix = self.dfs(start_pos, goal_pos, depth_matrix)
        return depth_matrix[goal_pos[0], goal_pos[1]]

    def get_manhattan_dist(self, pos1, pos2):
        return float(np.abs(pos1 - pos2).sum())
    
    
    def reset(self, *, seed=None, options= None):
        super().reset(seed=seed)

        goal_dist = None
        if options is not None:
            goal_dist = options.get('goal_dist', None)
        
        self.full_state['n_steps'] = 0
        
        sample_idx = np.random.randint(self.num_mazes)
        self.full_state['sample_idx'] = sample_idx
        
        maze_arr = self.maze_arrs[sample_idx]
        maze_img = self.maze_imgs[sample_idx]
        
        if goal_dist is None:
            start_pos, goal_pos = np.random.randint([0,0], [self.maze_shape[0], self.maze_shape[1]], (2,2))
            while (goal_pos == start_pos).all():
                goal_pos = np.random.randint([0,0],[self.maze_shape[0], self.maze_shape[1]],(2,))
        else:
            start_pos = np.random.randint([0,0], [self.maze_shape[0], self.maze_shape[1]], (2,))
            goal_pos_offset = np.random.randint([-goal_dist,-goal_dist],[goal_dist, goal_dist], (2,))
            while (goal_pos_offset == np.zeros(2)).all():
                 goal_pos_offset = np.random.randint([-goal_dist,-goal_dist],[goal_dist, goal_dist], (2,))
            goal_pos = np.clip((start_pos + goal_pos_offset),0, self.maze_shape[0])
            
        self.full_state['pos'] = start_pos
        self.full_state['visited_cells'] = {self.get_cell_idx(start_pos)}
        self.full_state['goal'] = goal_pos

        self.full_state['maze_img_original'] = maze_img
        maze_img = self._draw_player_goal(maze_img, start_pos, goal_pos)
        self.full_state['maze_arr'] = maze_arr
        self.full_state['maze_img'] = maze_img

        self.full_state['wall_north'] = bool(maze_arr[start_pos[0], start_pos[1]] & self.NORTH)
        self.full_state['wall_east'] = bool(maze_arr[start_pos[0], start_pos[1]] & self.EAST)
        self.full_state['wall_west'] = bool(maze_arr[start_pos[0], start_pos[1]] & self.WEST)
        self.full_state['wall_south'] = bool(maze_arr[start_pos[0], start_pos[1]] & self.SOUTH)

        self.full_state['step_rewards'] = {
                'base' : None,
                'goal': None,
                'new_cell': None,
                'wall': None,
                'distance': None,
            }
        
        state = itemgetter(*self.observables)(self.full_state)
        
        telemetry = np.array(state[1:5], dtype=np.float32)

        return {'image': state[0], 'telemetry': telemetry}, {}

        
    def step(self, action):
        reward = self.rew_weights['base']
        terminated = False
        self.full_state['step_rewards']['base'] = self.rew_weights['base']
        self.full_state['n_steps'] += 1
        

        old_pos = self.full_state['pos']
        self.full_state['pos'] = self.take_action(action)
        curr_pos = self.full_state['pos']

        self.full_state['step_rewards']['wall'] = 0
        if (old_pos == curr_pos).all():
            reward += self.rew_weights['wall']
            self.full_state['step_rewards']['wall'] = self.rew_weights['wall']
            if self.end_on_wall:
                terminated = True

        self.full_state['step_rewards']['new_cell'] = 0
        if self.get_cell_idx(curr_pos) not in self.full_state['visited_cells']:
            reward += self.rew_weights['new_cell']
            self.full_state['visited_cells'].add(self.get_cell_idx(curr_pos))
            self.full_state['step_rewards']['new_cell'] = self.rew_weights['new_cell']
        
        goal_pos = self.full_state['goal']
        
        reached_goal = (goal_pos == curr_pos).all()
        
        self.full_state['step_rewards']['goal'] = 0
        if reached_goal:
            terminated = True
            reward += self.rew_weights['goal']
            self.full_state['step_rewards']['goal'] = self.rew_weights['goal']

        if self.is_dfs_dist:
            dist = self.get_dfs_dist(curr_pos, goal_pos)
        else:
            dist = self.get_manhattan_dist(curr_pos, goal_pos)
        self.full_state['step_rewards']['distance'] = self.rew_weights['distance'] * dist
        reward += self.full_state['step_rewards']['distance']
        
        truncated = self.full_state['n_steps'] >= self.max_steps
        
        self.full_state['maze_img'] = self._draw_player_goal(self.full_state['maze_img_original'], curr_pos, goal_pos)
        
        maze_arr = self.full_state['maze_arr']
        self.full_state['wall_north'] = bool(maze_arr[curr_pos[0], curr_pos[1]] & self.NORTH)
        self.full_state['wall_east'] = bool(maze_arr[curr_pos[0], curr_pos[1]] & self.EAST)
        self.full_state['wall_west'] = bool(maze_arr[curr_pos[0], curr_pos[1]] & self.WEST)
        self.full_state['wall_south'] = bool(maze_arr[curr_pos[0], curr_pos[1]] & self.SOUTH)

        state = itemgetter(*self.observables)(self.full_state)

        telemetry = np.array(state[1:5], dtype=np.float32)

        info = {}
        info ['step_rewards'] = self.full_state['step_rewards']

        return {'image': state[0], 'telemetry': telemetry}, reward, terminated, truncated, info


    def take_action(self, action):
        dirn_names = ['north', 'east', 'west', 'south']
        if self.full_state[f"wall_{dirn_names[action]}"]:
            return self.full_state['pos']
        else:
            return self.full_state['pos'] + self.dirns[action]
    
    # Helper: Convert maze (row, col) to pixel (x, y)
    def maze_to_pixel(self, pos, cell_size):
        y = 2 * pos[0] * cell_size + cell_size
        x = 2 * pos[1] * cell_size + cell_size
        return (x, y)

    
    def _draw_player_goal(self, maze_img, agent_pos=None, goal_pos=None, agent_value=128, goal_value=192):
        
        img = maze_img.copy()
        cell_size = int(self.img_shape[0]/(2*self.maze_shape[0] + 1)) 
        
        # Draw Goal (Circle)
        if goal_pos is not None:
            cx, cy = self.maze_to_pixel(goal_pos, cell_size)
            size = 2*cell_size//3
            gap = (cell_size - size)//2
            square = np.array([
                [cx + gap, cy + gap],  # top left
                [cx + cell_size - gap, cy + gap],  # top right
                [cx + cell_size - gap, cy + cell_size - gap],  # bottom right
                [cx + gap , cy + cell_size - gap] # bottom left
            ], dtype=np.int32)
            cv2.fillPoly(img, [square], (0, 255, 0))
    
        # Draw Agent (Triangle)
        if agent_pos is not None:
            cx, cy = self.maze_to_pixel(agent_pos, cell_size)
            size = 2*cell_size//3
            gap = (cell_size - size)//2
            square = np.array([
                [cx + gap, cy + gap],  # top left
                [cx + cell_size - gap, cy + gap],  # top right
                [cx + cell_size - gap, cy + cell_size - gap],  # bottom right
                [cx + gap , cy + cell_size - gap] # bottom left
            ], dtype=np.int32)

            cv2.fillPoly(img, [square], (255, 0, 0))
    
        return img

    def close(self):
        cv2.destroyAllWindows()


class CustomVecMonitor(VecMonitor):
    def __init__(self, venv, filename=None, info_keywords=(), custom_stat_keys=[]):
        
        super().__init__(venv, filename, info_keywords)
        self.custom_stat_keys = custom_stat_keys
        self.default_stat_keys = {'episode_rewards': 'r',
                                  'episode_lengths': 'l',
                                  'episode_times': 't',}
        
        self.episode_custom_stats = {key: [] for key in custom_stat_keys + list(self.default_stat_keys.keys()) + ["env_id"]}
        
        self._current_episode_custom = [
            {key: 0.0 for key in custom_stat_keys} for _ in range(self.num_envs)
        ]
        
    def reset(self):
        obs = super().reset()
        # Reset per-episode accumulators
        self._current_episode_custom = [{key: 0.0 for key in self.custom_stat_keys} for _ in range(self.num_envs)]
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = super().step_wait()

        for i, info in enumerate(infos):
            # Accumulate step_rewards into per-episode trackers
            if "step_rewards" in info:
                for key in self.custom_stat_keys:
                    self._current_episode_custom[i][key] += info["step_rewards"].get(key, 0.0)

            # On episode end, store accumulated stats
            if dones[i]:
                for key, info_key in self.default_stat_keys.items():
                    self.episode_custom_stats[key].append(info["episode"][info_key])
                    
                self.episode_custom_stats['env_id'].append(i)
                
                for key in self.custom_stat_keys:
                    
                    self.episode_custom_stats[key].append(self._current_episode_custom[i][key])
                
                    # Reset accumulator for the next episode
                    self._current_episode_custom[i][key] = 0.0

        return obs, rewards, dones, infos



