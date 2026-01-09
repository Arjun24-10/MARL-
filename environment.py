import networkx as nx
import numpy as np

class EVChargingEnv:
    def __init__(self, num_evs=2):
        self.num_evs = num_evs
        self.grid_size = 5
        self.graph = nx.grid_2d_graph(self.grid_size, self.grid_size)
        self.stations = [(0, 0), (4, 4)] 
        self.station_queues = {s: 0 for s in self.stations}
        self.current_paths = [[] for _ in range(num_evs)] 
        
        # New: Charging logic variables
        self.charging_timers = [0 for _ in range(num_evs)]
        self.charging_duration = 3  # EV must wait 3 steps to fully charge
        
        self.reset()

    def reset(self):
        self.ev_positions = [list(self.graph.nodes)[np.random.choice(25)] for _ in range(self.num_evs)]
        self.ev_soc = np.random.uniform(0.4, 0.7, self.num_evs)
        self.station_queues = {s: 0 for s in self.stations}
        self.charging_timers = [0 for _ in range(self.num_evs)]
        self.current_paths = [[] for _ in range(self.num_evs)]
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.num_evs):
            pos = self.ev_positions[i]
            dists = [nx.shortest_path_length(self.graph, pos, s) for s in self.stations]
            # Observation includes current SoC and distances to stations
            obs.append(np.array([pos[0], pos[1], self.ev_soc[i]] + dists, dtype=np.float32))
        return obs

    def step(self, actions):
        rewards = []
        temp_queues = {s: 0 for s in self.stations}
        
        for i, act in enumerate(actions):
            # 1. Handle EVs that are currently charging
            if self.charging_timers[i] > 0:
                self.charging_timers[i] -= 1
                self.current_paths[i] = [self.ev_positions[i]] # Stay put
                
                if self.charging_timers[i] == 0:
                    self.ev_soc[i] = 1.0 # Refill SoC only when timer hits 0
                
                rewards.append(-0.5) # Small "time cost" penalty for waiting
                continue

            # 2. Handle EVs that are moving to a station
            target = self.stations[act]
            temp_queues[target] += 1
            
            path = nx.shortest_path(self.graph, self.ev_positions[i], target)
            self.current_paths[i] = path 
            
            dist = len(path) - 1
            self.ev_positions[i] = target
            self.ev_soc[i] -= (dist * 0.05)
            
            # 3. Calculate Reward
            reward = -(dist * 1.5) - (temp_queues[target] * 2.0)
            
            if self.ev_soc[i] <= 0:
                reward -= 100 # Heavy penalty for running out of battery
                # Optional: reset SoC to 0.1 so it doesn't stay negative
                self.ev_soc[i] = 0.1 
            else:
                reward += 20 # Bonus for reaching a station
                self.charging_timers[i] = self.charging_duration # Start the wait
                
            rewards.append(reward)
            
        self.station_queues = temp_queues
        return self._get_obs(), rewards