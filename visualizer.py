import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class EVVisualizer:
    def __init__(self, env):
        self.env = env
        # Create two subplots: Grid View and Reward Trend
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Grid layout for consistent positioning
        self.pos = {node: node for node in env.graph.nodes()}
        self.reward_history = []

    def update(self, frame, agents_info=None):
        """
        Main animation update function called by main.py
        """
        # --- Left Subplot: The Environment Grid ---
        self.ax1.clear()
        self.ax1.set_title(f"EV Navigation Grid (Step: {frame})")
        
        # 1. Draw the base grid
        nx.draw(self.env.graph, self.pos, ax=self.ax1, 
                node_color='#f0f0f0', node_size=100, edge_color='#dddddd')
        
        # 2. Draw stations (Blue Squares)
        for s in self.env.stations:
            self.ax1.scatter(s[0], s[1], s=400, c='blue', marker='s', label='Charger')

        # 3. Highlight current paths (Red Lines)
        for path in self.env.current_paths:
            if len(path) > 1:
                edges = list(zip(path, path[1:]))
                nx.draw_networkx_edges(self.env.graph, self.pos, edgelist=edges, 
                                       edge_color='red', width=2, ax=self.ax1)

        # 4. Draw EVs and Status Labels
        for i, p in enumerate(self.env.ev_positions):
            # Draw EV as a green dot
            self.ax1.scatter(p[0], p[1], s=200, c='green', zorder=5)
            
            # Logic for Status Label
            if self.env.charging_timers[i] > 0:
                status_text = f"CHARGING ({self.env.charging_timers[i]}s)"
                color = 'darkorange'
            else:
                status_text = "MOVING"
                color = 'black'
            
            # Display ID, SoC (Battery), and Status above the EV
            label = f"EV{i}\nSoC: {self.env.ev_soc[i]:.2f}\n{status_text}"
            self.ax1.text(p[0], p[1] + 0.35, label, fontsize=9, 
                         ha='center', fontweight='bold', color=color)

        # --- Right Subplot: The Reward Graph ---
        self.ax2.clear()
        self.ax2.set_title("Training Performance (Total Reward)")
        self.ax2.set_xlabel("Steps")
        self.ax2.set_ylabel("Reward Sum")
        
        if len(self.reward_history) > 0:
            self.ax2.plot(self.reward_history, color='orange', linewidth=1.5)
            # Add a horizontal line at 0 for reference
            self.ax2.axhline(0, color='black', linestyle='--', alpha=0.3)

        plt.tight_layout()