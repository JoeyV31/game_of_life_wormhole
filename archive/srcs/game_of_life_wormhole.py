import numpy as np
from PIL import Image
import os
from collections import defaultdict

class GameOfLifeWithWormholes:
    def __init__(self, start_img_path, horizontal_tunnel_path, vertical_tunnel_path):
        # Load images
        self.start_img = Image.open(start_img_path)
        self.horizontal_tunnel = Image.open(horizontal_tunnel_path)
        self.vertical_tunnel = Image.open(vertical_tunnel_path)
        
        # Convert to numpy arrays
        self.grid = self._image_to_grid(self.start_img)
        self.height, self.width = self.grid.shape
        
        # Process tunnel images and build wormhole maps
        self.h_tunnel_map = self._process_tunnel_image(self.horizontal_tunnel)
        self.v_tunnel_map = self._process_tunnel_image(self.vertical_tunnel)
        
        # Build wormhole connections
        self.h_wormholes = self._build_wormhole_connections(self.h_tunnel_map)
        self.v_wormholes = self._build_wormhole_connections(self.v_tunnel_map)

    def _image_to_grid(self, img):
        """Convert image to binary grid (1 for alive cells, 0 for dead)"""
        img_array = np.array(img.convert('L'))
        # White (255) = alive (1), Black (0) = dead (0)
        return (img_array > 128).astype(np.uint8)
    
    def _grid_to_image(self, grid):
        """Convert binary grid to black and white image"""
        img_array = grid.astype(np.uint8) * 255
        return Image.fromarray(img_array, mode='L')

    def _process_tunnel_image(self, img):
        """Process tunnel image to extract color information"""
        rgb_img = img.convert('RGB')
        img_array = np.array(rgb_img)
        return img_array

    def _build_wormhole_connections(self, tunnel_map):
        """Build connections between wormhole points"""
        color_to_positions = defaultdict(list)
        
        # Find all colored pixels
        for y in range(self.height):
            for x in range(self.width):
                pixel = tuple(tunnel_map[y, x])
                # Skip black pixels (0, 0, 0)
                if pixel != (0, 0, 0):
                    color_to_positions[pixel].append((y, x))
        
        # Create wormhole connections
        wormhole_map = {}
        for color, positions in color_to_positions.items():
            if len(positions) == 2:
                pos1, pos2 = positions
                wormhole_map[pos1] = pos2
                wormhole_map[pos2] = pos1
        
        return wormhole_map

    def _get_wormhole_neighbor(self, y, x, dy, dx):
        """Get neighbor through wormhole if exists, otherwise get regular neighbor"""
        # Check if we're at a wormhole position
        if dy == 0 and dx != 0:  # Horizontal movement
            if (y, x) in self.h_wormholes:
                # Transport to other side of horizontal wormhole
                ny, nx = self.h_wormholes[(y, x)]
                # Adjust position based on direction
                if dx > 0:  # Moving right
                    return ny, (nx + 1) % self.width
                else:  # Moving left
                    return ny, (nx - 1) % self.width
        
        if dx == 0 and dy != 0:  # Vertical movement
            if (y, x) in self.v_wormholes:
                # Transport to other side of vertical wormhole
                ny, nx = self.v_wormholes[(y, x)]
                # Adjust position based on direction
                if dy > 0:  # Moving down
                    return (ny + 1) % self.height, nx
                else:  # Moving up
                    return (ny - 1) % self.height, nx
        
        # Handle diagonal movement with wormholes according to precedence order
        if dy != 0 and dx != 0:
            # Precedence order: top wormhole > right wormhole > bottom wormhole > left wormhole
            
            # Check top wormhole (if moving up)
            if dy < 0 and (y, x) in self.v_wormholes:
                ny, nx = self.v_wormholes[(y, x)]
                # Adjust for diagonal
                new_x = (nx + dx) % self.width
                return (ny - 1) % self.height, new_x
            
            # Check right wormhole (if moving right)
            if dx > 0 and (y, x) in self.h_wormholes:
                ny, nx = self.h_wormholes[(y, x)]
                # Adjust for diagonal
                new_y = (ny + dy) % self.height
                return new_y, (nx + 1) % self.width
            
            # Check bottom wormhole (if moving down)
            if dy > 0 and (y, x) in self.v_wormholes:
                ny, nx = self.v_wormholes[(y, x)]
                # Adjust for diagonal
                new_x = (nx + dx) % self.width
                return (ny + 1) % self.height, new_x
            
            # Check left wormhole (if moving left)
            if dx < 0 and (y, x) in self.h_wormholes:
                ny, nx = self.h_wormholes[(y, x)]
                # Adjust for diagonal
                new_y = (ny + dy) % self.height
                return new_y, (nx - 1) % self.width
        
        # No wormhole, return regular neighbor coordinates
        new_y = (y + dy) % self.height
        new_x = (x + dx) % self.width
        return new_y, new_x

    def _count_neighbors(self, grid, y, x):
        """Count the number of alive neighbors considering wormholes"""
        count = 0
        
        # Check all 8 surrounding cells
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue  # Skip the cell itself
                
                # Get neighbor coordinates, considering wormholes
                ny, nx = self._get_wormhole_neighbor(y, x, dy, dx)
                
                # Check if neighbor is alive
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    count += grid[ny, nx]
        
        return count

    def step(self):
        """Advance the game by one step"""
        new_grid = np.copy(self.grid)
        
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self._count_neighbors(self.grid, y, x)
                
                # Apply Conway's Game of Life rules
                if self.grid[y, x] == 1:  # Live cell
                    if neighbors < 2 or neighbors > 3:
                        new_grid[y, x] = 0  # Cell dies
                else:  # Dead cell
                    if neighbors == 3:
                        new_grid[y, x] = 1  # Cell becomes alive
        
        self.grid = new_grid
        return new_grid

    def run_simulation(self, steps):
        """Run the simulation for a given number of steps"""
        for _ in range(steps):
            self.step()
        return self.grid

    def save_grid_as_image(self, grid, output_path):
        """Save the grid as a PNG image"""
        img = self._grid_to_image(grid)
        img.save(output_path)


def process_directory(directory):
    """Process a single directory containing input files"""
    start_img_path = os.path.join(directory, "starting_position.png")
    h_tunnel_path = os.path.join(directory, "horizontal_tunnel.png")
    v_tunnel_path = os.path.join(directory, "vertical_tunnel.png")
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [start_img_path, h_tunnel_path, v_tunnel_path]):
        print(f"Missing input files in {directory}")
        return
    
    # Initialize game
    game = GameOfLifeWithWormholes(start_img_path, h_tunnel_path, v_tunnel_path)
    
    # Run and save results for required iterations
    iterations = [1, 10, 100, 1000]
    
    for i in iterations:
        # Run simulation
        if i == 1:
            grid = game.run_simulation(1)
        elif i == 10:
            grid = game.run_simulation(9)  # Already ran 1 step before
        elif i == 100:
            grid = game.run_simulation(90)  # Already ran 10 steps before
        elif i == 1000:
            grid = game.run_simulation(900)  # Already ran 100 steps before
        
        # Save result
        output_path = os.path.join(directory, f"{i}.png")
        game.save_grid_as_image(grid, output_path)
        print(f"Generated {output_path}")


def main():
    """Process all example directories"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)  # Assuming script is in srcs/
    
    # Find all example directories
    for item in os.listdir(parent_dir):
        item_path = os.path.join(parent_dir, item)
        if os.path.isdir(item_path) and (item.startswith("example-") or item.startswith("problem-")):
            print(f"Processing {item}...")
            process_directory(item_path)


if __name__ == "__main__":
    main()