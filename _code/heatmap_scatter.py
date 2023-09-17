import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Filter agent_out data for Land agents
df = agent_out.query("agent_type == 'Land'")
df = df.dropna(axis=1, how='all').reset_index(drop=True)
df = df.reset_index(drop=True)

# Get the range of time steps
time_steps = df['time_step'].unique()
num_time_steps = len(time_steps)
middle_time_step = time_steps[num_time_steps // 2]

# Find the overall min and max values for the color scale
min_price = df['warranted_price'].min()
max_price = df['warranted_price'].max()

# Create subplots with 2 rows and 3 columns
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Create a common colorbar axis
divider = make_axes_locatable(axs[0, -1])
cax = divider.append_axes("right", size="5%", pad=0.1)

# Iterate over the subplots and time steps
for i, time_step in enumerate([time_steps[0], middle_time_step, time_steps[-1]]):
    land_agents = df.query("time_step == @time_step")
    
    # Create a grid to represent the space
    grid_size = (df['x'].max() + 1, df['y'].max() + 1)
    heatmap = np.zeros(grid_size)
    
    # Fill the heatmap with the values
    for index, row in land_agents.iterrows():
        x = int(row['x'])
        y = int(row['y'])
        price = row['warranted_price']
        heatmap[x, y] = price
    
    # Display the heatmap in the current subplot
    im = axs[0, i].imshow(heatmap, cmap='viridis', origin='lower', extent=[0, grid_size[0], 0, grid_size[1]], vmin=min_price, vmax=max_price)
    axs[0, i].set_title(f'Time Step {time_step}')
    axs[0, i].set_xlabel('X')
    axs[0, i].set_ylabel('Y')
    axs[0, i].grid(False)
    
    # Create scatter plot for distance from center vs warranted price
    scatter = axs[1, i].scatter(land_agents['distance_from_center'], land_agents['warranted_price'], c='blue', alpha=0.5)
    axs[1, i].set_title(f'Distance vs Warranted Price at Time Step {time_step}')
    axs[1, i].set_xlabel('Distance from Center')
    axs[1, i].set_ylabel('Warranted Price')
    axs[1, i].grid(True)
    
    # Set same min and max values for scatter plots
    axs[1, i].set_ylim(min_price, max_price)

# Add a common colorbar
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Warranted Price')

plt.tight_layout()
plt.show()
