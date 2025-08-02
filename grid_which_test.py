import matplotlib.pyplot as plt
import numpy as np

# --- 1. Create Data and Basic Plot ---
# Generate sample data for a sine wave
x = np.linspace(0, 10, 1000)
y = np.sin(x)

# Create a figure and axes object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
ax.plot(x, y, label='sin(x)')

# --- 2. Set Major Ticks ---
# Set the locations of the major ticks on the x-axis (every 2 units)
ax.set_xticks(np.arange(0, 11, 2))
# Set the locations of the major ticks on the y-axis (every 0.5 units)
ax.set_yticks(np.arange(-1, 1.1, 0.5))

# --- 3. Activate Minor Ticks ---
# Automatically add minor ticks between the major ones
ax.minorticks_on()

# --- 4. Style Ticks for Clarity ---
# Customize the appearance of major and minor ticks
# which='major': applies only to major ticks
# which='minor': applies only to minor ticks
# which='both': applies to both
ax.tick_params(which='major', length=10, width=2, direction='in', labelsize=12)
ax.tick_params(which='minor', length=5, width=1, direction='in', color='gray')

# --- 5. Add Grid Lines ---
# Add a grid for the major ticks
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
# Add a grid for the minor ticks
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')


# --- 6. Final Touches ---
# Add title and labels
ax.set_title('Major vs. Minor Ticks Example', fontsize=16)
ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.legend()

# Display the plot
plt.show()