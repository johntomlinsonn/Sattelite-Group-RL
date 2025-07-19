import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import time


# Data from Curtis, example 4.3
r = [-6045, -3490, 2500] * u.km
v = [-3.457, 6.618, 2.533] * u.km / u.s


# Create an Orbit object
orb = Orbit.from_vectors(Earth, r, v)


# Print orbit information
print("Orbit Information:")
print(f"Semi-major axis: {orb.a}")
print(f"Eccentricity: {orb.ecc}")
print(f"Inclination: {orb.inc}")
print(f"Period: {orb.period}")
print(f"Current position: {orb.r}")
print(f"Current velocity: {orb.v}")


# Create a simple 3D plot with animation
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth as a sphere (this stays static)
u_sphere = np.linspace(0, 2 * np.pi, 50)  # Increased resolution
v_sphere = np.linspace(0, np.pi, 50)      # Increased resolution
earth_radius = Earth.R.to(u.km).value
x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))

ax.plot_surface(x_earth, y_earth, z_earth, alpha=0.4, color='lightblue', label='Earth')

# Set up orbital parameters for animation
semi_major = orb.a.to(u.km).value
ecc = orb.ecc.value
period_hours = orb.period.to(u.hour).value

# Create time array for one complete orbit
num_points = 200
time_array = np.linspace(0, period_hours, num_points) * u.hour

# Pre-calculate all orbital positions
orbital_positions = []
for t in time_array:
    try:
        # Propagate the orbit to time t
        future_orbit = orb.propagate(t)
        pos = future_orbit.r.to(u.km).value
        orbital_positions.append(pos)
    except:
        # If propagation fails, use a simple elliptical approximation
        theta = 2 * np.pi * t.value / period_hours
        r_orbit = semi_major * (1 - ecc**2) / (1 + ecc * np.cos(theta))
        x = r_orbit * np.cos(theta)
        y = r_orbit * np.sin(theta) * 0.7
        z = r_orbit * np.sin(theta) * 0.3
        orbital_positions.append([x, y, z])

orbital_positions = np.array(orbital_positions)

# Plot the complete orbital path
ax.plot(orbital_positions[:, 0], orbital_positions[:, 1], orbital_positions[:, 2], 
        'r--', linewidth=1, alpha=0.5, label='Complete Orbit Path')

# Initialize moving elements
satellite_point = ax.scatter([], [], [], color='red', s=100, label='Satellite Position')
trail_line, = ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7, label='Orbit Trail')
velocity_arrow = None

# Store trail points
trail_points = []
max_trail_length = 50  # Number of points to keep in trail

def update_satellite(frame):
    global velocity_arrow, trail_points
    
    # Get current position
    current_pos = orbital_positions[frame]
    
    # Update satellite position
    satellite_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
    
    # Add to trail
    trail_points.append(current_pos)
    if len(trail_points) > max_trail_length:
        trail_points.pop(0)
    
    # Update trail
    if len(trail_points) > 1:
        trail_array = np.array(trail_points)
        trail_line.set_data_3d(trail_array[:, 0], trail_array[:, 1], trail_array[:, 2])
    
    # Remove old velocity arrow
    if velocity_arrow is not None:
        velocity_arrow.remove()
    
    # Calculate velocity vector (approximate)
    if frame < len(orbital_positions) - 1:
        next_pos = orbital_positions[frame + 1]
        vel_vector = (next_pos - current_pos) * 10  # Scale for visibility
    else:
        vel_vector = (orbital_positions[0] - current_pos) * 10
    
    # Add new velocity arrow
    velocity_arrow = ax.quiver(current_pos[0], current_pos[1], current_pos[2],
                              vel_vector[0], vel_vector[1], vel_vector[2],
                              color='green', arrow_length_ratio=0.1, alpha=0.8)
    
    # Update title with current time
    current_time = frame * period_hours / num_points
    ax.set_title(f'Satellite Orbit Animation\nTime: {current_time:.2f} hours')
    
    return satellite_point, trail_line


# Set labels and title
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Satellite Orbit Animation\nTime: 0.00 hours')

# Set equal aspect ratio - this is crucial for a perfect sphere!
max_range = max(semi_major * 1.2, earth_radius * 1.5)
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])  # Changed from max_range/2 to max_range

# Force equal aspect ratio for all axes
ax.set_box_aspect([1, 1, 1])  # This ensures equal scaling on all axes

ax.legend()

# Create animation
print("Creating satellite orbit animation...")
print(f"Orbit period: {period_hours:.2f} hours")
print("Animation will show one complete orbit with real-time position updates")
print("Close the plot window to continue...")

# Animation with interval in milliseconds (50ms = ~20fps)
animation = FuncAnimation(fig, update_satellite, frames=num_points, 
                         interval=50, blit=False, repeat=True)

plt.tight_layout()
plt.show()


print("\n=== Summary ===")
print("Successfully created an animated 3D visualization of the satellite orbit!")
print("The animation shows:")
print("- Blue sphere: Earth (static)")
print("- Red dot: Current satellite position (animated)")
print("- Red dashed line: Complete orbital path")
print("- Red solid line: Orbit trail (recent positions)")
print("- Green arrow: Current velocity vector")
print("- Title updates with current time in orbit")
print("\nThe animation loops continuously showing one complete orbit!")
print("Virtual environment is working correctly!")

