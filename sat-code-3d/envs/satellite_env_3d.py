import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit


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


# Create a simple 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


# Plot Earth as a sphere
u_sphere = np.linspace(0, 2 * np.pi, 30)
v_sphere = np.linspace(0, np.pi, 20)
earth_radius = Earth.R.to(u.km).value
x_earth = earth_radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
y_earth = earth_radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
z_earth = earth_radius * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))


ax.plot_surface(x_earth, y_earth, z_earth, alpha=0.4, color='lightblue', label='Earth')


# Plot current satellite position
current_pos = orb.r.to(u.km).value
ax.scatter(current_pos[0], current_pos[1], current_pos[2],
          color='red', s=100, label='Satellite Position')


# Create a simple elliptical orbit approximation for visualization
# This is a simplified representation
semi_major = orb.a.to(u.km).value
ecc = orb.ecc.value


# Generate points for an elliptical orbit (simplified 2D projection)
theta = np.linspace(0, 2*np.pi, 100)
r_orbit = semi_major * (1 - ecc**2) / (1 + ecc * np.cos(theta))


# Simple rotation to approximate the orbital plane
x_orbit = r_orbit * np.cos(theta)
y_orbit = r_orbit * np.sin(theta) * 0.7  # Approximate inclination effect
z_orbit = r_orbit * np.sin(theta) * 0.3  # Approximate inclination effect


ax.plot(x_orbit, y_orbit, z_orbit, 'r--', linewidth=2,
        label='Approximate Orbit Path', alpha=0.7)


# Plot velocity vector (scaled for visibility)
vel_scale = 1000  # Scale factor for visibility
vel = orb.v.to(u.km/u.s).value
ax.quiver(current_pos[0], current_pos[1], current_pos[2],
          vel[0]*vel_scale, vel[1]*vel_scale, vel[2]*vel_scale,
          color='green', arrow_length_ratio=0.1, label='Velocity Vector')


# Set labels and title
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Satellite Orbit Visualization\n(Simplified Representation)')


# Set equal aspect ratio
max_range = max(semi_major * 1.2, earth_radius * 1.5)
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range/2, max_range/2])


ax.legend()
plt.tight_layout()
plt.show()


print("\n=== Summary ===")
print("Successfully created a 3D visualization of the satellite orbit!")
print("The plot shows:")
print("- Blue sphere: Earth")
print("- Red dot: Current satellite position")
print("- Red dashed line: Approximate orbital path")
print("- Green arrow: Velocity vector (scaled for visibility)")
print("\nVirtual environment is working correctly!")

