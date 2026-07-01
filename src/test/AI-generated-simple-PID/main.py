"""
main.py
Runs the figure-8 navigation controller in a simulated loop.
Replace the simulation block with real sensor reads and actuator writes.
"""

import math
import time
import boat_control as bc

# ---------------------------------------------------------------------------
# Simulation state  (replace with real sensor interface)
# ---------------------------------------------------------------------------
sim_lat     = 51.200225    # start near midpoint between buoys
sim_lon     = 4.400100
sim_heading = 0.0          # degrees, north
sim_speed   = 2.0          # m/s (used only for position update in sim)

def sim_update(lat, lon, heading, rudder_norm, throttle, dt):
    """
    Toy kinematic model: constant speed, yaw rate proportional to rudder.
    Remove entirely when running on the real boat.
    """
    yaw_rate_deg_per_s = rudder_norm * bc.MAX_RUDDER_DEG * 0.3   # rough gain
    new_heading = (heading + yaw_rate_deg_per_s * dt) % 360.0

    speed = sim_speed * throttle / bc.THROTTLE   # scale with throttle if you want
    rad   = math.radians(new_heading)
    dlat  = (speed * dt * math.cos(rad)) / bc.EARTH_R
    dlon  = (speed * dt * math.sin(rad)) / (bc.EARTH_R * math.cos(math.radians(lat)))

    new_lat = lat + math.degrees(dlat)
    new_lon = lon + math.degrees(dlon)
    return new_lat, new_lon, new_heading

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():

    origin_lat, origin_lon = bc.BUOY_A
    path       = bc.build_path(origin_lat, origin_lon)
    controller = bc.create_controller()

    print(f"Path has {len(path)} waypoints.")
    print(f"Kp={bc.KP}  Kd={bc.KD}  Lookahead={bc.LOOKAHEAD_M} m  "
          f"Max rudder={bc.MAX_RUDDER_DEG}°  Throttle={bc.THROTTLE}")
    print("-" * 60)

    lat, lon, heading = sim_lat, sim_lon, sim_heading
    dt      = 0.1          # control cycle, seconds
    n_steps = 2000

    for step in range(n_steps):
        rudder, throttle, desired = bc.navigation_step(
            controller, lat, lon, heading, dt, path
        )

        # --- replace from here with real actuator commands ---
        lat, lon, heading = sim_update(lat, lon, heading, rudder, throttle, dt)
        # --- replace to here ---

        if step % 50 == 0:
            print(f"t={step*dt:6.1f}s  lat={lat:.6f}  lon={lon:.6f}  "
                  f"hdg={heading:6.1f}°  desired={desired:6.1f}°  "
                  f"rudder={rudder:+.3f}")

        time.sleep(0)   # swap for time.sleep(dt) on real hardware

if __name__ == "__main__":
    main()
