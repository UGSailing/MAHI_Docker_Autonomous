# Simple PID control
This is a simple AI generated PID control, that needs to be adjusted to real buoy coordinates, actual rudder angle interfaces and actual MAHI Sense interface.

It exists as a reference/ fallback if more complex algorithms fail. It make a path by constructing 2 circle around the buoys and setting waypoints on it. First the waypoints of the first circles are activated (as to not have trouble at the intersection of the 8 figure). Straight lines are drawn between the points and the intersection of a circle with the straigt line is used as a waypoint navigator (with heading error and constant speed) 
