# Bird Flock Simulation with PySpark

This project simulates the behavior of a flock of birds using PySpark for parallel computation. The simulation involves updating the positions and velocities of a group of birds based on forces like alignment, separation, and cohesion. Additionally, a leader bird guides the flock. The simulation results in a visualization and a GIF representing the flock’s behavior over time.

**Features**
	•	Simulates bird flock movement using rules of flocking:
	•	Cohesion: Birds are attracted to the nearest bird within a specific distance.
	•	Separation: Birds maintain a minimum distance to avoid collisions.
	•	Following the leader: Birds follow the leader bird as it moves in a circular path.
	•	Uses PySpark for distributed computation to handle large numbers of birds efficiently.
	•	Visualizes the simulation and generates a compressed GIF.

**How It Works**
	1.	Initialization:
  	•	Random positions and velocities are assigned to the birds.
  	•	A leader bird’s trajectory is defined as a circular path.
	2.	Computation:
  	•	For each frame of the simulation:
  	•	Forces such as cohesion, separation, and alignment are computed for each bird relative to its neighbors.
  	•	The birds’ velocities are updated based on the computed forces, and their positions are updated using the velocities.
	3.	Visualization:
  	•	The simulation is visualized using Matplotlib.
  	•	A GIF of the simulation is created using helper functions.

   
