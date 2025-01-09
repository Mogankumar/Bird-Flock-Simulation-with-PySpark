from pyspark import SparkContext
import numpy as np
import argparse
import time
from get_gif import *

#function to compute speed of a bird
def compute_speed(velocity):
    return np.linalg.norm(velocity)

#function to limit speed
def limit_speed(velocity, min_speed, max_speed):
    speed = compute_speed(velocity)
    if speed < 1e-10:
        return np.zeros_like(velocity)
    if speed < min_speed:
        velocity = velocity / speed * min_speed
    elif speed > max_speed:
        velocity = velocity / speed * max_speed
    return velocity

#function to compute forces (alignment, separation, cohesion)
def compute_forces(bird_data, all_positions, min_distance, max_distance):
    bird_index, bird_position, bird_velocity = bird_data
    distances = np.linalg.norm(all_positions - bird_position, axis=1)

    #following the leader
    d_lead = distances[0]
    lead_force = (all_positions[0] - bird_position) * (1 / d_lead) if d_lead > 10 else np.zeros(3)

    #cohesion
    nearest_idx = np.argmin(distances)
    d_near = distances[nearest_idx]
    cohesion_force = (all_positions[nearest_idx] - bird_position) * (d_near / 1) ** 2 if d_near > max_distance else np.zeros(3)

    #separation
    close_neighbors = all_positions[distances < min_distance]
    close_distances = distances[distances < min_distance]
    separation_force = np.sum([(bird_position - neighbor) / (dist ** 2)
                                for neighbor, dist in zip(close_neighbors, close_distances) if dist > 0], axis=0) if len(close_neighbors) > 0 else np.zeros(3)

    return cohesion_force + separation_force + lead_force

#update bird positions
def update_bird_position(bird_data, all_positions, min_speed, max_speed, time_step, min_distance, max_distance):
    bird_index, bird_position, bird_velocity = bird_data
    forces = compute_forces((bird_index, bird_position, bird_velocity), all_positions, min_distance, max_distance)
    new_velocity = limit_speed(bird_velocity + forces, min_speed, max_speed)
    new_position = bird_position + new_velocity * time_step
    return bird_index, new_position, new_velocity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bird Flock Simulation with PySpark")
    parser.add_argument('--num_birds', type=int, default=10000, help="Number of birds")
    args = parser.parse_args()

    #simulation parameters
    num_birds = args.num_birds
    num_frames = 500
    time_step = 1 / 4
    std_dev_position = 10.0
    lead_bird_speed = 20.0
    lead_bird_radius = 300.0
    min_speed = 10.0
    max_speed = 30.0
    max_distance = 20.0
    min_distance = 10.0

    #initializing positions and velocities
    positions = np.random.normal(loc=np.array([0, 0, 1.5 * lead_bird_radius]), scale=std_dev_position, size=(num_birds, 3))
    velocities = np.zeros((num_birds, 3))

    sc = SparkContext("local[*]", "BirdFlockSimulation")

    simulation = []
    time_cost = []
    for frame in range(num_frames):
        start = time.time()

        #update leader bird position
        angle = lead_bird_speed * frame * time_step / lead_bird_radius
        leader_position = np.array([lead_bird_radius * np.cos(angle),
                                     lead_bird_radius * np.sin(angle) * np.cos(angle),
                                     lead_bird_radius * (1 + 0.5 * np.sin(angle / 5))])
        positions[0] = leader_position

        #parallelize bird data
        bird_data_rdd = sc.parallelize([(i, positions[i], velocities[i]) for i in range(1, num_birds)])

        broadcast_positions = sc.broadcast(positions)

        updated_bird_data = bird_data_rdd.map(
            lambda bird_data: update_bird_position(
                bird_data,
                broadcast_positions.value,
                min_speed,
                max_speed,
                time_step,
                min_distance,
                max_distance
            )
        ).collect()

        for bird_index, new_position, new_velocity in updated_bird_data:
            positions[bird_index] = new_position
            velocities[bird_index] = new_velocity

        simulation.append(positions.copy())
        frame_cost = time.time() - start
        time_cost.append(frame_cost)
        print(f"Frame {frame + 1} simulation time: {frame_cost:.4f}s")

    mean_time = np.mean(time_cost)
    print(f"Average time cost per frame: {mean_time:.4f}")

    visualize_simulation(simulation, lead_bird_radius)
    create_compressed_gif("./plot", gif_name="bird_simulation_spark.gif", duration=100, loop=1, resize_factor=0.5)

    sc.stop()
