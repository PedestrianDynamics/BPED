"""Run JuPedSim for curvature analysis."""

import jupedsim as jps
from pathlib import Path
import json
from shapely.geometry import Polygon
import random
import numpy as np
import pedpy
import subprocess

PATH = Path("..")
benchmark_file = "files/cornering.json"
dxf_filename = "data/geometries/cornering.dxf"

print(f"Converting DXF to WKT {dxf_filename = }...")

wkt_filename = dxf_filename.split(".")[0] + ".wkt"
print(wkt_filename)


def load_geometry_from_wkt(wkt_file: Path) -> pedpy.WalkableArea:
    """Load geometry from a WKT file."""
    import shapely.wkt

    with open(wkt_file, "r") as f:
        geometry = shapely.wkt.loads(f.read())
    return pedpy.WalkableArea(geometry)


def load_benchmark_config(config_file: Path) -> dict:
    """Load benchmark configuration from JSON file."""
    with open(config_file, "r") as f:
        return json.load(f)


convert_cmd = [
    "python",
    "scripts/dxf2wkt.py",
    "convert",
    "-i",
    dxf_filename,
    "-o",
    wkt_filename,
]

subprocess.run(convert_cmd, check=True)


walkable_area = load_geometry_from_wkt(wkt_filename)
geometry = walkable_area.polygon


# ---- read json ----
config = load_benchmark_config(benchmark_file)
radius = config["simulation_parameters"]["radius"]
desired_speed = config["simulation_parameters"]["desired_speed"]
num_agents = config["simulation_parameters"]["num_agents"]
time_start = config["simulation_parameters"]["time_start"]
time_end = config["simulation_parameters"]["time_end"]
x_min = config["agent_generation"]["start_area"]["x_min"]
x_max = config["agent_generation"]["start_area"]["x_max"]
y_min = config["agent_generation"]["start_area"]["y_min"]
y_max = config["agent_generation"]["start_area"]["y_max"]

distribution_area = Polygon(
    [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
)

x_min_end = config["agent_generation"]["goal_area"]["x_min"]
x_max_end = config["agent_generation"]["goal_area"]["x_max"]
y_min_end = config["agent_generation"]["goal_area"]["y_min"]
y_max_end = config["agent_generation"]["goal_area"]["y_max"]
goal_area = Polygon(
    [
        [x_min_end, y_min_end],
        [x_max_end, y_min_end],
        [x_max_end, y_max_end],
        [x_min_end, y_max_end],
    ]
)

# --- setup simulation ---
models = ["Collision Free Speed Model", "Social Force Model"]
output_dir = PATH / "submissions/cornering/jupedsim"
output_dir.mkdir(parents=True, exist_ok=True)
simulations = [
    jps.Simulation(
        model=jps.CollisionFreeSpeedModel(),
        geometry=geometry,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=Path(output_dir / "jupedsim_csm.sqlite"),
        ),
    ),
    jps.Simulation(
        model=jps.SocialForceModel(),
        geometry=geometry,
        trajectory_writer=jps.SqliteTrajectoryWriter(
            output_file=Path(output_dir / "jupedsim_sfm.sqlite"),
        ),
    ),
]
agent_parameters = [
    jps.CollisionFreeSpeedModelAgentParameters,
    jps.SocialForceModelAgentParameters,
]

spawn_times = np.linspace(time_start, time_end, num_agents)

for i, (simulation, agent_parameter) in enumerate(zip(simulations, agent_parameters)):
    print("Running simulation with model:", models[i])
    exit_id = simulation.add_exit_stage(goal_area)
    journey = jps.JourneyDescription()
    journey.add(exit_id)
    journey_id = simulation.add_journey(journey)
    # Track which agents have been added
    agents_to_add = list(range(num_agents))
    next_agent_idx = 0
    while simulation.agent_count() > 0 or next_agent_idx < num_agents:
        time_step = simulation.elapsed_time()
        while next_agent_idx < num_agents and spawn_times[next_agent_idx] <= time_step:
            positions = jps.distribute_by_number(
                polygon=distribution_area,
                number_of_agents=1,
                distance_to_agents=0.4,
                distance_to_polygon=0.2,
                seed=random.randint(1, 100000),
            )
            print(
                f"Adding agent {next_agent_idx} at time {time_step:.2f}s at position {positions[0]}"
            )
            parameters = agent_parameter(
                journey_id=journey_id,
                stage_id=exit_id,
                radius=radius,
                desired_speed=desired_speed,
                position=positions[0],
            )
            simulation.add_agent(parameters)
            next_agent_idx += 1

        simulation.iterate()


# import pedpy
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1, figsize=(10, 8))
# data = pedpy.load_trajectory_from_jupedsim_sqlite(Path("cornering_jupedsim_csm.sqlite"))
# pedpy.plot_trajectories(traj=data, walkable_area=walkable_area, axes=ax)
# fig.savefig("cornering_csm.png", dpi=160)
