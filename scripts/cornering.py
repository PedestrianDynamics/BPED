"""Run JuPedSim for curvature analysis.

Call as module
uv run python -m scripts.cornering.py

"""

import json
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import jupedsim as jps
import numpy as np
import pedpy
import shapely.wkt
from shapely.geometry import Polygon

from src.bped.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


# ============================================================================
# Configuration and File Handling
# ============================================================================


def get_file_paths() -> Dict[str, Path]:
    """Get all file paths used in the simulation."""
    return {
        "benchmark": Path("files/cornering.json"),
        "dxf": Path("data/geometries/cornering.dxf"),
        "wkt": Path("data/geometries/cornering.wkt"),
        "output_dir": Path("simulation_results/cornering/jupedsim"),
    }


def load_benchmark_config(config_file: Path) -> dict:
    """Load benchmark configuration from JSON file."""
    logger.info(f"Loading benchmark configuration from {config_file}")
    with open(config_file, "r") as f:
        return json.load(f)


# ============================================================================
# Geometry Processing
# ============================================================================


def convert_dxf_to_wkt(dxf_file: Path, wkt_file: Path) -> None:
    """Convert DXF file to WKT format."""
    logger.info(f"Converting DXF to WKT: {dxf_file} -> {wkt_file}")

    convert_cmd = [
        "python",
        "scripts/dxf2wkt.py",
        "convert",
        "-i",
        str(dxf_file),
        "-o",
        str(wkt_file),
    ]

    subprocess.run(convert_cmd, check=True)
    logger.info("DXF conversion completed successfully")


def load_geometry_from_wkt(wkt_file: Path) -> pedpy.WalkableArea:
    """Load geometry from a WKT file."""
    logger.info(f"Loading geometry from {wkt_file}")

    with open(wkt_file, "r") as f:
        geometry = shapely.wkt.loads(f.read())

    return pedpy.WalkableArea(geometry)


def create_polygon_from_bounds(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> Polygon:
    """Create a rectangular polygon from min/max coordinates."""
    return Polygon([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])


# ============================================================================
# Simulation Setup
# ============================================================================


def visualize_results(
    walkable_area: pedpy.WalkableArea, trajectory_file: Path, output_file: Path
) -> None:
    """Visualize simulation trajectories."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    data = pedpy.load_trajectory_from_jupedsim_sqlite(trajectory_file)
    pedpy.plot_trajectories(traj=data, walkable_area=walkable_area, axes=ax)
    fig.savefig(output_file, dpi=160)
    logger.info(f"Trajectory visualization saved to {output_file}")


def extract_simulation_parameters(config: dict) -> dict:
    """Extract simulation parameters from config."""
    sim_params = config["simulation_parameters"]

    return {
        "radius": sim_params["radius"],
        "desired_speed": sim_params["desired_speed"],
        "num_agents": sim_params["num_agents"],
        "time_start": sim_params["time_start"],
        "time_end": sim_params["time_end"],
    }


def extract_area_polygons(config: dict) -> Tuple[Polygon, Polygon]:
    """Extract start and goal area polygons from config."""
    start = config["agent_generation"]["start_area"]
    goal = config["agent_generation"]["goal_area"]

    start_area = create_polygon_from_bounds(
        start["x_min"], start["x_max"], start["y_min"], start["y_max"]
    )

    goal_area = create_polygon_from_bounds(
        goal["x_min"], goal["x_max"], goal["y_min"], goal["y_max"]
    )

    return start_area, goal_area


def create_simulations(geometry: Polygon, output_dir: Path) -> List[Dict]:
    """Create simulation configurations for different models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    simulations = [
        {
            "name": "Collision Free Speed Model",
            "model": jps.CollisionFreeSpeedModel(),
            "agent_parameters": jps.CollisionFreeSpeedModelAgentParameters,
            "output_file": output_dir / "jupedsim_csm.sqlite",
        },
        {
            "name": "Social Force Model",
            "model": jps.SocialForceModel(),
            "agent_parameters": jps.SocialForceModelAgentParameters,
            "output_file": output_dir / "jupedsim_sfm.sqlite",
        },
    ]

    # Create JuPedSim simulation objects
    for sim_config in simulations:
        sim_config["simulation"] = jps.Simulation(
            model=sim_config["model"],
            geometry=geometry,
            trajectory_writer=jps.SqliteTrajectoryWriter(
                output_file=sim_config["output_file"]
            ),
        )

    return simulations


# ============================================================================
# Agent Management
# ============================================================================


def calculate_spawn_times(
    time_start: float, time_end: float, num_agents: int
) -> np.ndarray:
    """Calculate spawn times for agents uniformly distributed over time."""
    return np.linspace(time_start, time_end, num_agents)


def spawn_agent(
    simulation: jps.Simulation,
    distribution_area: Polygon,
    agent_parameter_class: type,
    journey_id: int,
    exit_id: int,
    radius: float,
    desired_speed: float,
    agent_idx: int,
    time_step: float,
) -> None:
    """Spawn a single agent in the simulation."""
    positions = jps.distribute_by_number(
        polygon=distribution_area,
        number_of_agents=1,
        distance_to_agents=0.4,
        distance_to_polygon=0.2,
        seed=random.randint(1, 100000),
    )

    logger.debug(
        f"Adding agent {agent_idx} at time {time_step:.2f}s at position {positions[0]}"
    )
    if agent_parameter_class == jps.SocialForceModelAgentParameters:
        # This workaround is necessary because of a bug in the SFM. Agents can't navigate this narrow geometry.
        parameters = agent_parameter_class(
            journey_id=journey_id,
            stage_id=exit_id,
            radius=radius,
            desired_speed=desired_speed,
            position=positions[0],
            obstacle_scale=2.0,  # Workaround for narrow geometry issue
        )

    else:
        parameters = agent_parameter_class(
            journey_id=journey_id,
            stage_id=exit_id,
            radius=radius,
            desired_speed=desired_speed,
            position=positions[0],
        )

    simulation.add_agent(parameters)


# ============================================================================
# Simulation Execution
# ============================================================================


def setup_journey(simulation: jps.Simulation, goal_area: Polygon) -> Tuple[int, int]:
    """Set up the journey (exit and path) for agents."""
    exit_id = simulation.add_exit_stage(goal_area)
    journey = jps.JourneyDescription()
    journey.add(exit_id)
    journey_id = simulation.add_journey(journey)

    return journey_id, exit_id


def run_simulation(
    sim_config: dict,
    distribution_area: Polygon,
    goal_area: Polygon,
    sim_params: dict,
    spawn_times: np.ndarray,
) -> None:
    """Run a single simulation with the given configuration."""
    logger.info(f"Running simulation with model: {sim_config['name']}")

    simulation = sim_config["simulation"]
    agent_parameter_class = sim_config["agent_parameters"]

    # Setup journey
    journey_id, exit_id = setup_journey(simulation, goal_area)

    # Run simulation with time-based agent spawning
    next_agent_idx = 0
    num_agents = len(spawn_times)

    while simulation.agent_count() > 0 or next_agent_idx < num_agents:
        time_step = simulation.elapsed_time()

        # Spawn agents whose time has come
        while next_agent_idx < num_agents and spawn_times[next_agent_idx] <= time_step:
            spawn_agent(
                simulation=simulation,
                distribution_area=distribution_area,
                agent_parameter_class=agent_parameter_class,
                journey_id=journey_id,
                exit_id=exit_id,
                radius=sim_params["radius"],
                desired_speed=sim_params["desired_speed"],
                agent_idx=next_agent_idx,
                time_step=time_step,
            )
            next_agent_idx += 1

        simulation.iterate()

    logger.info(f"Simulation completed: {sim_config['name']}")
    paths = get_file_paths()
    walkable_area = load_geometry_from_wkt(paths["wkt"])
    figname = paths["output_dir"] / (sim_config["name"].replace(" ", "_") + ".png")
    visualize_results(
        walkable_area=walkable_area,
        trajectory_file=sim_config["output_file"],
        output_file=figname,
    )
    logger.info(f"Plot trajectories to {figname}")


def run_all_simulations(
    simulations: List[Dict],
    distribution_area: Polygon,
    goal_area: Polygon,
    sim_params: dict,
    spawn_times: np.ndarray,
) -> None:
    """Run all configured simulations."""
    for sim_config in simulations:
        run_simulation(
            sim_config=sim_config,
            distribution_area=distribution_area,
            goal_area=goal_area,
            sim_params=sim_params,
            spawn_times=spawn_times,
        )


# ============================================================================
# Main Orchestration
# ============================================================================


def main() -> None:
    """Main function to orchestrate the cornering benchmark simulation."""
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    log_file = f"cornering_{timestamp}.log"
    setup_logging(log_file=log_file)

    logger.info("=" * 70)
    logger.info("Starting JuPedSim Cornering Simulation")
    logger.info("=" * 70)

    try:
        paths = get_file_paths()

        # Convert geometry if needed
        if not paths["wkt"].exists():
            convert_dxf_to_wkt(paths["dxf"], paths["wkt"])
        else:
            logger.info(f"Using existing WKT file: {paths['wkt']}")

        walkable_area = load_geometry_from_wkt(paths["wkt"])
        geometry = walkable_area.polygon

        config = load_benchmark_config(paths["benchmark"])
        sim_params = extract_simulation_parameters(config)
        distribution_area, goal_area = extract_area_polygons(config)

        logger.info(f"Simulation parameters: {sim_params}")

        # Calculate spawn times
        spawn_times = calculate_spawn_times(
            sim_params["time_start"], sim_params["time_end"], sim_params["num_agents"]
        )

        # Create and run simulations
        simulations = create_simulations(geometry, paths["output_dir"])
        run_all_simulations(
            simulations=simulations,
            distribution_area=distribution_area,
            goal_area=goal_area,
            sim_params=sim_params,
            spawn_times=spawn_times,
        )

        logger.info("=" * 70)
        logger.info("All simulations completed successfully")
        logger.info(f"Results saved to: {paths['output_dir']}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Simulation failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
