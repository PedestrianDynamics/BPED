#!python3.12
import os
import random
import logging
import matplotlib.pyplot as plt
import shapely.plotting
import cad_to_shapely.dxf as dxf
import utils
import jupedsim as jps
from jupedsim.internal.notebook_utils import animate, read_sqlite_file
import pathlib
import pedpy
import ezdxf
import time
import math
import numpy as np
import pandas as pd
from dxf2wkt import convert as dxfconvert
from enum import Enum
import random

class Agents(Enum):
   STANDARD = 0
   SLOW     = 1
   DISABLED = 2

#CUSTOM read and extraction of information from a single CAD FILE [replaced by dxf2wkt.py JuPedSim script]
def load_file_dxf( filename, multipoly=False ):
   print( f'Extracting layer: {filename}' )

   #Gestione DXF_CAD_file ed estrazione di un singolo LAYER
   DXF_CAD_file = ezdxf.readfile( "FileCompleto.dxf" )

   key_func = DXF_CAD_file.layers.key
   layer_key = key_func( filename )
   # The trashcan context-manager is a safe way to delete entities from the
   # entities database while iterating.
   with DXF_CAD_file.entitydb.trashcan() as trash:
      for entity in DXF_CAD_file.entitydb.values():
         if not entity.dxf.hasattr("layer"):
            # safe destruction while iterating
            continue

         #Dobbiamo tenere anche tutti gli oggeti nel LAYER 0
         if layer_key != key_func(entity.dxf.layer) and key_func(entity.dxf.layer) != "0":
            trash.add(entity.dxf.handle)

   DXF_CAD_file.saveas( filename + ".dxf")

   #Apertura del nuovo file dxf del layer
   print( f'Reading file: {filename}.dxf' )
   dxf_filepath = os.path.join( os.getcwd(), filename + ".dxf" )
   my_dxf = dxf.DxfImporter(dxf_filepath)
   my_dxf.process(spline_delta = 0.1)
   print(f'Units are {my_dxf.units}')
   my_dxf.polygonize( force_zip = False )

   polygons = my_dxf.polygons
   print (f"Found {len(polygons)} polygons")

   if ( not multipoly ):
      #Generazione poligono di area massima
      area_max = -1e-10
      for p in polygons:
          print( f"POLYGON FOUND: {p.area} - maxArea = {area_max}" )
          area_max = max( area_max, p.area )

      for p in polygons:
          if ( p.area == area_max ):
              single_polygon = p
              break

      #Rimozione degli holes dalla single_polygon
      new = utils.find_holes(polygons)

      for hole in new.interiors:
          single_polygon = single_polygon.difference( hole )

      return single_polygon

   else:
      return polygons

def plot_simulation_configuration(geometry, starting_positions, exit_areas):
    """Plot setup for visual inspection."""
    walkable_area = pedpy.WalkableArea(geometry)
    fig, ax = plt.subplots()
    pedpy.plot_walkable_area( axes=ax, walkable_area=walkable_area)
    for exit_area in exit_areas:
        ax.fill(*exit_area.exterior.xy, color="indianred")

    ax.scatter(*zip(*starting_positions), label="Starting Position")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.savefig( "walkable_area_" + str( iAnalysis ) + ".png" )
    plt.close( fig )

if __name__ == "__main__":
   nAgents  = 15
   nIterMax = 100000
   probabilityCloseExit = 0.90
   Exec     = True
   PostProc = True
   dTime    = 0.01
   #model    = "SocialForceModel"        #pushing behavior in dense crowds [evacuation] [alternative: GeneralizedCentrifugalForceModel]
   model    = "CollisionFreeSpeedModel" #normal walking situations [alternative: AnticipationVelocityModel]

   nAnalysis = 10 #Number of analysis for stocastic approach
   
   #Agend types
   AgentSlow    = 0.16 # 1 = 100%
   AgentDisable = 0.00 # 1 = 100%
   Agent        = 1 - AgentSlow - AgentDisable

   #Files
   filename      = "Counterflow.dxf"
   WALLS         = "WALLS"
   EXITS         = [ "EXITS" ]
   DISTRIBUTIONS = [ "DISTRIBUTIONS" ]

   print( "########## SETUP #################" )
   #official reader from JuPedSim
   dxfconvert( input=pathlib.Path( filename ), 
               output=pathlib.Path( "output.wkt" ), 
               dxf_output=None,  
               walkable=WALLS,
               obstacles=( [] ), 
               exits=( EXITS ), 
               distributions=( DISTRIBUTIONS ), 
               waypoints=( [] ), 
               journeys=( [] ) 
              )

   #### STOCASTIC APPROACH of ANALYSES ##################
   evacuationTime     = pd.DataFrame( { "Evacuation time": [ ] }  )
   countAgentsPerType = pd.DataFrame( { Agents.STANDARD.name: [ ], Agents.SLOW.name    : [ ], Agents.DISABLED.name: [ ] })
   fig_AgentsTime, ax_AgentsTime = plt.subplots()

   for iAnalysis in range( nAnalysis ):
      print( "############## NEW ANALISIS {} / {} ###########################".format( iAnalysis, nAnalysis ) )

      #Reading WKT file
      with open('output.wkt', 'r') as file:
         wkt_string = file.read().replace('\n', '')
      geometry_collection = shapely.wkt.loads( wkt_string )

      #### GEOMETRY #########################################
      #TODO geometry = load_file_dxf( "WALLS", multipoly=False )
      geometry = geometry_collection.geoms[0].geoms[0]

      if ( model == "SocialForceModel" ):
         jpsmodel = jps.SocialForceModel()
      elif ( model == "CollisionFreeSpeedModel" ):
         jpsmodel = jps.CollisionFreeSpeedModel()
      else:
         print( "Error model - DEFINITION" )
         exit()

      if 'simulation_cfsm' in globals():
         del simulation_cfsm

      trajectory_file="traj_" + str( iAnalysis ) + ".sqlite"

      simulation_cfsm = jps.Simulation(
         model=jpsmodel,
         geometry=geometry,
         dt=dTime,
         trajectory_writer=jps.SqliteTrajectoryWriter( output_file=pathlib.Path( trajectory_file), every_nth_frame=int( 1 / dTime ) ) )

      #### EGRESS PATHS #####################################
      #TODO exits = load_file_dxf( "EXITS", multipoly=True )
      exits = geometry_collection.geoms[1].geoms

      exit_ids = []

      for ex in exits:
         exit_id = simulation_cfsm.add_exit_stage( ex )
         exit_ids.append( exit_id )

      journey = jps.JourneyDescription(exit_ids)
      journey_id = simulation_cfsm.add_journey(journey)

      #### AGENTS DISTRIBUTION ##############################
      distributions = geometry_collection.geoms[2].geoms

      #if no distribution layer is preset => agents scattered on the whole geometry
      if len( distributions ) == 0:
         distributions = [ geometry ]

      start_positions = []
      for distribution in distributions:
     
         #Number of agent in the current distribution
         nLocalAgent = int( nAgents/len(distributions) )

         #TODO attempt to define a min agent distance in the current distribution area
         #distance    = max( math.sqrt( distribution.area / math.pi / nLocalAgent ), 0.5 )
         distance    = max( math.sqrt( distribution.area / math.pi / nLocalAgent ) / 10 , 0.5 )

         start_positions_local = jps.distribute_by_number( polygon=distribution, number_of_agents=nLocalAgent, distance_to_agents=distance, distance_to_polygon=distance )
         start_positions += start_positions_local

      #GEOMETRY and initial agent position plot
      plot_simulation_configuration( geometry, start_positions, exits )

      countAgentsPerType_i = pd.DataFrame(
       { Agents.STANDARD.name: [ 0 ],
         Agents.SLOW.name    : [ 0 ],
         Agents.DISABLED.name: [ 0 ]
       })

      #Agent definition and link to an emergency exit
      for position in start_positions:
         point = shapely.Point( position )
         AgentType = Agents( np.random.choice( 3, p=[ Agent, AgentSlow, AgentDisable] ) ).name
         countAgentsPerType_i.loc[ 0, AgentType ] = countAgentsPerType_i.loc[ 0, AgentType ] + 1

         match AgentType:
            case Agents.STANDARD.name:
               desired_speed=1.12
               reaction_time=0.5
               radius=0.25
               scale_force=2000
            case Agents.SLOW.name:
               desired_speed=0.93
               reaction_time=1.0
               radius=0.25
               scale_force=1000
            case Agents.DISABLED.name:
               desired_speed=0.66
               reaction_time=0.75
               radius=0.5
               scale_force=800

         #Assegnazione uscita di emergenza a ciascun agente
         #Criterio 1 -> assegnamo una uscita di emergenza vicina
         #Criterio 2 -> ogni tanto saltiamo il criterio di vicinanza [con percentuale definita probabilityCloseExit]
         #Criterio 3 -> quando non si assegna niente, si va completamente random
         distance = 1E+30
         exit_id  = 0
         curr_id  = 0
         for ex in exits:
            if ( shapely.distance( point, ex ) < distance and random.random() < probabilityCloseExit ):
               distance = shapely.distance( point, ex )
               exit_id  = exit_ids[ curr_id ]

            curr_id = curr_id + 1

         if ( exit_id == 0 ):
            #exit_id = int( math.ceil( random.random() * len( exit_ids ) ) )
            exit_id = random.choice( exit_ids )
            print( f"Full random USER - afterRandom {exit_id}" )

         if ( model == "SocialForceModel" ):
            simulation_cfsm.add_agent(
                   jps.SocialForceModelAgentParameters(
                       journey_id=journey_id,
                       stage_id=exit_id,
                       desired_speed=desired_speed,
                       reaction_time=reaction_time,
                       radius=radius,
                       agent_scale=scale_force,
                       obstacle_scale=scale_force,
                       position=position,
                       orientation=([ 1.0, 0.0 ])
                   )
            )
         elif ( model == "CollisionFreeSpeedModel" ):
           simulation_cfsm.add_agent(
                 jps.CollisionFreeSpeedModelAgentParameters(
                 journey_id=journey_id,
                 desired_speed=desired_speed,
                 radius=radius,
                 stage_id=exit_id,
                 position=position
               )
            )
         else:
            print( "Error model - AGENT" )
            exit()

      countAgentsPerType = pd.concat( [countAgentsPerType, countAgentsPerType_i], ignore_index=True )
   
      if ( Exec ):
         print( "########## SIMULATION ############" )

         times  = []
         agents = []

         times.append( 0 )
         agents.append( nAgents )

         while ( simulation_cfsm.agent_count() > 0 and simulation_cfsm.iteration_count() < nIterMax ):
            if ( simulation_cfsm.iteration_count() % 1000 == 0 ):
                print( f"Iteration n. {simulation_cfsm.iteration_count():6d} / {nIterMax} - Agents n. {simulation_cfsm.agent_count():4d} / {nAgents} - Time {simulation_cfsm.elapsed_time():.2f}s ")
            simulation_cfsm.iterate()

            times.append( simulation_cfsm.elapsed_time() )
            agents.append( simulation_cfsm.agent_count() )

         ax_AgentsTime.plot( times, agents, label="line {}".format( iAnalysis ) )
         ax_AgentsTime.set_xbound( lower=0 )
         ax_AgentsTime.set_ybound( lower=0 )
         plt.xlabel("Time [s]")
         plt.ylabel("Agents []")
         plt.legend(loc="upper left")
         plt.grid( visible=True )
   
         print(f"Evacuation time: {simulation_cfsm.elapsed_time():.2f}s")

         evacuationTime_i = pd.DataFrame( { "Evacuation time": [ simulation_cfsm.elapsed_time() ] }  )
         evacuationTime = pd.concat( [evacuationTime, evacuationTime_i], ignore_index=True )
         #evacuationTime.loc[-1] = [ simulation_cfsm.elapsed_time() ]

#      if ( PostProc ):
#         print( "########## POSTPROC ##############" )
#   
#         print( "trajectories ..." )
#         trajectory_data, walkable_area = read_sqlite_file( trajectory_file )
#         fig, ax = plt.subplots()
#         pedpy.plot_trajectories( traj=trajectory_data, axes=ax, walkable_area=pedpy.WalkableArea(geometry) )
#         ax.set_aspect("equal")
#         fig.savefig( "trajectories_" + str( iAnalysis ) + ".png" )
#   
#         print( "egress animation ..." )
#         plot = animate(trajectory_data, walkable_area, radius=2, every_nth_frame=60)
#         plot.write_html("egress_" + str( iAnalysis ) + ".html")

   fig_AgentsTime.savefig( "agentsVStimes.png" )

   #Print and plot of agent types
   countAgentsPerType.to_csv( 'countAgentsPerType.csv' )
   ax = countAgentsPerType.plot(kind='bar' )
   ax.set_xlabel("Agents types")
   ax.set_ylabel("Agents []")
   fig = ax.get_figure()
   fig.savefig( "agentsTypes.png" )

   evacuationTime.to_csv( 'evacuationTime.csv' )
   print( "AVG evac time: {}".format( evacuationTime.mean()['Evacuation time'] ) )
   print( "STD deviation: {}".format( evacuationTime.std()['Evacuation time'] ) )

   ax = evacuationTime.plot(kind='bar' )
   ax.set_xlabel("Simulations")
   ax.set_ylabel("Evacuation time [s]")
   fig = ax.get_figure()
   fig.savefig( "EvacuationTimeStats.png" )
   
   #      print( "Density and speed ..." )
   #      min_frame_profiles = 45000
   #      max_frame_profiles = 50000
   #
   #      individual_speed = pedpy.compute_individual_speed(
   #          traj_data=trajectory_data,
   #          frame_step=5,
   #          speed_calculation=pedpy.SpeedCalculation.BORDER_SINGLE_SIDED,
   #      )
   #
   #      individual_voronoi_cells = pedpy.compute_individual_voronoi_polygons(
   #          traj_data=trajectory_data,
   #          walkable_area=walkable_area,
   #          cut_off=pedpy.Cutoff(radius=0.8, quad_segments=3),
   #      )
   #
   #      density_profiles, speed_profiles = pedpy.compute_profiles(
   #          individual_voronoi_speed_data=pd.merge(
   #              individual_voronoi_cells[ individual_voronoi_cells.frame.between( min_frame_profiles, max_frame_profiles ) ],
   #              individual_speed[ individual_speed.frame.between( min_frame_profiles, max_frame_profiles ) ],
   #              on=["id", "frame"],
   #          ),
   #          walkable_area=walkable_area.polygon,
   #          grid_size=0.25,
   #          speed_method=pedpy.SpeedMethod.ARITHMETIC,
   #      )
   #
   #      fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
   #
   #      cm = pedpy.plot_profiles(
   #          walkable_area=walkable_area,
   #          profiles=density_profiles,
   #          axes=ax0,
   #          label="$\\rho$ / 1/$m^2$",
   #          vmin=0,
   #          vmax=10,
   #          title="Density",
   #      )
   #
   #      cm = pedpy.plot_profiles(
   #          walkable_area=walkable_area,
   #          profiles=speed_profiles,
   #          axes=ax1,
   #          label="v / m/s",
   #          vmin=0,
   #          vmax=2,
   #          title="Speed",
   #      )
   #
   #      fig.tight_layout(pad=2)
   #      plt.show()

