# ENPH 479 Reinforcement Learning Agent Training

## Installation
Refer to [INSTALL.md](INSTALL.md)

To train an AI go to directory where gym-gazebo is contained, then run:
```
source gym_gazebo/envs/ros_ws/devel/setup.bash
cd examples/gazebo_marblemaze  
python gazebo_marblemaze_v0_xentropy.py
```

### Killing background processes

If some processes remain on after ending the simulation run
```
killjim
```
