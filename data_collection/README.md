# haptics_enhanced_imitation_learning

## Bilateral Teleoperation
Follow the installation instructions from [the bilateral_teleop branch of https://github.com/noahdrakes/sawIntuitiveResearchKit](https://github.com/noahdrakes/sawIntuitiveResearchKit/tree/bilateral_teleop)

```
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
cd src/dvrk/dvrk_config_jhu/jhu-dVRK
ros2 run dvrk_robot dvrk_system -j system-MTMR-PSM2-bilateral-teleop.json -m manager-MTMR-PSM2-bilateral-ros.json -K -C -p 0.001 -e IRE_IPYTHON
run study_gui.py
```
## Force Sensor
Follow the setup instructions from [the sawATIForceSensor repository](https://github.com/jhu-saw/sawATIForceSensor)

## Data Collection
### ROS topics being collected
* Action Space
  * /PSM2/measured_js – measured joint position, velocity, and effort (only use position & velocity)
  * /PSM2/jaw/measured_js – measured jaw position, velocity, and effort (only use position & velocity)
* State Space
  * /PSM2/setpoint_js – commanded joint position, velocity, and effort (only use position & velocity)
  * /PSM2/jaw/setpoint_js – commanded jaw position, velocity, and effort (only use position & velocity)
* PSM Force Topics
  * /PSM2/spatial/measured_cf - spatial measured force/torque
  * /PSM2/body/measured_cf - body measured force/torque
  * /PSM2/gravity_compensation/setpoint_js – commanded gravity compensation
    * used as joint_force = arm_effort – gravity_compensation
    * arm_effort = joint effort from measured_js
* External Force Topics
  * /measured_cf - Cartesian force (x, y, z) | 3 | ATI NetFT
* Wrist Cameras
  * /wrist/cam1/image_raw – wrist camera 1
  * /wrist/cam2/image_raw – wrist camera 2
* Stereo Endoscope
  * /console1/right/image_raw
  * /console1/left/image_raw

### Data collection instructions
1. start publishing stereo camera topics 
```
source /opt/ros/jazzy/setup.bash 
cd ~/ros2_ws  
source install/setup.bash 
ros2 launch dvrk_video decklink_stereo_goovis.launch.py stereo_rig_name:=console1 
```

2. start publishing wrist camera topics
 ```
source dvrk_venv/bin/activate
source /opt/ros/jazzy/setup.bash
cd ~/haptics_enhanced_imitation_learning/data_collection
python wrist_cameras_video.py
```

3. start publishing force sensor topic
```
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws  
source install/setup.bash 
ros2 run ati_ft ati_ft -i 192.168.1.1 -r 0.001
```
* re-bias the sensor at startup

4. start teleop/bilateral teleop
```
  cd ~/ros2_ws
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  cd src/dvrk/dvrk_config_jhu/jhu-dVRK
  ros2 run dvrk_robot dvrk_system -j system-MTMR-PSM2-bilateral-teleop.json -m manager-MTMR-PSM2-bilateral-ros.json -K -C -p 0.001 -e IRE_IPYTHON
  run study_gui.py
```

5. start data collection
* **Option A: PyQt Graphical Interface (Recommended)**
```bash
source dvrk_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/haptics_enhanced_imitation_learning/data_collection
python record_gui.py
```
*You can also launch the GUI via `python record_dvrk_episodes.py --gui`.*

* **Option B: Terminal / CLI Mode (Headless / SSH)**
```bash
source dvrk_venv/bin/activate
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/haptics_enhanced_imitation_learning/data_collection
python record_dvrk_episodes.py --no-gui --psm PSM2 --fps 1000 --image_fps 20 --seconds_per_episode 10
```
Press `q` to start recording a trial. Once the trial finishes, the script will show the capture statistics and prompt you to `Save episode_XXX? [Y/n]` or discard it.