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
1. start publishing wrist camera topics
 ```
source dvrk_venv/bin/activate
source /opt/ros/jazzy/setup.bash
cd ~/cis2_haptics/CIS2_Haptics_Enhanced_Imitation_Learning/act_dvrk
python wrist_cameras_video.py
```

2. start publishing force sensor topic
```
source /opt/ros/jazzy/setup.bash
cd ~/ros2_ws  
source install/setup.bash 
ros2 run ati_ft ati_ft -i 192.168.1.1
```
* re-bias the sensor at startup

3. start teleop/bilateral teleop
* Starting teleop
```
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
cd src/dvrk/dvrk_config_jhu/jhu-dVRK
ros2 run dvrk_robot dvrk_system -j ~/ros2_ws/src/dvrk/dvrk_config_jhu/jhu-dVRK/system-MTML-MTMR-PSM1-PSM2-Teleop.json
```
* Starting bilateral teleop
```
  cd ~/ros2_ws
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  cd src/dvrk/dvrk_config_jhu/jhu-dVRK
  ros2 run dvrk_robot dvrk_system -j system-MTMR-PSM2-bilateral-teleop.json -m manager-MTMR-PSM2-bilateral-ros.json -K -C -p 0.001 -e IRE_IPYTHON
  run study_gui.py
```

4. start data collection code
* Terminal 1:
```
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
cd src/dvrk/dvrk_config_jhu/jhu-dVRK/sdi
ros2 launch dvrk_console stereo_video_pipeline.launch.py config_dir:=share/config
```
* Terminal 2:
```
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
cd src/dvrk
ros2 run dvrk_data record -c dvrk_data/dvrk_configs/record.json
```
* Terminal 3:
```
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run dvrk_console control_panel
```
