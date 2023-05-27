# GazeML-RTGazeEstimation
This software is designed to accurately track and analyze a driver's eye movements in real-time, leverages advanced computer vision and gaze estimation algorithms to determine the direction and focus of a driver's gaze. With its primary intention for integration within smart car cabins, the software aims to enhance driver safety and situational awareness by either alerting drivers or initiating appropriate countermeasures when their attention drifts away from the road.

See the demo video [here](https://youtu.be/SfrehL-qcjc).

## Running the demo

To run the webcam demo, perform the following:

    cd src
    python3 elg_demo.py

To see available options, please run ```python3 elg_demo.py --help``` instead.

## Structure

- ```outputs/``` - any output for a model will be placed here, including logs, summaries, and checkpoints.
- ```src/``` - all source code.
  - ```core/``` - base classes
  - ```datasources/``` - routines for reading and preprocessing entries for training and testing
  - ```models/``` - neural network definitions
  - ```util/``` - utility methods
