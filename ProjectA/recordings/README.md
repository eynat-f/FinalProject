# Recordings

This folder holds collected recordings' database, and the corresponding logs and plots.

## Project Structure

- `move/` - Drone movement in a predetermined path (no noise).
  * `horizontal/` - Drone moving parallel to the microphone.
  * `vertical/` - Drone moving perpendicular to the microphone.
- `no_fly/` - Recordings without a drone present (no noise).
- `no_move/` - Drone hovering in place (no noise).
- `noise_move/` - Drone movement in a predetermined path (with background noise).
  * `horizontal/` - Drone moving parallel to the microphone.
  * `vertical/` - Drone moving perpendicular to the microphone.
- `noise_no_fly/` - Recordings without a drone present (with background noise).
- `noise_no_move/` - Drone hovering in place (with background noise).

### Within each folder:

- For all folders:
  * `<recording_name>.wav` - WAV audio recording of the drone flight (if present, otherwise a recording of silence or background noise).
  * `<recording_name>_specto.wav` - A spectrogram derived from the WAV recording. Was used to help us gain information regarding the audio patterns from observing the plots.

- Also, For folders containing drone movement: 
  * `<recording_name>.csv` - CSV of the drone's positions throughout the flight, extracted using the drone's logging ability. 
  * `<recording_name>_path_x.csv` - A plot of the drone's path in the x-axis using the CSV logs.
  * `<recording_name>_path_x_z.csv` - A plot of the drone's 3D path in the x-z axes using the CSV logs.
  * `<recording_name>_path_z.csv` - A plot of the drone's path in the z-axis using the CSV logs.
  * `split/` - Split recordings from each full recording, based on the movement commands of the drone throughout the flight.
    * `<recording_name>_<movement_type>.wav` - For each movement type. 
    These will be re-organized by movement type (for easier labeling) and reworked further in Project B.
  

### Important to Note:

Not all data was showcased within the paper, as some was used only in the beginning steps of Project B.
For example, the `no_fly/` and `noise_no_fly/` recordings were only used to assess the general ability of detecting whether a drone is present in the recording. 
As this was accomplished easily with even the simplest classifier, there was no longer any need for these recordings.