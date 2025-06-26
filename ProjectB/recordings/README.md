# Full Recordings

This folder contains all full recordings of the drone.
There are 97 recordings in total, 28 of which are of a horizontal flight, and the remaining 69 are of an idle flight.
Each horizontal flight is around 22.5 seconds long, and the idle flights are about 41 seconds long.
This means a little under an hour of drone flight recordings.

## Folder Structure

- `move/` - Recordings of drone moving horizontally.
  * `horizontal/` - Drone moving parallel to the microphone.
  * `vertical/` - Drone moving perpendicular to the microphone.
- `no_move/` - Recordings of drone idle flight.
- `noise_move/` - Recordings of drone moving horizontally, with background noise.
  * `horizontal/` -Drone moving parallel to the microphone.
  * `vertical/` - Drone moving perpendicular to the microphone.
- `noise_no_move/` - Recordings of drone idle flight, with background noise.

### Within each folder:

For each recording:
- `<recording_name>.wav` - WAV of drone flight recording.
- `<recording_name>.csv` - CSV of logged drone location.
- `<recording_name>_path_x.png` - Plot of logged drone x-axis path.
- `<recording_name>_path_x_z.png` - Plot of logged drone x-z-axes 3D path.
- `<recording_name>_path_z.png` - Plot of logged drone z-axis path.
