# Split Recordings

This folder contains all split recordings of the drone, meaning the splits of the full recordings according to each movement type.
For the old database (until experiment 2.1, including):
- `ascent`- 97 recordings of 3 seconds.
- `away`- 76 recordings, 62 of 5 seconds, 14 of 2.5 seconds.
- `descent`- 97 recordings of 4 seconds.
- `hover`- 304 recordings, 28 recordings of 35 seconds, 276 recordings of 1 second.
- `towards`- 76 recordings, 62 of 5 seconds, 14 of 2.5 seconds.
- `turn`- 69 recordings of 2.5 seconds.

For the new database (from experiment 2.2):
- `ascent`- 97 recordings of 4 seconds.
- `away`- 76 recordings, 62 of 6 seconds, 14 of 3 seconds.
- `descent`- 97 recordings of 4 seconds.
- `hover`- 28 recordings of 34 seconds.
- `towards`- 76 recordings, 62 of 6 seconds, 14 of 3 seconds.
- `turn`- 69 recordings of 3.5 seconds.

## Folder Structure

- `new/` - Data for experiments from 2.2, including the final tests.
- `old/` - Data for experiments up to 2.1 (including).

### Within each folder:

The movement types:
- `ascent/` - Recording of the drone ascending.
  * `move/`, `noise_move/` - Horizontal flights (with and without noise).
    * `horizontal/` - Parallel to the microphone.
    * `vertical/` - Perpendicular to the microphone.
  * `no_move/`, `noise_no_move/` - Recordings of drone idle flight (with and without noise).
- `away/` - Recording of the drone moving away from the microphone.
  * `move/`, `noise_move/` - Horizontal flights (with and without noise).
    * `horizontal/` - Parallel to the microphone.
    * `vertical/` - Perpendicular to the microphone.
- `descent/` - Recording of the drone ascending.
  * `move/`, `noise_move/` - Horizontal flights (with and without noise).
    * `horizontal/` - Parallel to the microphone.
    * `vertical/` - Perpendicular to the microphone.
  * `no_move/`, `noise_no_move/` - Recordings of drone idle flight (with and without noise).
- `hover/` - Recording of the drone levitating.
  * `move/`, `noise_move/` - Horizontal flights (with and without noise).
    * `horizontal/` - Parallel to the microphone.
    * `vertical/` - Perpendicular to the microphone.
  * `no_move/`, `noise_no_move/` - Recordings of drone idle flight (with and without noise).
- `towards/` - Recording of the drone moving towards the microphone.
  * `move/`, `noise_move/` - Horizontal flights (with and without noise).
    * `horizontal/` - Parallel to the microphone.
    * `vertical/` - Perpendicular to the microphone.
- `turn/` - Recording of the drone turning.
  * `move/`, `noise_move/` - Horizontal flights (with and without noise).
    * `horizontal/` - Parallel to the microphone.
    * `vertical/` - Perpendicular to the microphone.

For each original full recording

### For each recording:

- For sub-folders of `ascent/`, `descent/`, `hover/`, `turn/`:
  * `<recording_name>_<movement_type>.wav` - WAV of partial drone flight recording, according to movement type.

- For sub-folders of `away/` and `towards/`:
  * For `new/` : 
    * `<recording_name>_<movement_type>.wav` - WAV of partial drone flight recording, according to movement type.
    * `<recording_name>_<movement_type>1.wav` - First half of WAV movement type recording (reached the microphone in the middle of the movement).
    * `<recording_name>_<movement_type>2.wav` - Second half of WAV movement type recording (reached the microphone in the middle of the movement).
  * For `old/` : 
    * `away/` -
      * `<recording_name>_forward.wav` - WAV of 'forward' movement that was away from the microphone.
      * `<recording_name>_back.wav` - WAV of 'back' movement that was away from the microphone.
      * `<recording_name>_forward_second.wav` - Second part of the WAV of a 'forward' movement that was away from the microphone (microphone in the middle of the movement).
      * `<recording_name>_back_second.wav` - Second part of the WAV of a 'back' movement that was away from the microphone (microphone in the middle of the movement).
    * `towards/` -
      * `<recording_name>_forward.wav` - WAV of 'forward' movement that was towards the microphone.
      * `<recording_name>_back.wav` - WAV of 'back' movement that was towards the microphone.
      * `<recording_name>_forward_first.wav` - First part of the WAV of a 'forward' movement that was towards the microphone (microphone in the middle of the movement).
      * `<recording_name>_back_first.wav` - First part of the WAV of a 'back' movement that was towards the microphone (microphone in the middle of the movement).
