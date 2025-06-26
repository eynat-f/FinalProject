# Project Code

This folder holds the main python scripts we ran, as well as a copy of the used libraries if needed for further exploration.

## Project Structure

- `crazyflie-clients-python/` - A copy of the crazyflie clients library, for flashing and basic control. 
- `crazyflie-lib-python/` A copy of the crazyflie lib library, containing cflib for drone communication. It also contains example code that we utilized as the basis of our flight script.
  * For example, `examples/step-by-step/sbs_motion_commander.py`.
- `code_plot.py` - Our code for plotting the saved audio recordings.
- `code_record.py` - Our code for simultaneously recording and flying the drone (by using two synchronized threads).

