''' NOTE: DO NOT RUN
    This file contains code to alter the csv files obtained during project A to fit with our current requirements.
    The functions were run once and must not be run again (else the data will be changed)! '''
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydub import AudioSegment  # for trimming wav

import plot_utils as pu


def fix_csv(csv_path, flight_type):
    """
        Clean the csv, and adjust their data
        (towards/away and hover, as elaborated in Experiment 0 and Experiment 2.2).

    Args:
        csv_path: path to file
        flight_type: the flight type of the recording
                    (a string if a parallel/perpendicular flight, None for idle flight).

    """
    df = pd.read_csv(csv_path, header=None,
                     names=["x_axis", "y_axis", "z_axis", "time", "drone_state_code", "task_name"])

    # Extract first row values for x,y,z axes
    first_x, first_y, first_z = df.iloc[0][["x_axis", "y_axis", "z_axis"]]

    # Generate missing ascent data (0 to first_z over 3 seconds, sample rate = 10 ms)
    time_steps = np.arange(0, 3, 0.01)
    z_values = np.linspace(0, first_z, len(time_steps))
    ascent_rows = pd.DataFrame({
        "x_axis": first_x,  # Copy from first row
        "y_axis": first_y,  # Copy from first row
        "z_axis": z_values,
        "time": time_steps,
        "drone_state_code": 30,  # Always 30
        "task_name": "ascent"
    })

    # Prepend ascent rows to the original dataframe
    df = pd.concat([ascent_rows, df], ignore_index=True)

    # Add 3 to the rest of the time steps to account for the missing ascent period
    df.loc[len(ascent_rows):, "time"] += 3

    # Adjust task_name based on flight_type
    if flight_type in ["mic_start", "mic_end"]:
        forward_label = "towards" if "mic_end" else "away"
        back_label = "away" if "mic_end" else "towards"
        df.loc[df["task_name"] == "forward", "task_name"] = forward_label
        df.loc[df["task_name"] == "back", "task_name"] = back_label

    # mic in the middle: split the movements
    elif flight_type == "mic_middle":
        # Find midpoints for each forward/back segment
        for task in ["forward", "back"]:
            task_indices = df[df["task_name"] == task].index
            if not task_indices.empty:
                midpoint = len(task_indices) // 2
                df.loc[task_indices[:midpoint], "task_name"] = "towards"
                df.loc[task_indices[midpoint:], "task_name"] = "away"

    if flight_type != "mic_end":  # if mic is at the end, x values stay the same!

        # Adjust x_axis based on the new labels using delta changes
        df["x_delta"] = df["x_axis"].diff().fillna(0)  # Compute delta change in x_axis
        # Adjust x_axis for 'towards' and 'away'
        df["adjusted_x"] = df["x_axis"]  # Start with the original x_axis

        if flight_type == "mic_start":
            for i in range(1, len(df)):
                df.loc[i, "adjusted_x"] = df.loc[i - 1, "adjusted_x"] - df.loc[i, "x_delta"]

        else:  # flight_type == "mic_middle":
            # Track the drone relative to mic.
            first_away = False
            turn = False
            second_away = False

            # Apply delta changes based on task
            for i in range(1, len(df)):
                task = df.loc[i, "task_name"]

                if not first_away and task == "away":
                    first_away = True

                if not turn and task == "turn":
                    turn = True

                if turn and task == "away":
                    second_away = True

                sign_plus = (not first_away) or second_away

                if sign_plus:
                    df.loc[i, "adjusted_x"] = df.loc[i - 1, "adjusted_x"] + df.loc[i, "x_delta"]
                else:
                    df.loc[i, "adjusted_x"] = df.loc[i - 1, "adjusted_x"] - df.loc[i, "x_delta"]

        # Replace the original x_axis with adjusted_x
        df["x_axis"] = df["adjusted_x"]
        df.drop(columns=["x_delta", "adjusted_x"], inplace=True)  # Remove the auxiliary columns

    # Remove unnecessary columns
    df.drop(columns=["y_axis", "drone_state_code"], inplace=True)

    # Replace done with descent
    df["task_name"] = df["task_name"].replace("done", "descent")

    # Replace sleep with hover
    df["task_name"] = df["task_name"].replace("sleep", "hover")

    # Overwrite the current file with the modified dataframe
    df.to_csv(csv_path, index=False)

    # update hovers to real task (if not idle flight)
    if flight_type is not None:
        df = pd.read_csv(csv_path)
        previous_task = None
        for idx, task in enumerate(df["task_name"]):
            if task == "hover" and previous_task is not None:
                df.at[idx, "task_name"] = previous_task
            else:
                previous_task = task

        df.to_csv(csv_path, index=False)
        df = pd.read_csv(csv_path)

        if flight_type == "mic_middle":
            i = 0
            while i < len(df):
                # Find start of "towards" sequence
                if df.at[i, "task_name"] == "towards":
                    start_towards = i
                    while i < len(df) and df.at[i, "task_name"] == "towards":
                        i += 1

                    # Find start of "away" sequence (immediately after "towards")
                    if i < len(df) and df.at[i, "task_name"] == "away":
                        while i < len(df) and df.at[i, "task_name"] == "away":
                            i += 1
                        end_away = i - 1  # Last "away" index

                        # Balance the sequences
                        midpoint = (start_towards + end_away) // 2

                        # Adjust labels
                        df.loc[start_towards:midpoint, "task_name"] = "towards"
                        df.loc[midpoint + 1:end_away, "task_name"] = "away"

                else:
                    i += 1  # Move forward if not part of "towards-away" sequence

            # Save the final version
            df.to_csv(csv_path, index=False)


def split_wav(input, stem, split_times, titles, flight_type):
    """
        Split the wav according to movement types.

    Args:
        input: input wav file
        stem: path stem
        split_times: times (in ms) to split the wav
        titles: titles for each section
        flight_type: the flight type of the recording
                    (a string if a parallel/perpendicular flight, None for idle flight).

    """
    split_path = Path(pu.project_root / "split")
    split_path.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(input)
    start_time = 0

    for i, end_time in enumerate(split_times):
        split_audio = audio[start_time:end_time]
        if flight_type is not None:
            two_parents = input.parent.parts[-2:]
            if flight_type == "mic_middle":
                if "towards" in titles[i]:
                    full_path = Path(split_path, "towards", *two_parents, f"{stem}_{titles[i]}.wav")
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    split_audio.export(str(full_path), format="wav")
                elif "away" in titles[i]:
                    full_path = Path(split_path, "away", *two_parents, f"{stem}_{titles[i]}.wav")
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    split_audio.export(str(full_path), format="wav")
                else:
                    full_path = Path(split_path, titles[i], *two_parents, f"{stem}_{titles[i]}.wav")
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    split_audio.export(str(full_path), format="wav")
            else:
                full_path = Path(split_path, titles[i], *two_parents, f"{stem}_{titles[i]}.wav")
                full_path.parent.mkdir(parents=True, exist_ok=True)
                split_audio.export(str(full_path), format="wav")
        else:   # idle
            parent = input.parent.name
            full_path = Path(split_path, titles[i], parent, f"{stem}_{titles[i]}.wav")
            full_path.parent.mkdir(parents=True, exist_ok=True)
            split_audio.export(str(full_path), format="wav")
        start_time = end_time

    # last segment
    if start_time < len(audio):
        split_audio = audio[start_time:]
        if flight_type is not None:
            two_parents = input.parent.parts[-2:]
            full_path = Path(split_path, "descent", *two_parents, f"{stem}_descent.wav")
            full_path.parent.mkdir(parents=True, exist_ok=True)
            split_audio.export(str(full_path), format="wav")
        else:
            parent = input.parent.name
            full_path = Path(split_path, "descent", parent, f"{stem}_descent.wav")
            full_path.parent.mkdir(parents=True, exist_ok=True)
            split_audio.export(str(full_path), format="wav")


if __name__ == '__main__':

    flag_run = False

    if flag_run:
        recordings_path = Path(pu.project_root / "recordings")
        data_folder_path = Path(pu.project_root / "data/")
        data_folder_path.mkdir(parents=True, exist_ok=True)
        with open(data_folder_path / "mic_position_map.json", "r") as f:
            pos_labels = json.load(f)

        for file in recordings_path.rglob('*.wav'):
            file_name = file.stem.split("_")[0]
            flight_type = pos_labels.get(file_name)
            if flight_type in ["mic_start", "mic_end"]:
                split_times = [4000, 10000, 13500, 19500]
                titles = (
                    ["ascent", "towards", "turn", "away"]
                    if flight_type == "mic_start"
                    else ["ascent", "away", "turn", "towards"]
                )
            elif flight_type == "mic_middle":
                split_times = [4000, 7000, 10000, 13500, 16500, 19500]
                titles = ["ascent", "towards1", "away1", "turn", "towards2", "away2"]
            else:  # idle:
                split_times = [4000, 38000]
                titles = ["ascent", "hover"]
            split_wav(file, file.stem, split_times, titles, flight_type)

        for file in recordings_path.rglob("*.csv"):
            file_name = file.stem.split("_")[0]
            flight_type = pos_labels.get(file_name)
            #fix_csv(file, flight_type)
            if flight_type is None:
                df = pd.read_csv(file)
                df.loc[300:399, "task_name"] = "ascent"
                df.to_csv(file, index=False)
