import os
import shutil
import tempfile

import os
import shutil
import tempfile

def clean_mlflow_metrics_efficient(mlflow_directory):
    metrics_directory = os.path.join(mlflow_directory, "metrics")

    if not os.path.exists(metrics_directory):  # check if metrics directory is empty
        print("No metric folder found in directory.")
        # then explore subdirectories
        for subdirectory in os.listdir(mlflow_directory):
            subdirectory_path = os.path.join(mlflow_directory, subdirectory)
            if os.path.isdir(subdirectory_path):  # make sure it's a directory, not a file
                print(f"Exploring subdirectory: {subdirectory_path}")
                clean_mlflow_metrics_efficient(subdirectory_path)
        return

    # Loop through all files in the metrics directory
    for root, dirs, files in os.walk(metrics_directory):
        for file in files:
            filepath = os.path.join(root, file)

            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                # Open each file and read it line by line
                print(f"Cleaning file: {filepath}")
                with open(filepath, 'r') as f:
                    for line in f:
                        values = line.split()
                        # If the line has exactly 3 values, write it to the temporary file
                        if len(values) == 3:
                            temp.write(line.encode())

                temp_filepath = temp.name

            # Replace the original file with the cleaned temporary file
            shutil.move(temp_filepath, filepath)

    print("Cleaned metrics in mlflow directory.")