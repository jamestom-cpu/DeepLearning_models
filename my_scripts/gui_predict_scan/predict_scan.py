import sys
import ast

# Get the parameters from the command line arguments
output_shape = ast.literal_eval(sys.argv[1]) 
patch_shape = ast.literal_eval(sys.argv[2]) 
certainty_level = float(sys.argv[3]) 
stride = ast.literal_eval(sys.argv[4])
mlflow_model_path = sys.argv[5]

# Now perform your machine learning operations using these parameters
print("output_shape: ", output_shape)
print("patch_shape: ", patch_shape)
print("certainty_level: ", certainty_level)
print("stride: ", stride)
print("mlflow_model_path: ", mlflow_model_path)
