import tkinter as tk
from tkinter import font as tk_font
from tkinter import simpledialog
import subprocess

default_model_path = mlflow_model_path = "/workspace/mlflow/378794452446859122/ffe6bb4b8c3845ef937a32ccd390640f/artifacts/models"
patch_shape = (30, 30)
certainty_level = 0.2
stride = (1e-3, 2e-3)
output_shape = (31,25)



class ParameterDialog(simpledialog.Dialog):
    def __init__(self, parent):
        super().__init__(parent, "Parameter Dialog")
        self.tk.call('wm', 'geometry', self._root, '800x600')  # larger window
        # define fontsize
        

    def body(self, master):
        master.option_add('*Font', 'Helvetica 24')  # larger font for all widgets
        self.custom_font = tk_font.Font(size=20)
        tk.Label(master, text="output_shape: ", font=self.custom_font).grid(row=0)
        tk.Label(master, text="patch_shape: ").grid(row=1)
        tk.Label(master, text="certainty_level: ").grid(row=2)
        tk.Label(master, text="stride: ").grid(row=3)
        tk.Label(master, text="mlflow_model_path: ").grid(row=4)

        self.e1 = tk.Entry(master)
        self.e2 = tk.Entry(master)
        self.e3 = tk.Entry(master)
        self.e4 = tk.Entry(master)
        self.e5 = tk.Entry(master)

        self.e1.grid(row=0, column=1)
        self.e2.grid(row=1, column=1)
        self.e3.grid(row=2, column=1)
        self.e4.grid(row=3, column=1)
        self.e5.grid(row=4, column=1)

        # Set default values
        self.e1.insert(0, str(output_shape))
        self.e2.insert(0, str(patch_shape))
        self.e3.insert(0, str(certainty_level))
        self.e4.insert(0, str(stride))
        self.e5.insert(0, default_model_path)

        return self.e1  # initial focus

    def apply(self):
        self.output_shape = eval(self.e1.get())
        self.patch_shape = eval(self.e2.get())
        self.certainty_level = float(self.e3.get())
        self.stride = eval(self.e4.get())
        self.mlflow_model_path = self.e5.get()
        
        # Here we pass the parameters to the predict_scan.py script via command line arguments
        subprocess.call(['python', '/workspace/my_scripts/gui_predict_scan/predict_scan.py', 
                        str(self.output_shape), 
                        str(self.patch_shape),
                        str(self.certainty_level),
                        str(self.stride),
                        self.mlflow_model_path])

def main():
    root = tk.Tk()
    dialog = ParameterDialog(root)
    dialog.geometry('800x600')  # larger window
    root.mainloop()

if __name__ == "__main__":
    main()