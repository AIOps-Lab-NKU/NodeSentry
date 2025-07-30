import os
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from functools import partial
from matplotlib import font_manager as fm, rcParams
from PlotCanvas2 import ZoomPan  # Import ZoomPan class
import tkinter.font as tkFont  # Import font module
import pandas as pd
import json

class ScrollableCanvasFrame:
    def __init__(self, root, x, y, w, h, path_list, metrics):
        self.path_list = path_list  # List of file paths
        self.metrics = metrics  # Metrics passed from main.py

        self.main_frame = ttk.Frame(root, style="TFrame")
        self.main_frame.place(x=x, y=y, width=w, height=h)

        self.canvas = tk.Canvas(self.main_frame)
        self.frame = ttk.Frame(self.canvas, style="TFrame")
        self.vsb = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)

        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas_frame = self.canvas.create_window((4, 4), window=self.frame, anchor="nw", tags="self.frame")

        self.frame.bind("<Configure>", self.onFrameConfigure)
        self.canvas.bind("<Configure>", self.onCanvasConfigure)
        self.last_click_label = None
        self.cur_selected_label = None
        self.cur_selected_id = None

    def path_to_file_name(self, path):  # Extract file name from path
        return path.split("\\")[-1].split(".")[0]

    def populate(self, metric_id):
        if not self.path_list:
            return

        data = [self.load_data_from_path(path) for path in self.path_list]
        metrics = data[0].columns[2:]  # Assume the first two columns are non-metric columns

        # No need to filter, use the passed metrics directly
        metrics = [metric for metric in metrics if metric in self.metrics]

        max_length = max(len(d) for d in data)
        aligned_data = []

        for d in data:
            aligned = pd.DataFrame(index=range(max_length), columns=d.columns)
            aligned.iloc[:len(d)] = d.values
            aligned_data.append(aligned)

        for i, metric in enumerate(metrics):
            sub_frame = ttk.Frame(self.frame, borderwidth=0, relief="flat", style="TFrame")

            fig = Figure(figsize=(20, 5))
            ax = fig.add_subplot(1, 1, 1)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)  # Show gridlines for both axes
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)  # Hide ticks and labels

            # Show y-axis labels
            ax.tick_params(left=True, bottom=False, labelleft=True, labelbottom=False)

            for j, metric_data in enumerate(aligned_data):
                metric_values = pd.to_numeric(metric_data[metric], errors='coerce')  # Convert to numeric type, NaN for invalid values
                ax.plot(metric_data.index, metric_values, color='blue')

            if self.cur_selected_id is not None:
                selected_data = aligned_data[self.cur_selected_id]
                selected_values = pd.to_numeric(selected_data[metric], errors='coerce')  # Convert to numeric type, NaN for invalid values
                ax.plot(selected_data.index, selected_values, color='red')

            fig.subplots_adjust(hspace=0.2)  # Adjust space between subplots
            fig.tight_layout(pad=5)  # Padding around figure

            canvas = FigureCanvasTkAgg(fig, master=sub_frame)
            canvas.draw()
            canvas.get_tk_widget().config(width=670, height=150)  # Increase subplot size
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # Fill subplot

            def click_label(event, id):
                if self.last_click_label is not None:
                    self.last_click_label.config(style="TLabel")

                self.last_click_label = event.widget
                self.last_click_label.config(style="TLabelframe.Label")
                self.cur_selected_id = id
                self.populate(metric_id)

            cmd = partial(click_label, id=i)
            sub_frame.grid(row=i, column=0, padx=5, pady=0)

    def load_data_from_path(self, path):  # Load data from path
        data = pd.read_csv(path)  # Read CSV file
        return data  # Return data

    def get_select_label(self):  # Get the currently selected label
        if self.cur_selected_id is None:  # If no label is selected
            return None
        return self.path_to_file_name(self.path_list[self.cur_selected_id])  # Return the selected label

    def onFrameConfigure(self, event=None):  # Configure frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))  # Configure canvas

    def onCanvasConfigure(self, event):  # Configure canvas
        canvas_width = event.width  # Get canvas width
        self.canvas.itemconfig(self.canvas_frame, width=canvas_width)  # Configure canvas
