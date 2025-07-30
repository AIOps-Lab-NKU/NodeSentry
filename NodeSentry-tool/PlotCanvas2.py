import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import gc

class ZoomPan:
    def __init__(self, fig, axs, line_color1, line_color2, existing_annotations=[]):
        self.cur_label = []
        self.axs = axs
        self.fig = fig
        self.line_color1, self.line_color2 = line_color1, line_color2

        self.x0, self.y0 = None, None
        self.pressed = False
        self.region_start = None
        self.region_end = None
        self.remove_start = None
        self.remove_end = None

        self.fig.canvas.mpl_connect('scroll_event', self.zoom)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)

        # Add existing annotations to cur_label
        for annotation in existing_annotations:
            self.annotate_position(annotation)

    def zoom(self, event):
        cur_xlims = [ax.get_xlim() for ax in self.axs]
        direction = 1 if event.button == 'up' else -1
        xdata = event.xdata
        if xdata is None:
            return
        scale_factor = 1.1 if direction > 0 else 0.9
        for i, ax in enumerate(self.axs):
            ax.set_xlim([xdata - (xdata - cur_xlims[i][0]) / scale_factor, xdata + (cur_xlims[i][1] - xdata) / scale_factor])
        self.fig.canvas.draw()

    def on_press(self, event):
        if event.button == 1:
            self.x0, self.y0 = event.xdata, event.ydata
            self.pressed = True
        elif event.button == 2:
            self.set_region(event)
        elif event.button == 3:
            self.set_remove_region(event)
    
    def on_release(self, event):
        self.pressed = False
        self.x0, self.y0 = None, None

    def on_move(self, event):
        if not self.pressed or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self.x0
        self.x0 = event.xdata
        cur_xlims = [ax.get_xlim() for ax in self.axs]
        for i, ax in enumerate(self.axs):
            ax.set_xlim([cur_xlims[i][0] - dx, cur_xlims[i][1] - dx])
        self.fig.canvas.draw()

    def annotate_position(self, pos):
        pos_int = round(pos)
        self.cur_label.append(f"Ann:{pos_int}\n")
        self.draw_line(pos_int, self.line_color1, '--')

    def annotate_range(self, start, end):
        start_int, end_int = round(start), round(end)
        for pos in range(start_int, end_int + 1):
            self.cur_label.append(f"RangeAnn:{pos}\n")
            self.draw_line(pos, self.line_color1, '--')

    def remove_annotation(self, start, end):
        start_int, end_int = round(start), round(end)
        self.cur_label = [label for label in self.cur_label if not any(start_int <= round(int(label.split(':')[1])) <= end_int for label in self.cur_label)]
        for ax in self.axs:
            lines_to_remove = [line for line in ax.lines if start_int <= round(line.get_xdata()[0]) <= end_int and line.get_gid() == 'annotation']
            for line in lines_to_remove:
                ax.lines.remove(line)
        self.fig.canvas.draw()

    def set_region(self, event):
        if self.region_start is None:
            self.region_start = round(event.xdata)
            print(f"Region start set at {self.region_start}")
        else:
            self.region_end = round(event.xdata)
            print(f"Region end set at {self.region_end}")
            self.annotate_range(self.region_start, self.region_end)
            self.region_start, self.region_end = None, None

    def set_remove_region(self, event):
        if self.remove_start is None:
            self.remove_start = round(event.xdata)
            print(f"Remove region start set at {self.remove_start}")
        else:
            self.remove_end = round(event.xdata)
            print(f"Remove region end set at {self.remove_end}")
            self.remove_annotation(self.remove_start, self.remove_end)
            self.remove_start, self.remove_end = None, None

    def draw_line(self, pos, col, style):
        for ax in self.axs:
            ax.axvline(pos, color=col, linestyle=style, gid='annotation')
        self.fig.canvas.draw()

    def get_all_annotations(self):
        annotations = []
        for ax in self.axs:  # Iterate through all subplots
            for line in ax.lines:  # Iterate through all lines in the subplot
                pos = round(line.get_xdata()[0])  # Get the x-coordinate of the line
                if f"Ann:{pos}\n" not in annotations and f"RangeAnn:{pos}\n" not in annotations:  # If the coordinate is not in annotations
                    annotations.append(f"Ann:{pos}\n")
        return annotations

class PlotApp2:
    def __init__(self, root, x_pos, y_pos, w, h, line_color1, line_color2):
        style = Style()
        
        def on_canvas_resize(event):
            nonlocal self 
            self.out_canvas.configure(scrollregion=self.out_canvas.bbox("all")) 
        self.root = root
        self.x_pos, self.y_pos, self.w, self.h = x_pos, y_pos, w, h
        self.line_color1, self.line_color2 = line_color1, line_color2
        self.job_nojob_color_config = {"job": ["darkgrey", "lightgrey"], "nojob": ["cyan", "yellow"]}

        self.frame = ttk.Frame(root, style="TFrame")
        self.frame.place(x=x_pos, y=y_pos, width=w, height=h)

        self.out_canvas = tk.Canvas(self.frame)
        self.out_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.out_canvas.yview)  # Create vertical scrollbar
        scrollbar.pack(side=tk.RIGHT, fill="y")  # Place scrollbar on the right

        self.out_canvas.configure(yscrollcommand=scrollbar.set)  # Set canvas's vertical scrollbar
        self.out_canvas.bind("<Configure>", on_canvas_resize)  # Bind canvas resize event

        self.chart_frame = ttk.Frame(self.out_canvas)
        self.out_canvas.create_window((0, 0), window=self.chart_frame, anchor="nw")

        self.fig = Figure(figsize=(9.6, 10)) 
        self.axs = []

        canvas_agg = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        canvas_agg.draw() 
        canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True) 

        self.zoom_pan = ZoomPan(self.fig, self.axs, self.line_color1, self.line_color2)

    def update(self, data, columns, display_job_flag, job_range, nojob_range, display_time_flag, timestep_data, existing_annotations=[]):
        self.existing_annotations = existing_annotations

        # Clear existing figures and redraw
        self.fig.clf()
        step_num, metrics_num = data.shape[0], data.shape[1]

        self.frame = ttk.Frame(self.root, style="TFrame")  # Create a new frame
        self.frame.place(x=self.x_pos, y=self.y_pos, width=self.w, height=self.h)  # Place frame

        self.out_canvas = tk.Canvas(self.frame)
        self.out_canvas.pack(side=tk.LEFT, fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.out_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")

        self.out_canvas.configure(yscrollcommand=scrollbar.set) 
        self.out_canvas.bind("<Configure>", lambda event: self.out_canvas.configure(scrollregion=self.out_canvas.bbox("all")))

        self.chart_frame = ttk.Frame(self.out_canvas)
        self.out_canvas.create_window((0, 0), window=self.chart_frame, anchor="nw")

        # Increase chart total height
        self.fig = Figure(figsize=(9.6, metrics_num * 1.5))
        self.axs = []
        canvas_agg = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        canvas_agg.draw()
        canvas_agg.get_tk_widget().pack(side=tk.TOP, fill=BOTH, expand=True)
        gc.collect()

        self.zoom_pan = ZoomPan(self.fig, self.axs, self.line_color1, self.line_color2, existing_annotations)
        gc.collect()
        self.axs = []
        for i in range(metrics_num):
            self.axs.append(self.fig.add_subplot(metrics_num, 1, i + 1))
            self.axs[i].tick_params(labelleft=True)  # Show y-axis labels

        x = [i for i in range(step_num)]
        
        for i, ax in enumerate(self.axs):
            if not display_job_flag:
                ax.plot(x, list(data[:, i]), label="" , marker='o', markersize=1, lw=1, color="steelblue", markerfacecolor="blue", markeredgecolor='blue')
            else:
                for nojob in nojob_range:
                    ax.plot(x[nojob[0]:nojob[1]], list(data[nojob[0]:nojob[1], i]), label="" if nojob == nojob_range[0] else "", marker='o', markersize=1, lw=1, color="blue", markerfacecolor="blue", markeredgecolor='blue')
                for job in job_range:
                    ax.plot(x[job[0]:job[1]], list(data[job[0]:job[1], i]), label="" if job == job_range[0] else "", marker='o', markersize=1, lw=1, color="purple", markerfacecolor="purple", markeredgecolor='purple')
            ax.grid(True)
            if display_time_flag:
                step = max(1, len(timestep_data) // 10)  # Adjust step size based on data length to avoid overlap
                ax.set_xticks(np.arange(0, len(timestep_data), step=step))  
                ax.set_xticklabels([timestep_data[j] for j in np.arange(0, len(timestep_data), step=step)], rotation=0, ha="center")

            # Hide the legend completely, originally was ax.legend()
            ax.legend().set_visible(False)

            # Set initial x-axis range to zoom in by default 20x
            xlim = ax.get_xlim()
            ax.set_xlim(xlim[0], xlim[1] / 1)

        # Adjust spacing between subplots
        self.fig.subplots_adjust(left=0.05, right=0.99, top=0.99, bottom=0.1, hspace=0.5)
        self.fig.canvas.draw()
        self.zoom_pan = ZoomPan(self.fig, self.axs, self.line_color1, self.line_color2, self.existing_annotations)
        gc.collect()

    def initialize_annotations(self):
        # Iterate through all existing annotations and draw on the plot
        for annotation in self.existing_annotations:
            self.zoom_pan.draw_line(annotation, self.line_color1, '--')

    def get_curlabel(self):
        return self.zoom_pan.get_all_annotations()

    def get_all_labels(self):
        # Assuming labels is a list containing all annotation information
        return self.labels

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("1000x500")
    app = PlotApp2(root, 0, 0, 1000, 500, 'green', 'blue')
    data = np.random.rand(100, 2)
    columns = ["Metric 1", "Metric 2"]
    existing_annotations = [20, 50, 75]
    app.update(data, columns, False, [], False, [], existing_annotations)
    root.mainloop()
