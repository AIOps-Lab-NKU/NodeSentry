import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
import ttkbootstrap.dialogs as dialogs
import os
import pandas as pd
import functools
import tkinter.filedialog as tf
from ListPicker import ListPicker  # Dropdown label list
from PlotCanvas2 import PlotApp2  # Label interface
from ListSignalPicker import ListSignalPicker  # Node list
from MultiCanvas import ScrollableCanvasFrame  # Multi-canvas
from SmallListSignalPicker import SmallListSignalPicker  # Small node list

def load_label_job(node_name):
    """
    Used to load job data, now based on the length of time_scope.txt to set is_job_list.
    Default splitting strategy: first half is jobs, second half is no jobs.
    Modify the splitting strategy here to suit your data. After clicking the 'Display Job Presence' button, you can modify whether to display the differences in the intervals.
    """
    # Read timestamp data
    with open(PATH.time_scope, 'r') as f:
        time_scope = f.read()
    
    # Calculate the number of timestamps based on the number of commas (i.e., data length)
    data_length = time_scope.count(',') + 1

    # Initialize is_job_list with the specified length
    is_job_list = [0] * data_length

    # Simple splitting: first half for jobs, second half for no jobs
    half_length = data_length // 2
    for i in range(half_length):
        is_job_list[i] = 1  # Mark the first half of the data as jobs

    return is_job_list

class PATH:
    @classmethod
    def load_all_path(cls):
        cwd = os.getcwd()
        print("cmd : " + cwd)
        cls.cluster_result = cwd + r"\config_files\cluster_result.txt"
        cls.cluster_adjust = cwd + r"\config_files\cluster_adjust.txt"
        cls.reference_cluster_path = cwd + r"\reference_cluster"
        cls.metrics_file_path = cwd + r"\config_files\metric_used.txt"  # Metrics
        cls.label_data_root = cwd + r"\labels"
        cls.time_scope = cwd + r"\config_files\time_scope.txt"
        cls.node_data_root = cwd + r"\node_data"
        cls.reference_root_path = cwd + r"\reference_models"
        cls.communicate_file_path = cwd + r"\communicate_files\annotation_history.txt"

    @classmethod
    def node_data(cls, node):  # Get node data
        return cls.node_data_root + "\\" + node + ".csv"

    @classmethod
    def label_data(cls, node_name):
        return cls.label_data_root + "\\" + node_name

class App:
    reference_color_list = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8633', '#33FFF2']
    ui_colorgroup = {
        "Label_Color1": "#E3F2FD", "Label_Color2": "#BBDEFB", "Label_Color3": "#90CAF9", "Label_Color4": "#64B5F6",
        "Cluster_Color1": "#FFCDD2", "Cluster_Color2": "#EF9A9A", "Cluster_Color3": "#E57373", "Cluster_Color4": "#EF5350", "Cluster_Color5": "#E53935",
        "Line_Color1": "#1E88E5", "Line_Color2": "#3949AB"
    }
    data_shape = None

    def menubar_init(self):
        self.data_root_dir = None

        def open_data_direction():
            dir_path = tf.askdirectory()
            self.data_root_dir = dir_path
            PATH.node_data_root = dir_path
            print(dir_path)
            if self.cluster_root.winfo_ismapped():  # If currently in category visualization module
                self.clusterlist_update()  # Update category list
            else:
                self.nodelist_update()  # Update node list

        self.used_reference_path = [] 
        self.all_reference_path = []

        self.main_menu = tk.Menu(self.root)  # Create menu bar
        self.root.config(menu=self.main_menu)  # Set menu bar

        self.file_menu = tk.Menu(self.main_menu, tearoff=0)  # File menu 
        self.file_menu.add_command(label="Open Folder", command=open_data_direction)  # Open data folder
        self.main_menu.add_cascade(label="File", menu=self.file_menu)  # File

        self.func_menu = tk.Menu(self.main_menu, tearoff=0)  # Function menu

        def resume_label_part():
            self.cluster_root.place_forget()  # Hide category visualization interface
            self.label_root.place(x=0, y=0, height=600, width=1200)  # Display data labeling interface

        self.func_menu.add_command(label="Data Labeling", command=resume_label_part)  # Data labeling

        def cluster_label_part():
            self.label_root.place_forget()  # Hide data labeling interface
            self.cluster_root.place(x=0, y=0, height=600, width=1200)  # Display category visualization interface

        self.func_menu.add_command(label="Category Visualization", command=cluster_label_part)  # Category visualization
        self.main_menu.add_cascade(label="Function", menu=self.func_menu)  # Function

        self.view_menu = tk.Menu(self.main_menu, tearoff=0)  # Display menu
        self.show_time = False
        self.time_scope = []

        def show_time():
            self.show_time = not self.show_time
            self.labelcanvas_update()  # Update label canvas

        self.view_menu.add_command(label="Display Time", command=show_time)
        self.show_job = True

        def show_job():
            self.show_job = not self.show_job
            self.labelcanvas_update() 

        self.view_menu.add_command(label="Display Job Presence", command=show_job)
        self.main_menu.add_cascade(label="Display", menu=self.view_menu)

        self.theme_menu = tk.Menu(self.main_menu, tearoff=0)  # Theme menu
        for theme in self.root.style.theme_names():
            self.theme_menu.add_command(label=theme, command=lambda t=theme: self.change_theme(t))
        self.main_menu.add_cascade(label="Theme", menu=self.theme_menu)

    def change_theme(self, theme):
        self.root.style.theme_use(theme)

    def nodelist_init(self):  # Initialize node list
        self.nodelist_frame = ttk.Frame(self.label_root)  # Create node list frame
        self.nodelist_frame.place(width=200, height=400, x=0, y=0)  # Set node list frame position

    def nodelist_update(self):  # Update node list
        data_name_list = os.listdir(self.data_root_dir)  # Get all filenames in the data folder
        self.data_path_list = [self.data_root_dir + "/" + name for name in data_name_list]  # Get all data file paths
        data_name_list = [name[:-4] for name in data_name_list]  # Remove file extensions

        def btn_cmd():  # Button command
            cur_id = self.nodelist_picker.get_current_selected()  # Get current selected node's index
            print(cur_id)
            self.cur_data_path = self.data_path_list[cur_id]  # Get the data path of the currently selected node
            print(self.cur_data_path)
            self.labelcanvas_update()  # Update label canvas

        self.nodelist_picker = ListSignalPicker(self.nodelist_frame, data_name_list, 0, 0, 200, 400, btn_cmd)  # Create node picker

    def load_time_data(self):  # Load time data
        with open(PATH.time_scope, 'r') as f:
            line = f.read()
            self.time_scope = [d for d in line.split(",")]

    def labelcanvas_init(self):  # Initialize label canvas
        self.job_data = {}
        self.current_selected_metrics = []  # Currently selected metrics
        # Create label canvas
        self.data_canvas = PlotApp2(self.label_root, 200, 0, 1000, 500, self.ui_colorgroup["Line_Color1"], self.ui_colorgroup["Line_Color2"])
        self.cur_data_path = None
        self.load_time_data()
        self.labelcanvas_update()

    def labelcanvas_update(self):
        self.label_list = []
        if self.cur_data_path is None:  # If current data path is empty, return
            return
        current_data_path = self.cur_data_path
        show_metrics = [m.split(":")[1] for m in self.current_selected_metrics]  # Get the currently selected metrics

        data = pd.read_csv(current_data_path)  # Read data
        data = data.loc[:, show_metrics]  # Get the specified metrics
        columns = data.columns  # Get column names
        data = data.to_numpy()  # Convert to numpy array

        node_name = os.path.basename(current_data_path).split(".")[0]  # Get node name
        is_job_list = load_label_job(node_name)  # Use the new splitting strategy

        nojob_range = []
        job_range = []
        start_point = 0  # Start point
        current_value = is_job_list[0]

        for i in range(1, len(is_job_list)):
            if is_job_list[i] != current_value:
                end_point = i
                if current_value == 0:
                    nojob_range.append((start_point, end_point))
                else:
                    job_range.append((start_point, end_point))
                start_point = i
                current_value = is_job_list[i]
        
        # Ensure the last segment is processed
        if current_value == 0:
            nojob_range.append((start_point, len(is_job_list)))
        else:
            job_range.append((start_point, len(is_job_list)))

        # Set instance variables
        self.job_range = job_range
        self.nojob_range = nojob_range

        # Update canvas display
        self.data_canvas.update(data, columns, self.show_job, job_range, nojob_range, self.show_time, self.time_scope)
        self.data_shape = data.shape  # Get data shape

    def referencelist_init(self):
        self.referencelist_frame = ttk.Frame(self.label_root)  # Create reference list frame
        self.referencelist_frame.place(width=200, height=199, x=0, y=401)
        self.reference_root_path = PATH.reference_root_path

        # Manually set display order
        self.reference_name_list = ['modify.py', 'USAD.py', '3-sigma.py']
        self.reference_path_list = [self.reference_root_path + "/modify.py", 
                                    self.reference_root_path + "/USAD.py", 
                                    self.reference_root_path + "/3-sigma.py"]

        def btn_cmd(self, path, i):
            if self.data_shape is None:  # If data shape is empty, return
                return
            else:  # Data shape is not empty
                print("path:", path)
                print("cur_data_path : ", self.cur_data_path)
                os.system(f"python {path} {self.cur_data_path}")
                with open(PATH.communicate_file_path, 'r') as f:
                    lines = f.readlines()
                    if not lines:
                        print("Warning: communicate file is empty")
                        return
                    line = lines[-1]  # Get the last line
                    all_pos = line.split(',')  # Get all anomaly point indices
                    if all_pos[-1][-1] == "\n":  # If the last index's last character is a newline
                        all_pos[-1] = all_pos[-1][:-1]  # Remove the last character from the last index
                    print(all_pos)
                    for pos in all_pos:
                        self.data_canvas.zoom_pan.draw_line(int(pos), col=self.reference_color_list[i], style=":")  # Draw line

        multi_var = tk.IntVar()  # Create integer variable
        self.nodelist_btns = []  # Node button list
        for i, name in enumerate(self.reference_name_list):  # Iterate through the reference model name list
            display_name = os.path.splitext(name)[0]  # Remove file extension
            fun = functools.partial(btn_cmd, self, self.reference_path_list[i], i)  # Create partial function
            btn = ttk.Radiobutton(self.referencelist_frame, text=display_name, command=fun, variable=multi_var, value=i + 1)
            btn.place(width=198, height=30, x=1, y=i * 31)
            self.nodelist_btns.append(btn)

    def reference_cluster_list_init(self):
        self.reference_cluster_frame = ttk.Frame(self.cluster_root)  # Create reference cluster method list frame
        self.reference_cluster_frame.place(width=200, height=199, x=0, y=401)
        self.reference_cluster_path = PATH.reference_cluster_path

        # Manually set display order and include .py suffix
        self.reference_cluster_name_list = ['HAC.py', 'k-means.py', 'DBSCAN.py']
        self.reference_cluster_path_list = [self.reference_cluster_path + "/HAC.py", 
                                            self.reference_cluster_path + "/k-means.py", 
                                            self.reference_cluster_path + "/DBSCAN.py"]

        def btn_cmd(self, path, i):
            print("path:", path)
            os.system(f"python {path}")
            self.clusterlist_update()  # Update category list after running the script
            self.update_cluster_select_metric_picker()  # Update node picker
            self.check_empty_clusters()  # Check for empty clusters

        multi_var = tk.IntVar()  # Create integer variable
        self.clusterlist_btns = []  # Node button list
        for i, name in enumerate(self.reference_cluster_name_list):  # Iterate through the reference model name list
            display_name = os.path.splitext(name)[0]  # Remove file extension
            fun = functools.partial(btn_cmd, self, self.reference_cluster_path_list[i], i)  # Create partial function
            btn = ttk.Radiobutton(self.reference_cluster_frame, text=display_name, command=fun, variable=multi_var, value=i + 1)
            btn.place(width=198, height=30, x=1, y=i * 31)
            self.clusterlist_btns.append(btn)

    def check_empty_clusters(self):
        empty_clusters = []
        for cluster, nodes in self.cluster_to_node_dir.items():
            if len(nodes) == 0:
                empty_clusters.append(cluster)

        if empty_clusters:
            dialogs.Messagebox.show_warning(
                title="Warning",
                message=f"Empty clusters: {', '.join(empty_clusters)}"
            )

    def contorlpanel_init(self):  # Initialize control panel
        self.contorlpanel = ttk.Frame(self.label_root)  # Create the control panel below the label
        self.contorlpanel.place(width=200, height=100, x=1000, y=500)  # Set control panel position

        def savelabel_func():  # Save label function
            labels = self.data_canvas.get_curlabel()
            data_length = self.data_shape[0] if self.data_shape else 0  # Get data length
            label_data = [0] * data_length  # Initialize with 0

            for label in labels:
                pos = int(label.split(':')[1].strip())  # Get the index of the anomaly point
                if "Ann" in label:  # If the label contains 'Ann'
                    label_data[pos] = 1  # Set the label at the index position to 1
                elif "RangeAnn" in label:  # If the label contains 'RangeAnn'
                    label_data[pos] = 1  # Set the label at the index position to 1

            # Ensure the label at index 0 is 0
            if data_length > 0:
                label_data[0] = 0

            # Set the boundary points of job_range and nojob_range to 0
            def set_boundary_to_zero(ranges):
                for r in ranges:
                    if r[0] < data_length:
                        label_data[r[0]] = 0
                    if r[1] < data_length:
                        label_data[r[1]] = 0

            set_boundary_to_zero(self.job_range)
            set_boundary_to_zero(self.nojob_range)

            # Save as CSV file
            csv_file_path = PATH.label_data(self.cur_data_path.split("/")[-1]).replace('.txt', '.csv')
            df = pd.DataFrame({'index': range(data_length), 'label': label_data})
            df.to_csv(csv_file_path, index=False)
            print("Save succeed as CSV")

        savelabel_btn = ttk.Button(self.contorlpanel, text="Save Labels", command=savelabel_func)
        savelabel_btn.place(width=120, height=30, x=50, y=20)

    def metric_select_dropdown_init(self):  # Initialize metrics dropdown
        with open(PATH.metrics_file_path, 'r') as f:
            line = f.read()
            self.all_metrics_list = line.split(",")
        self.metric_select_dropdown = ListPicker(self.label_root, self.all_metrics_list, 220, 500, 100, 350)

        self.refresh_btn = ttk.Button(self.label_root, text="Refresh", style="TButton")
        self.refresh_btn.place(x=600, y=520, height=30, width=100)

        self.select_all_btn = ttk.Button(self.label_root, text="Select All", style="TButton")
        self.select_all_btn.place(x=720, y=520, height=30, width=100)

        self.clear_btn = ttk.Button(self.label_root, text="Clear Metrics", style="TButton")
        self.clear_btn.place(x=840, y=520, height=30, width=100)

        self.current_selected_metrics = []

        def refresh_btn_click(event):
            self.current_selected_metrics = self.metric_select_dropdown.get_current_selected()
            self.labelcanvas_update()

        def select_all_btn_click(event):
            self.current_selected_metrics = [f"{i}:{m}" for i, m in enumerate(self.all_metrics_list)]
            self.metric_select_dropdown.entry.delete(0, "end")
            self.metric_select_dropdown.entry.insert(0, ",".join(self.current_selected_metrics))
            self.labelcanvas_update()

        def clear_btn_click(event):
            self.current_selected_metrics = []
            self.metric_select_dropdown.entry.delete(0, "end")
            self.metric_select_dropdown.listbox.selection_clear(0, tk.END)

        self.refresh_btn.bind("<ButtonPress-1>", refresh_btn_click)
        self.select_all_btn.bind("<ButtonPress-1>", select_all_btn_click)
        self.clear_btn.bind("<ButtonPress-1>", clear_btn_click)

    def metric_select_dropdown_update(self):
        pass

    def label_part_init(self):
        self.nodelist_init()
        self.referencelist_init()
        self.labelcanvas_init()
        self.contorlpanel_init()
        self.metric_select_dropdown_init()

    def label_part_destory(self):
        pass

    def clusterlist_init(self):  # Initialize category list
        self.clusterlist_frame = ttk.Frame(self.cluster_root)
        self.clusterlist_frame.place(width=200, height=600, x=0, y=0)
        self.clusterlist_update()

    def clusterlist_update(self):  # Update category list
        def btn_cmd(cur_cluster_nodes, cur_cluster):
            self.cur_cluster_nodes = cur_cluster_nodes
            self.cur_selected_cluster = cur_cluster  # Currently selected category
            self.clustercanvas_update()  # Update category canvas
            self.clustermodifylist_update()  # Update category modification list
            self.update_cluster_select_metric_picker()  # Update right-side dropdown

        multi_var = tk.IntVar()  # Reset multi_var variable
        self.cluster_to_node_dir = {}

        # Clear old buttons
        if hasattr(self, 'clusterlist_btns'):  # Ensure the attribute exists
            for btn in self.clusterlist_btns:
                btn.destroy()
        self.clusterlist_btns = []  # Reset button list

        # Read the latest category information
        with open(PATH.cluster_result, 'r') as f:
            lines = f.readlines()
            for line in lines:
                elements = line.strip().split('|')
                self.cluster_to_node_dir[elements[0]] = elements[1:]

        # Update category buttons
        for i, k in enumerate(self.cluster_to_node_dir.keys()):
            display_name = f"cluster-{k}"  # Format the category name as cluster-X
            fun = functools.partial(btn_cmd, self.cluster_to_node_dir[k], k)
            btn = ttk.Radiobutton(self.clusterlist_frame, text=display_name, command=fun, variable=multi_var, value=i + 1)
            btn.place(width=198, height=30, x=1, y=i * 31)
            self.clusterlist_btns.append(btn)

    def clustercanvas_init(self):  # Initialize category canvas
        self.cluster_canvas_frame = ttk.Frame(self.cluster_root)  # Create category canvas frame
        self.cluster_canvas_frame.place(width=700, height=600, x=200, y=0)  # Set category canvas frame position
        self.cur_cluster_nodes = []  # Current category nodes, initially an empty list
        self.cur_selected_cluster_node = None  # Currently selected category node, initially None
        self.cur_selected_cluster = None  # Currently selected category, initially None
        self.cur_selected_cluster_metric = 0  # Currently selected category metric, initially 0

    def clustercanvas_update(self):  # Update category canvas
        if not self.cur_cluster_nodes:  # If current category nodes are empty
            return
        path_list = [PATH.node_data(node) for node in self.cur_cluster_nodes]  # Get the data paths of the current category nodes
        # Read metric_data.txt file
        with open(PATH.metrics_file_path, 'r') as f:
            metrics_line = f.read().strip()
            metrics = metrics_line.split(',')
        self.cluster_canvas = ScrollableCanvasFrame(self.cluster_canvas_frame, 0, 0, 700, 550, path_list, metrics=metrics)  # Create scrollable canvas frame
        self.cluster_canvas.populate(self.cur_selected_cluster_metric)

    def clustermodifylist_init(self):  # Initialize category modification list
        col = self.ui_colorgroup["Cluster_Color1"]  # Get color
        self.cur_target_class = 0  # Current target category, initially None
        self.clustermodifylist_frame = ttk.Frame(self.cluster_root, style="TFrame")  # Create category modification list frame
        self.clustermodifylist_frame.place(x=900, y=0, height=110, width=300)  # Set category modification list frame position
        pointer_label = ttk.Label(self.clustermodifylist_frame, text="———>")  # Create pointer label
        pointer_label.place(x=120, y=20)  # Set pointer label position
        self.start_class_label = ttk.Label(self.clustermodifylist_frame, text="From cluster : 0")
        self.start_class_label.place(x=20, y=12, height=20, width=100)
        self.end_class_label = ttk.Label(self.clustermodifylist_frame, text="To cluster : 0")
        self.end_class_label.place(x=180, y=12, height=20, width=100)
        cluster_list = list(self.cluster_to_node_dir.keys())

        def picker_cmd():  # Picker command
            id = self.end_class_picker.get_current_selected()
            self.end_class_label.config(text="To class : " + cluster_list[id])
            self.cur_target_class = cluster_list[id]

        self.end_class_picker = SmallListSignalPicker(self.clustermodifylist_frame, cluster_list, x=180, y=32, w=100, h=50, cmd=picker_cmd)

        def btn_cmd(event):  # Button command
            self.cur_selected_cluster_node = self.cluster_canvas.get_select_label()  # Get the selected category node
            if self.cur_selected_cluster_node is None or self.cur_target_class is None:  # If the selected category node or target category is None, return
                return
            self.cluster_to_node_dir[self.cur_selected_cluster].remove(self.cur_selected_cluster_node)  # Remove the selected category node from the current category node list
            self.cluster_to_node_dir[self.cur_target_class].append(self.cur_selected_cluster_node)  # Add the selected category node to the target category's node list
            self.cluster_node_dir_save()  # Save category node directory
            self.clustercanvas_update()  # Update category canvas
            self.update_cluster_select_metric_picker()  # Update right-side dropdown
            self.cluster_select_metric_picker.update_list(self.cur_cluster_nodes)  # Refresh "Select Node" dropdown

        self.cluster_modify_save_btn = ttk.Button(self.clustermodifylist_frame, text="Save")
        self.cluster_modify_save_btn.place(x=30, y=50, width=70, height=30)
        self.cluster_modify_save_btn.bind("<Button-1>", btn_cmd)

    def clustermodifylist_update(self):  # Update category modification list
        self.start_class_label.config(text="From class : " + self.cur_selected_cluster)

    def cluster_node_dir_save(self):  # Save category node directory
        with open(PATH.cluster_adjust, 'w') as f:  # Open file
            line = ""
            for k in self.cluster_to_node_dir.keys():  # Iterate through the category node directory keys
                line += k  # Add key to line
                for s in self.cluster_to_node_dir[k]:  # Iterate through the category node directory values
                    line += "|"  # Add delimiter
                    line += s  # Add value
                line += "\n"
            f.write(line)  # Write line

    def cluster_select_metric_bar_init(self):  # Initialize category select node bar
        col = self.ui_colorgroup["Cluster_Color4"]
        self.cluster_select_metric_frame = ttk.Frame(self.cluster_root, style="TFrame")  # Create category select node frame
        self.cluster_select_metric_frame.place(x=900, y=110, height=500, width=300)  # Set category select node frame position
        ttk.Label(self.cluster_select_metric_frame, text="Select Node").place(x=25, y=0)

        self.cluster_select_metric_picker = SmallListSignalPicker(self.cluster_select_metric_frame, item_list=self.cur_cluster_nodes if self.cur_cluster_nodes else [], x=20, y=30, w=200, h=400, cmd=self.update_cluster_canvas)

    def update_cluster_select_metric_picker(self):  # Update right-side dropdown
        if hasattr(self, 'cluster_select_metric_picker'):  # If the cluster_select_metric_picker attribute exists
            self.cluster_select_metric_picker.update_list(self.cur_cluster_nodes)  # Update list

    def update_cluster_canvas(self):  # Update category canvas
        id = self.cluster_select_metric_picker.cur_selected_id  # Get the current selected category node's index
        if id is not None and id < len(self.cur_cluster_nodes):  # If the index is not None and less than the current category node length
            self.cur_selected_cluster_node = self.cur_cluster_nodes[id]  # Get the current selected category node
            self.cluster_canvas.cur_selected_id = id  # Update the current selected ID
            self.cluster_canvas.populate(self.cur_selected_cluster_metric)

    def cluster_part_init(self):  # Initialize category part
        self.clusterlist_init()  # Initialize category list
        self.clustercanvas_init()  # Initialize category canvas
        self.clustermodifylist_init()  # Initialize category modification list
        self.cluster_select_metric_bar_init()  # Initialize category select node bar
        self.reference_cluster_list_init()  # Initialize reference cluster list frame

    def __init__(self):
        self.root = tb.Window(themename="journal")  # Create window
        self.root.title("Cluster Adjustment and Anomaly Labeling Software for High-Performance Computing Systems")  # Title
        self.root.geometry(newGeometry="1200x600+300+100")
        self.menubar_init()

        self.label_root = ttk.Frame(self.root)  # Create label root
        self.label_root.place(x=0, y=0, height=600, width=1200)
        self.label_part_init()

        self.cluster_root = ttk.Frame(self.root)
        self.cluster_root.place(x=0, y=0, height=600, width=1200)
        self.cluster_part_init()  # Initialize category part
        self.cluster_root.place_forget()  # Hide category part

        self.job_range = []
        self.nojob_range = []

        # Initialize clusterlist_btns attribute
        self.clusterlist_btns = [] 

    def run(self):
        self.root.mainloop()  # Run window

PATH.load_all_path()
app = App()
app.run()
