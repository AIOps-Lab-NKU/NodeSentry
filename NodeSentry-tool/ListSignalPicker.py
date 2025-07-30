import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style

class ListSignalPicker:
    def __init__(self, root, item_list, x, y, w, h, cmd):
        style = Style()  # Using ttkbootstrap style

        self.cmd = cmd  # Callback function after selection
        self.last_selected_id = 0
        self.listbox = tk.Listbox(root, selectmode=tk.SINGLE, selectbackground=style.colors.primary, selectborderwidth=0, bd=2)
        self.listbox.place(x=x, y=y, height=h, width=w)
        self.col = ["#FFFFFF", "#E0E0E0"]  # Modify color values to use white and gray for better contrast
        for i, item in enumerate(item_list):  # Add list items
            self.listbox.insert(tk.END, item)
            self.listbox.itemconfig(i, bg=self.col[i % 2])

        self.listbox.bind("<<ListboxSelect>>", self.show_in_entry)  # Bind selection event

        scrollbar = ttk.Scrollbar(root)
        scrollbar.place(x=x + w, y=y, height=h)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

    def show_in_entry(self, event):  # Callback function for selection event
        cur_selected_id_list = event.widget.curselection()
        if len(cur_selected_id_list) != 0:
            self.cur_selected_id = cur_selected_id_list[0]
            self.listbox.itemconfig(self.last_selected_id, bg=self.col[self.last_selected_id % 2])
            self.listbox.itemconfig(self.cur_selected_id, bg=Style().colors.primary)
            self.last_selected_id = self.cur_selected_id
            self.cmd()

    def get_current_selected(self):  # Get the currently selected index
        return self.cur_selected_id

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("450x400")
    item_list = ["jjer", "edw", "dewf", "dwdf", "hyt", "edw", "dewf", "dwdf", "hyt", "edw", "dewf", "dwdf", "hyt", "edw", "dewf", "dwdf", "hyt", "edw", "dewf", "dwdf", "hyt"]
    
    def cmd1():
        print("ok1")
    
    def cmd2():
        print("ok2")
    
    list_picker1 = ListSignalPicker(root, item_list, 0, 0, 200, 200, cmd1)
    list_picker2 = ListSignalPicker(root, item_list, 202, 0, 200, 200, cmd2)
    
    root.mainloop()
