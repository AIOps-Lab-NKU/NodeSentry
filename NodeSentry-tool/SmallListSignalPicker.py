import tkinter as tk
from tkinter import ttk
from ttkbootstrap import Style

class SmallListSignalPicker:
    def __init__(self, root, item_list, x, y, w, h, cmd):
        style = Style()

        self.cmd = cmd  # Callback function after selection
        self.last_selected_id = 0  # Last selected index
        self.cur_selected_id = None  # Currently selected index
        self.listbox = tk.Listbox(root, selectmode=tk.SINGLE, selectbackground=style.colors.primary, selectborderwidth=0, bd=2)  # Create listbox
        self.listbox.place(x=x, y=y, height=h, width=w)  # Place listbox
        
        for i, item in enumerate(item_list):  # Add list items
            self.listbox.insert(tk.END, item)  # Add list items

        self.listbox.bind("<<ListboxSelect>>", self.show_in_entry)  # Bind selection event

        scrollbar = ttk.Scrollbar(root)  # Create scrollbar
        scrollbar.place(x=x + w, y=y, height=h, width=10)  # Place scrollbar
        self.listbox.config(yscrollcommand=scrollbar.set)  # Configure listbox's scrollbar
        scrollbar.config(command=self.listbox.yview)

    def show_in_entry(self, event):  # Callback function for selection event
        cur_selected_id_list = event.widget.curselection()  # Get currently selected index 
        if len(cur_selected_id_list) != 0:  # If there is a selected index
            self.cur_selected_id = cur_selected_id_list[0]  # Get currently selected index
            self.listbox.itemconfig(self.last_selected_id, bg="white")  # Set last selected background to white
            self.listbox.itemconfig(self.cur_selected_id, bg=Style().colors.primary)  # Set current selected background to primary theme color
            self.last_selected_id = self.cur_selected_id  # Update last selected index
            self.cmd()  # Execute callback function

    def get_current_selected(self):  # Get the currently selected category index, "To class"
        return self.cur_selected_id
    
    def update_list(self, new_items):
        # Clear existing list items
        self.listbox.delete(0, tk.END)
        # Add new list items
        for item in new_items:
            self.listbox.insert(tk.END, item)
        self.cur_selected_id = None  # Reset current selected item
        self.last_selected_id = 0  # Reset last selected item
