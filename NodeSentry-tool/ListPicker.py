import tkinter as tk
from tkinter import ttk

class ListPicker:
    def __init__(self, root, item_list, x, y, h, w):
        self.last_selected_id = []

        self.listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, bg="#F0F0F0", fg="#000000", font=("Helvetica", 12), relief="flat")
        self.listbox.place(x=x, y=y+20, height=h-20, width=w)
        for i, item in enumerate(item_list):
            self.listbox.insert(tk.END, str(i) + ":" + item)

        self.entry = tk.Entry(root, bg="#FFFFFF", fg="#000000", font=("Helvetica", 12), relief="flat")
        self.entry.place(x=x, y=y, height=19, width=w)

        def show_in_entry(event):
            cur_selected_id = event.widget.curselection()
            if len(cur_selected_id) != 0:
                cur_selected_id = list(cur_selected_id)
                self.entry.delete(0, "end")
                for i in cur_selected_id:
                    self.entry.insert(tk.END, str(i) + ":" + item_list[i] + ",")

        self.listbox.bind("<<ListboxSelect>>", show_in_entry)

        scrollbar = ttk.Scrollbar(root)
        scrollbar.place(x=x + w, y=y, height=h)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

    def get_current_selected(self):
        return self.entry.get().split(",")[:-1]