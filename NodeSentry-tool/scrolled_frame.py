from tkinter import *

def data():
    for i in range(100):
        Label(rollFrame, text="my text" + str(i)).pack()

# This is necessary for scrolling
def myfunction(event):
    canvas.configure(scrollregion=canvas.bbox("all"), width=200, height=200)

root = Tk()
root.wm_geometry("800x600")

canvas = Canvas(root)  # Create canvas
canvas.place(x=0, y=0, height=300, width=500)

myscrollbar = Scrollbar(root, orient="vertical", command=canvas.yview)  # Create scrollbar
myscrollbar.place(x=500, y=0, height=300)
canvas.configure(yscrollcommand=myscrollbar.set)

rollFrame = Frame(canvas)  # Create frame on canvas
canvas.create_window((0, 0), window=rollFrame, anchor='nw')  # Use create_window to follow canvas scroll
rollFrame.bind("<Configure>", myfunction)

data()
root.mainloop()
