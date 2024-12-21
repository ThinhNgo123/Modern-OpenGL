import tkinter as tk
import tkinter.ttk as ttk
from threading import Thread

from shared_variables import SharedVariables

# Widgets
# Label, button, entry, text, checkbutton, 
# radiobutton, listbox, canvas, frame, scrollbar, 
# scale, menu, combobox, progressbar

'''
1. Label
Hiển thị văn bản hoặc hình ảnh tĩnh.

    from tkinter import *
    root = Tk()
    label = Label(root, text="Hello, Tkinter!")
    label.pack()
    root.mainloop()

2. Button
Tạo các nút bấm có thể gán hành động.

    button = Button(root, text="Click me", command=lambda: print("Button clicked"))
    button.pack()

3. Entry
Nhập dữ liệu văn bản một dòng.

    entry = Entry(root)
    entry.pack()

4. Text
Nhập và hiển thị văn bản nhiều dòng.

    text = Text(root, height=5, width=40)
    text.pack()

5. Checkbutton
Cho phép lựa chọn nhiều tùy chọn dạng hộp kiểm.

    var = IntVar()
    check = Checkbutton(root, text="Option 1", variable=var)
    check.pack()

6. Radiobutton
Lựa chọn một trong nhiều tùy chọn.

    var = StringVar(value="Option1")
    radio1 = Radiobutton(root, text="Option 1", variable=var, value="Option1")
    radio2 = Radiobutton(root, text="Option 2", variable=var, value="Option2")
    radio1.pack()
    radio2.pack()

7. Listbox
Hiển thị danh sách các mục.

    listbox = Listbox(root)
    listbox.insert(1, "Item 1")
    listbox.insert(2, "Item 2")
    listbox.pack()

8. Canvas
Dùng để vẽ đồ họa như hình tròn, hình chữ nhật, hoặc hình ảnh.

    canvas = Canvas(root, width=200, height=100)
    canvas.create_line(0, 0, 200, 100, fill="blue")
    canvas.create_rectangle(50, 25, 150, 75, fill="red")
    canvas.pack()

9. Frame
Dùng để tổ chức các widget con bên trong một khung.

    frame = Frame(root, borderwidth=2, relief="sunken")
    frame.pack()

10. Scrollbar
Thêm thanh cuộn cho widget như Text hoặc Canvas.

    scrollbar = Scrollbar(root)
    scrollbar.pack(side=RIGHT, fill=Y)

11. Scale
Dùng để chọn một giá trị trong phạm vi bằng cách kéo thanh trượt.

    scale = Scale(root, from_=0, to=100, orient=HORIZONTAL)
    scale.pack()

12. Menu
Tạo thanh menu hoặc menu ngữ cảnh.

    menu = Menu(root)
    root.config(menu=menu)
    file_menu = Menu(menu)
    menu.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open")
    file_menu.add_command(label="Save")

13. Combobox (từ ttk)
Một hộp thả xuống với các tùy chọn.

    from tkinter.ttk import Combobox
    combo = Combobox(root, values=["Option 1", "Option 2", "Option 3"])
    combo.pack()

14. Progressbar (từ ttk)
Hiển thị tiến trình của một tác vụ.

    from tkinter.ttk import Progressbar
    progress = Progressbar(root, orient=HORIZONTAL, length=200, mode='determinate')
    progress.pack()
    progress.start(10)  # Tiến trình tự động chạy
'''

class GUI:
    def __init__(self, variables: SharedVariables):
        self.var = variables

        self.thread = Thread(target=self.start)
        self.thread.daemon = True
        self.thread.start()

    def test(self, operator):
        if self.var == None:
            return
        if operator == "+":
            self.var.set("pingpong_count", self.var.get("pingpong_count") + 1)
        elif operator == "-":
            self.var.set("pingpong_count", self.var.get("pingpong_count") - 1)
        print(self.var.get("pingpong_count"))

    def setup(self):
        self.root = tk.Tk()
        self.root.geometry("200x200+50+50")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.label1 = tk.Label(self.root, text="")
        self.label1.pack()

        self.button1 = tk.Button(self.root, text="Decrease", command=lambda: self.test("-"))
        self.button1.pack()

        self.button2 = tk.Button(self.root, text="Increase", command=lambda: self.test("+"))
        self.button2.pack()

    def update_gui(self):
        print("update gui")

        self.label1.config(text=f'{self.var.get("pingpong_count")}')

    def start(self):
        self.setup()
        self.update_gui()
        self.root.mainloop()

    def on_close(self):
        self.root.destroy()
        print("tkinter destroy")
        # gui_thread.join()