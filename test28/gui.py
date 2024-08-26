import sys
import pygame
from typing import List
import tkinter as tk
from threading import Thread
from OpenGL.GL import *

class Widget:
    def __init__(self, widget_class, vars, kwargs) -> None:
        self._cls = widget_class
        self._vars = vars
        self._kwargs = kwargs

class GUI:
    def __init__(self, shared_variable) -> None:
        self.shared_variable = shared_variable
        self._thread_gui = Thread(target=self._mainloop)
        # self._thread_update = Thread(target=self._update_var)
        # self._thread_update.setDaemon(True)
        # self._thread_gui.setDaemon(True)
        self._widgets: List[Widget] = []
        self._vars = {}
        self._var_count = 0
        self._running = True

    def _mainloop(self):
        self._gui = tk.Tk("OpenGL Toolkit")
        self._gui.protocol("WM_DELETE_WINDOW", lambda: None)
        self._gui.configure(bg='lightblue')
        for widget in self._widgets:
            kwargs = {}
            kwargs.update(**widget._kwargs)
            # print(widget._vars)

            for key, value in widget._vars.items():
                var = value[1]()
                var.trace_add("write", self._update_var)
                kwargs[value[0]] = var
                self._vars[key] = var

            widget._cls(self._gui, **kwargs).pack()
        # print(self._vars)
        # self._thread_update.start()
        self._gui.mainloop()

    def mainloop(self):
        self._thread_gui.start()

    def _add_widget(self, widget, vars={}, kwargs={}):
        self._widgets.append(Widget(widget, vars, kwargs))

    def add_entry(self):
        self._add_widget(
            tk.Entry, 
            kwargs={
                "background": "green"
            }
        )

    def add_color3(self, var_name: str):
        # var = tk.DoubleVar()
        for index in range(3):
            self._add_widget(
                tk.Scale(),
                vars={
                    var_name + str(index): ("variable", tk.IntVar)
                },
                kwargs={
                    "orient": "horizontal",
                    
                }
            )

    def _update_var(self, *args):
        # while self._running:
        # print(args)
        for name, value in self.shared_variable.items():
            if isinstance(value, list):
                index = 0
                for v in value:
                    # self._vars[name + str(index)].set(v)
                    self.shared_variable[name][index] = self._vars[name + str(index)].get()
                    index += 1
                continue
            self.shared_variable[name] = self._vars[name].get()
            # self._vars[name].set(v)

    def quit(self):
        # print(type(self._gui))
        self._running = False
        self._gui.quit()