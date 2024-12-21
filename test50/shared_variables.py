class SharedVariables:
    def __init__(self):
        self.listen_event = []
        self.variables_init()

    def variables_init(self):
        self.var_dict = {
            "pingpong_count": 10
        }

    def set_event(self, event):
        self.listen_event.append(event)

    def dispatch_event(self):
        # print(self.var_dict["pingpong_count"])
        for event in self.listen_event:
            event()

    def set(self, var_name: str, value):
        if self.var_dict.get(var_name, "No variable") == "variable":
            # print("return")
            return
        self.var_dict[var_name] = value
        self.dispatch_event()

    def get(self, var_name):
        return self.var_dict.get(var_name, None)

    # def set(self, var_name: str, value):
    #     if not hasattr(self, var_name):
    #         return
    #     setattr(self, var_name, value)
    #     self.dispatch_event()

    # def get(self, var_name):
    #     return getattr(self, var_name, None)