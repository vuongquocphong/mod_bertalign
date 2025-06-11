class Argument:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Argument, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "args"): return

        self.args = dict()
        self.args["skip"] = -0.1
        # self.args["sentence_num_penalty"] = 0.01
        # self.args["union_score"] = 0.15
        self.args["union_score"] = 0.0
        self.args["sentence_num_penalty"] = 0.0
    
    def __getitem__(self, key):
        return self.args[key]
    
    def __setitem__(self, key, value):
        self.args[key] = value
