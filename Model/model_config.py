# configuration for the GPT 124M parameter model. Unable to run 355M parameter model on my hardware.
# TODO: Try out the 355M parameter model on a clous provider 

class Config:
    def __init__(self):
        self.VOCAB_SIZE = 50257
        self.CONTEXT_LENGTH = 1024
        self.EMBED_DIM = 768
        self.DROPOUT = 0.1
        self.N_HEADS = 12
        self.N_LAYERS = 12
        self.QKV_BIAS = True