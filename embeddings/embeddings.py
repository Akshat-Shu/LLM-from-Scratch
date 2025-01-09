import torch as t

def get_embeddings(vocab_size, output_dim, context_length):
    t.manual_seed(123)

    embedding_layer = t.nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = t.nn.Embedding(context_length, output_dim)
    
    return embedding_layer + pos_embedding_layer