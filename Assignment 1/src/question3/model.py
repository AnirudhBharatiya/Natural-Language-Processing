import torch.nn as nn

class NPLM(nn.Module):
    """
    Neural Probabilistic Language Model (Bengio et al., 2003).
    """
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim, activation='tanh'):
        super(NPLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Input to hidden layer is the concatenation of context word embeddings
        self.input_dim = context_size * embedding_dim

        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        else:
            act_fn = nn.Tanh()

        self.hidden_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            act_fn
        )
        self.activation_type = activation.lower()
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs) 
        concat_input = embeds.view(inputs.size(0), -1)
        hidden_out = self.hidden_layer(concat_input)
        logits = self.output_layer(hidden_out)
        return logits