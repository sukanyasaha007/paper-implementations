"""Transformer from "Attention Is All You Need" (Vaswani et al., 2017), built from scratch in PyTorch."""
import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        #ex of tokenizer entries with vocab_size = 32_000
        # "<pad>"  -> 0
        # "<unk>"  -> 1
        # "hello"  -> 2
        # "world"  -> 3
        # "ing"    -> 4
        # ...
        self.d_model = d_model  # the size of the vector used to represent each token throughout the Transformer - here 512
        self.vocab_size = vocab_size #number of tokens known by the tokenizer

        #Create embeddings table with:
            # One row per known token: vocab_size
            # One learned vector per row: d_model values
        #eg. nn.Embedding(32_000, 512) contains 32,000 token embeddings, each represented by a vector of 512 numbers.

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # token IDs into embedding vectors
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    1/10000^(2i/d_model) = exp(2i * (-ln(10000) / d_model)), which is how div_term is computed.
    For d_model = 4: 2i = [0, 2] -> div_term = [1.0, 0.01].

    position:            (seq_len, 1)
    div_term:            (d_model / 2,)
    position * div_term: (seq_len, d_model / 2)

    Early dimension pairs vary quickly with position, later pairs vary slowly, so the model can
    represent both short- and long-range positional relationships.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) #a vector of shape (seq_len x 1)
        #div term is calculated in log prob for numerical stability
        div_term = torch.exp(
                            torch.arange(0, d_model, 2).float() #produces the even dimension indices: [0, 2, 4, ..., 510]
                                 * (-math.log(10000.0)/d_model)
                                )
        pe[:, 0::2] = torch.sin(position * div_term) # even positions
        pe[:, 1::2] = torch.cos(position * div_term) # odd positions

        #prepare batch of this
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)
        self.register_buffer('pe', pe) # saved with the model and moved with .to(device), but not a learned parameter

    def forward(self, x):
        # x comes from input embedding output layer: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.shape[1], :] #they are not learned through back prop gradient decent
        # (1, seq_len, d_model) repeated for each batch element
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps # clipping to avoid div by zero
        #learning params
        self.alpha = nn.Parameter(torch.ones(1)) #this is multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #this is added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) #dim -1 = last dim layer norm is done at item level, keepdim to still have vector instead of number
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model : int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiheadHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "Embedddings dimension can not be devided to h heads"

        self.d_k = d_model // h
        # self.w_q  # what this token is looking for
        # self.w_k  # what this token can be matched by
        # self.w_v  # information this token can provide
        self.w_q = nn.Linear(d_model, d_model)  # nn.Linear() does many things here- creates weight tensor, bias tensor, registers them as model parameters, parameter initializaton, saves them, forward operation that applies them
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout : nn.Dropout):
        d_k = query.shape[-1]

        #(batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        #musk upper triangle
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        #softmax
        attention_scores = attention_scores.softmax(dim=-1) #(batch, h, seq_len, se_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return attention_scores @ value, attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(batch, seq_len, d_model)

        key = self.w_k(k)
        value = self.w_v(v)
        # (batch_size, seq_len, d_model) -> (batch, seq_len, heads, d_k) -> (batch, heads, seq_len, d_k)
        # Before: (2, 10, 512) -> (2, 10, 8, 64) -> (2, 8, 10, 64)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # break into h heads
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) # break into h heads
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) # break into h heads

        # attention_scores are kept on the module for visualization
        x, self.attention_scores = MultiheadHeadAttention.attention(query, key, value, mask, self.dropout)
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.shape[0], x.shape[1], self.h * self.d_k)

        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        # pre-norm: normalize, run the sublayer, then add back the input
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block =  feed_forward_block
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

        # embeddings
        # multihead attention
        # add + layer norm
        # reisdual connection with embeddings and layer norm
        # feedforward
        # add + layer norm
        # residual block with last residual block output

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # q, k, v  all x, it goes to multi head attention and to residual block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiheadHeadAttention, cross_attention_block: MultiheadHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask): # 2 mask for encoder and decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """Linear layer to project the learned embeddings into the vocabulary."""

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embeddings: InputEmbeddings, target_embeddings: InputEmbeddings, src_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.target_embeddings = target_embeddings
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer

    #instead of 1 forward we break it down for later inference and visualization
    def encode(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, target, encoder_output, src_mask, target_mask):
        target = self.target_embeddings(target)
        target = self.target_pos(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.projection_layer(x)


def build_transformer(src_vocab: int, target_vocab: int, src_seq_len: int, target_seq_len: int, d_model: int = 512, N : int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    #create embeddings
    src_embed = InputEmbeddings(d_model, src_vocab)
    target_embed = InputEmbeddings(d_model, target_vocab)

    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiheadHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiheadHeadAttention(d_model, h, dropout)
        cross_attenttion_block = MultiheadHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, cross_attenttion_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    projection_layer = ProjectionLayer(d_model, target_vocab)

    transformer = Transformer(encoder, decoder, src_embed, target_embed, src_pos, target_pos, projection_layer)

    #Intialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer
