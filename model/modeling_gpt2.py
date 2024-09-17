import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, context_len, num_heads):
        super().__init__()
        assert output_dim % num_heads == 0, "num_heads must be divisible by output_dim"
        
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.input_dim = input_dim

        self.qkv_layer = nn.Linear(input_dim, 3 * output_dim)
        self.linear_layer = nn.Linear(output_dim, output_dim)
        self.register_buffer('mask', torch.triu(torch.ones(context_len, context_len), diagonal=1))
        self.dropout = nn.Dropout(0.1)  # Optional dropout layer

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()

        qkv_layer = self.qkv_layer(x)
        qkv_layer = qkv_layer.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv_layer.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        attn_scores = q @ k.transpose(-1, -2)
        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attn_scores.masked_fill_(mask_bool, float('-inf'))
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)  

        context_vec = (attn_weights @ v).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batch_size, seq_len, self.output_dim)
        context_vec = self.linear_layer(context_vec)

        return context_vec

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2 * output_dim),
            nn.GELU(),
            nn.Dropout(0.1),  # Optional dropout layer
            nn.Linear(2 * output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = MultiHeadAttention(
            input_dim=args.emb_dim,
            output_dim=args.emb_dim,
            context_len=args.max_length,
            num_heads=args.num_heads
        )
        self.ff = MLP(
            input_dim=args.emb_dim,
            output_dim=args.emb_dim
        )
        self.norm1 = nn.LayerNorm(args.emb_dim)
        self.norm2 = nn.LayerNorm(args.emb_dim)
        self.dropout = nn.Dropout(0.1)  # Optional dropout layer

    def forward(self, x):
        # Attention block with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)  # Apply dropout
        x = residual + x

        # Feedforward block with residual connection
        residual2 = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)  # Apply dropout
        x = residual2 + x

        return x

class GPT2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_emb = nn.Embedding(args.vocab_size, args.emb_dim)
        self.pos_emb = nn.Embedding(args.max_length, args.emb_dim)
        
        self.transformers = nn.Sequential(
            *[TransformerDecoderBlock(args) for _ in range(args.num_layers)]
        )
        self.layer_norm = nn.LayerNorm(args.emb_dim)
        self.final_layer = nn.Linear(args.emb_dim, args.vocab_size, bias=False)

    def forward(self, x):
        batch_size, seq_len = x.size()
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        #pos_embeds = pos_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        x = tok_embeds + pos_embeds
        x = self.transformers(x)
        x = self.layer_norm(x)
        logits = self.final_layer(x)

        return logits