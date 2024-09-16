import torch
import torch.nn as nn
import copy

from model.block.decoder_block import DecoderBlock
from model.embedding.embedding import JamoEmbedding
from model.layer.multi_head_attention import MultiHeadAttention
from model.layer.residual_connection import Residual
from model.layer.position_feed_forward import PositionWiseFeedForwardLayer
from model.decoder import Decoder
from model.encoder import CNNEncoder
from model.transformer import HangulViT

random_image = torch.randn(32, 1, 360, 360) # image
random_input = torch.randint(51, 54, (32, 12)) # label (char_to_index 처리 되어있음)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(
        device = torch.device('cuda'), decoder = Decoder, embedding_dim = 512, vocab_size = 54, norm_eps = 1e-5):

    import copy
    copy = copy.deepcopy

    norm = nn.LayerNorm(embedding_dim, eps=norm_eps).to(device)

    attention = MultiHeadAttention(
        embedding_dim = embedding_dim,
        num_heads = 1,
        dropout = 0.1
    )
    position_ff = PositionWiseFeedForwardLayer(
        embedding_dim = embedding_dim,
        hidden_dim = 2048,
        dropout = 0.1
    )
    decoder_block = DecoderBlock(
        self_attention=attention,
        cross_attention=attention,
        position_ff=position_ff,
        norm = copy(norm)
    )
    decoder = decoder(
        n_layer=1,
        decoder_block=decoder_block,
        norm = copy(norm)
    )


    model = HangulViT(
        input_embed = JamoEmbedding().to(device),
        decoder = decoder,
        vocab_size = vocab_size,
        embedding_dim = embedding_dim).to(device)

    model.to(device)

    return model

# model = HangulViT(input_embed = JamoEmbedding, encoder = CNNEncoder,
#                   decoder = Decoder, embedding_dim = 512, vocab_size = 54).cuda()
model = build_model(device = torch.device('cuda'), decoder = Decoder, embedding_dim = 512, vocab_size = 54)

try:
    output = model(random_image, random_input)
    print("성공!")
    print("출력 크기: {}".format(output.size()))
except Exception as e:
    print(e)