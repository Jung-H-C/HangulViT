import torch
import torch.nn

from model.block.decoder_block import DecoderBlock
from model.embedding.embedding import JamoEmbedding
from model.layer.multi_head_attention import MultiHeadAttention
from model.layer.residual_connection import Residual
from model.layer.position_feed_forward import PositionWiseFeedForwardLayer
from model.decoder import Decoder
from model.encoder import CNNEncoder
from model.transformer import HangulViT

random_image = torch.randn(32, 1, 360, 360)
random_input = torch.randint(0, 54, (32, 12))
model = HangulViT(input_embed = JamoEmbedding, encoder = CNNEncoder,
                  decoder = Decoder, embedding_dim = 512, vocab_size = 54).cuda()

try:
    output = model(random_image, random_input)
    print("성공!")
    print("출력 크기: {}".format(output.size()))
except Exception as e:
    print(e)