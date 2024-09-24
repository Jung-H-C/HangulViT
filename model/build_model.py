import torch
import torch.nn as nn

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


def build_model():

    import copy
    copy = copy.deepcopy

    embedding_dim = 512
    vocab_size = 54
    norm_eps = 1e-5
    num_heads = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    norm = nn.LayerNorm(embedding_dim, eps=norm_eps).to(device)

    attention = MultiHeadAttention(
        embedding_dim = embedding_dim,
        num_heads = num_heads,
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
    encoder = CNNEncoder()

    decoder = Decoder(
        n_layer=1,
        decoder_block=decoder_block,
        norm = copy(norm)
    )
    generator = nn.Linear(embedding_dim, vocab_size)


    model = HangulViT(
        input_embed = JamoEmbedding().to(device),
        encoder = encoder.to(device),
        decoder = decoder.to(device),
        vocab_size = vocab_size,
        embedding_dim = embedding_dim,
        device = device,
        generator = generator).to(device)

    model.to(device)

    return model

# model = HangulViT(input_embed = JamoEmbedding, encoder = CNNEncoder,
#                   decoder = Decoder, embedding_dim = 512, vocab_size = 54).cuda()
# model = build_model()

def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters: {}".format(total_params))

def model_size_in_MB(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    model_size_mb = param_size / (1024 ** 2)
    print("Model size in MB: {}".format(model_size_mb))

# images, input_labels, output_labels = return_one_batch()
#
# print("images")
# print(images[0, ...])
# print("input_labels")
# print(input_labels[0, ...])
# print("output_labels")
# print(output_labels[0, ...])
#
#
# # count_model_parameters(model) # 파라미터 수: 7.90M
# # model_size_in_MB(model) # 모델 크기: 30.16MB
#
# try:
#     output = model(images, input_labels)
#     print("성공!")
#     print("output: {}".format(output))
#     print("출력 크기: {}".format(output.size()))
# except Exception as e:
#     print(e)
# #
