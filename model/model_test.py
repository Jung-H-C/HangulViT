import torch

from model.build_model import build_model
from model.model.config import *
from data import *
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from Vocabulary import *
from build_model import count_model_parameters, model_size_in_MB

def image_postprocess(image):
    image = torch.squeeze(image, dim = 1)
    image = image.to('cpu')
    image = image.permute(1, 2, 0)
    image = image * 0.5 + 0.5
    return image

def convert_to_index(output_tensor):
    indices = torch.argmax(output_tensor, dim = -1)
    indices = indices.unsqueeze(-1)
    target_indices = indices[..., -1]
    return target_indices

def index_to_char(tensor):
    char_list = Vocabulary().vocabulary
    char_tensor = [[char_list[idx] for idx in seq] for seq in tensor]
    return char_tensor



def main():
    model = build_model()
    checkpoint = torch.load('D:/HangulViT/HangulViT/model/checkpoint/0000.pt')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    count_model_parameters(model)
    model_size_in_MB(model)
    # loss = checkpoint['loss']

    # with open('test_labels.pkl', 'rb') as f:
    #     test_labels = pickle.load(f)
    #
    # test_dataset = HangulOCRDataset(img_dir = IMG_DIR, labels = test_labels, transform = TRANSFORM)
    # test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = True, collate_fn = my_collate_fn, num_workers = 1)
    #
    # images, input_labels, output_labels = next(iter(test_loader))
    #
    # image = image_postprocess(images)
    # input_labels = input_labels.to(model.device)
    # output_labels = output_labels.to(model.device)
    #
    # output = model(images, input_labels)
    #
    # output = convert_to_index(output)
    #
    # plt.imshow(image.numpy())
    # plt.axis('off')
    # plt.show()
    #
    # print("output: {}, target value: {}".format(index_to_char(output), index_to_char(output_labels)))

if __name__ == '__main__':
    main()