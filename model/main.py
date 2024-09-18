import os, sys, time
import logging

import torch
from torch import nn, optim

from build_model import build_model

def train(model, data_loader, optimizer, criterion, epoch, checkpoint_dir):
    model.train()
    epoch_loss = 0

    for idx, (image, input_label, output_label) in enumerate(data_loader):
        image = image.to(model.device)
        input_label = input_label.to(model.device)
        output_label = output_label.to(model.device)

        optimizer.zero_grad()
        output = model(image, input_label)

        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = output_label.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 2.0)
        optimizer.step()

        epoch_loss += loss.item()
    num_samples = idx + 1

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{epoch:04d}.pt") # epoch 번호를 4자리 숫자 형식으로 파일명 저장
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, checkpoint_path)

    return epoch_loss / num_samples

def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0

    total_bleu = []
    with torch.no_grad():
        for idx, (image, input_label, output_label) in enumerate(data_loader):
            image = image.to(model.device)
            input_label = input_label.to(model.device)
            output_label = output_label.to(model.device)

            output = model(image, input_label)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = output_label.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)
