import numpy as np
import torch
import torch.nn as nn
from reprod_log import ReprodLogger
from transformers import BertForSequenceClassification

if __name__ == "__main__":

    # def logger
    reprod_logger = ReprodLogger()

    criterion = nn.CrossEntropyLoss()

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
    classifier_weights = torch.load(
        "../classifier_weights/torch_classifier_weights.bin")
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()
    # read or gen fake data
    fake_data = np.load("../fake_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)

    fake_label = np.load("../fake_data/fake_label.npy")
    fake_label = torch.from_numpy(fake_label)

    # forward
    out = model(fake_data)[0]

    loss = criterion(out, fake_label)
    #
    reprod_logger.add("loss", loss.cpu().detach().numpy())
    reprod_logger.save("loss_torch.npy")
