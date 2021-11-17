import numpy as np
import torch
from reprod_log import ReprodLogger
from transformers.models.bert import BertForSequenceClassification

if __name__ == "__main__":
    # def logger
    reprod_logger = ReprodLogger()

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
    classifier_weights = torch.load(
        "../classifier_weights/torch_classifier_weights.bin")
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()

    # read or gen fake data
    fake_data = np.load("../fake_data/fake_data.npy")
    fake_data = torch.from_numpy(fake_data)
    # forward
    out = model(fake_data)[0]
    #
    reprod_logger.add("logits", out.cpu().detach().numpy())
    reprod_logger.save("forward_torch.npy")
