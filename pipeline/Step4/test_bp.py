import numpy as np
import paddle
import torch
from paddlenlp.transformers import (BertForSequenceClassification as
                                    PDBertForSequenceClassification, )
from reprod_log import ReprodLogger
from transformers import AdamW
from transformers import (BertForSequenceClassification as
                          HFBertForSequenceClassification, )


def pd_train_some_iters(model,
                        criterion,
                        optimizer,
                        fake_data,
                        fake_label,
                        max_iter=2):
    model = PDBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_classes=2)
    classifier_weights = paddle.load(
        "../classifier_weights/paddle_classifier_weights.bin")
    model.load_dict(classifier_weights)
    model.eval()
    criterion = paddle.nn.CrossEntropy()
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=3e-5,
        parameters=model.parameters(),
        weight_decay=1e-2,
        epsilon=1e-6,
        apply_decay_param_fun=lambda x: x in decay_params, )
    loss_list = []
    for idx in range(max_iter):
        input_ids = paddle.to_tensor(fake_data)
        labels = paddle.to_tensor(fake_label)

        output = model(input_ids)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        loss_list.append(loss)
    return loss_list


def hf_train_some_iters(fake_data, fake_label, max_iter=2):
    model = HFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)
    classifier_weights = torch.load(
        "../classifier_weights/torch_classifier_weights.bin")
    model.load_state_dict(classifier_weights, strict=False)
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 1e-2,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)

    loss_list = []
    for idx in range(max_iter):
        input_ids = torch.from_numpy(fake_data)
        labels = torch.from_numpy(fake_label)

        output = model(input_ids)[0]
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss)
    return loss_list


if __name__ == "__main__":
    print("Start training")
    paddle.set_device("cpu")
    fake_data = np.load("../fake_data/fake_data.npy")
    fake_label = np.load("../fake_data/fake_label.npy")
    hf_reprod_logger = ReprodLogger()
    hf_loss_list = hf_train_some_iters(fake_data, fake_label, 10)
    for idx, loss in enumerate(hf_loss_list):
        hf_reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    hf_reprod_logger.save("bp_align_torch.npy")

    pd_reprod_logger = ReprodLogger()
    pd_loss_list = hf_train_some_iters(fake_data, fake_label, 10)
    for idx, loss in enumerate(pd_loss_list):
        pd_reprod_logger.add(f"loss_{idx}", loss.detach().cpu().numpy())
    pd_reprod_logger.save("bp_align_paddle.npy")
