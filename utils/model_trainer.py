import logging

import torch
from torch import nn, optim


class ModelTrainer():

    def __init__(self, model, args=None):
        self.model = model  # 模型本体
        self.cid = 0  # 所属客户id
        self.args = args # 运行参数
        self.criterion = nn.CrossEntropyLoss() # 损失函数
    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def print_current_lr(self):
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])
    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.num_local_update):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            print(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.cid, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )

    def test(self, test_data, device):
        model = self.model
        model.to(device)
        model.eval()
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        criterion = nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                target = target.long()
                loss = criterion(pred, target)  # pylint: disable=E1102
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics

