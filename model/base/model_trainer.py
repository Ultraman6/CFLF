import copy
import random
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn
from torch.autograd import grad

from model.base.model_dict import _modeldict_zeroslike, _modeldict_clone, _modeldict_add, _modeldict_tuple, \
    _modeldict_sub, _modeldict_norm, _modeldict_to_cpu, _modeldict_to_np, _modeldict_scale, _modeldict_to_device
from utils.running import create_optimizer, create_loss_function, schedule_lr, js_divergence, direct_kl_sum


class ModelTrainer:
    def __init__(self, model, device, args=None, cid=-1):
        self.model = copy.deepcopy(model)
        self.cid = cid
        self.args = args
        self.device = device
        self.all_epoch_losses = {}  # 用于存储所有 round 的损失及相关信息
        # 2024-1-30新增学习率策略
        self.criterion = create_loss_function(args.loss_function)
        self.optimizer = create_optimizer(self.model, args)

    def get_model_params(self, device=None):
        params = copy.deepcopy(self.model.state_dict())
        _modeldict_to_device(params, device)
        return params

    def set_model_params(self, model_parameters):  # 由于设计设备的移动，所以需重新装载优化器
        params = copy.deepcopy(model_parameters)
        _modeldict_to_device(params, self.device)
        self.model.load_state_dict(params)
        for param_group in self.optimizer.param_groups:
            param_group['params'] = list(self.model.parameters())

    def zero_grad(self):
        self.model.zero_grad()
        self.optimizer.zero_grad()

    def upgrate_lr(self, roundidx):
        lr = self.optimizer.param_groups[0]['lr']
        new_lr = schedule_lr(roundidx, lr, self.args)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def train(self, train_data, global_round):
        self.model.to(self.device)
        epoch_losses = []
        for epoch in range(self.args.epoch):
            batch_loss = []
            self.model.train()
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(self.device), labels.to(self.device).long()
                log_probs = self.model(x)
                self.zero_grad()
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            epoch_losses.append(sum(batch_loss) / len(batch_loss) if batch_loss else 0.0)
            # print(f"Client Index = {self.cid}\tEpoch: {epoch}\tLoss: {epoch_losses[-1]:.6f}")
        # 存储每个 round 的详细信息，包括平均损失和每个 epoch 的损失
        round_avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.all_epoch_losses[global_round] = {
            "round_avg_loss": round_avg_loss,
            "epoch_losses": epoch_losses,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        self.upgrate_lr(global_round - 1)  # 更新学习率

    def train_hessian(self, train_data, device, global_round):
        self.model.to(device)
        self.model.train()
        epoch_losses = []
        batches_to_sample = random.sample(range(len(train_data)), 3)
        gradients = []
        hessian_estimations = []
        # 在训练开始前记录模型参数
        initial_params = _modeldict_clone(self.get_model_params())
        for epoch in range(self.args.epoch):
            epoch_losses = []
            for i, (inputs, targets) in enumerate(train_data):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                self.zero_grad()
                loss = self.criterion(outputs, targets)
                # 捕获梯度
                grads = _modeldict_tuple(self.model.state_dict(),
                                         torch.autograd.grad(loss, self.model.parameters(), create_graph=True))
                gradients.append(grads)
                # 更新模型参数
                self.optimizer.step()
                epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                # print(f'Epoch {epoch + 1}/{self.args.epoch}, Loss: {epoch_loss:.4f}')

        # 在优化器步骤之后使用捕获的梯度来估计Hessian向量积
        for i, grads in enumerate(gradients):
            if i in batches_to_sample:
                for grad_i, param in zip(grads, self.model.parameters()):
                    v = torch.randn_like(param.data)
                    hvp = torch.autograd.grad(grad_i, param, grad_outputs=v, only_inputs=True, retain_graph=True)[0]
                    hessian_estimations.append(hvp.detach())
        if hessian_estimations:
            average_hessian_estimation = torch.mean(torch.stack(hessian_estimations), dim=0)
        else:
            average_hessian_estimation = None

        # 存储每个 round 的详细信息，包括平均损失和每个 epoch 的损失
        round_avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.all_epoch_losses.append({
            "client_id": self.cid,
            "global_round": global_round,
            "round_avg_loss": round_avg_loss,
            "epoch_losses": epoch_losses,
            "current_learning_rate": self.optimizer.param_groups[0]['lr']
        })
        self.upgrate_lr(global_round)  # 更新学习率
        return average_hessian_estimation

    def get_all_epoch_losses(self):
        return self.cid, self.all_epoch_losses

    def test(self, test_data):
        self.model.to(self.device)
        self.model.eval()
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        with torch.no_grad():  # 在测试过程中，我们不需要计算梯度
            for batch_idx, (x, target) in enumerate(test_data):
                x, target = x.to(self.device), target.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        # 计算平均损失
        average_test_loss = metrics["test_loss"] / metrics["test_total"]
        # 计算正确率
        accuracy = metrics["test_correct"] / metrics["test_total"]
        return accuracy, average_test_loss

    def test_JSD(self, test_data):
        self.model.to(self.device)
        self.model.eval()
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0, "jsd_sum": 0}
        with torch.no_grad():  # 在测试过程中，我们不需要计算梯度
            for batch_idx, (x, target) in enumerate(test_data):
                x, target = x.to(self.device), target.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                # 计算JSD
                target_one_hot = F.one_hot(target, num_classes=pred.size(1)).float()
                jsd = js_divergence(pred, target_one_hot.to(self.device))

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
                metrics["jsd_sum"] += jsd.item()

        # 计算平均损失
        average_test_loss = metrics["test_loss"] / metrics["test_total"]
        # 计算正确率
        accuracy = metrics["test_correct"] / metrics["test_total"]
        # 计算平均JSD
        average_jsd = metrics["jsd_sum"] / len(test_data)
        return accuracy, average_test_loss, average_jsd

    def test_direct_KL(self, test_data):
        self.model.to(self.device)
        self.model.eval()
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0, "kl_sum": 0}
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x, target = x.to(self.device), target.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                # 确保预测和目标都转换为概率分布
                pred_prob = F.softmax(pred, dim=1)
                target_one_hot = F.one_hot(target, num_classes=pred_prob.size(1)).float().to(self.device)
                target_prob = target_one_hot.to(self.device)

                # 计算正向和逆向KL散度的直接和
                kl_sum = direct_kl_sum(pred_prob, target_prob)

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
                metrics["kl_sum"] += kl_sum.item()

        # 计算平均损失和正确率
        average_test_loss = metrics["test_loss"] / metrics["test_total"]
        accuracy = metrics["test_correct"] / metrics["test_total"]
        # 计算正向和逆向KL散度的直接和的平均值
        average_kl_sum = metrics["kl_sum"] / len(test_data)

        return accuracy, average_test_loss, average_kl_sum

    def test_pred(self, test_data):
        preds = []
        self.model.to(self.device)
        self.model.eval()
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        with torch.no_grad():  # 在测试过程中，我们不需要计算梯度
            for batch_idx, (x, target) in enumerate(test_data):
                x, target = x.to(self.device), target.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                preds.append(F.softmax(pred, dim=1).cpu().numpy())  # 先转到CPU，然后转为NumPy数组
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        # 计算平均损失
        average_test_loss = metrics["test_loss"] / metrics["test_total"]
        # 计算正确率
        accuracy = metrics["test_correct"] / metrics["test_total"]
        # 将预测结果合并为一个大的NumPy数组
        all_preds = np.concatenate(preds, axis=0)
        return accuracy, average_test_loss, all_preds

    def test_info(self, test_data):
        self.model.to(self.device)
        self.model.eval()
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0, "info_entropy": 0}
        with torch.no_grad():  # 在测试过程中，我们不需要计算梯度
            for batch_idx, (x, target) in enumerate(test_data):
                x, target = x.to(self.device), target.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                # 计算信息熵
                softmax_preds = F.softmax(pred, dim=1)
                info_entropy_batch = -(softmax_preds * softmax_preds.log()).sum(dim=1).mean().item()  # 计算每个样本的信息熵并求平均
                metrics["info_entropy"] += info_entropy_batch * target.size(0)
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        # 计算平均损失
        average_test_loss = metrics["test_loss"] / metrics["test_total"]
        # 计算正确率
        accuracy = metrics["test_correct"] / metrics["test_total"]
        # 计算平均信息熵
        average_info_entropy = metrics["info_entropy"] / metrics["test_total"]

        return accuracy, average_test_loss + average_info_entropy

    def test_grad(self, test_data):
        self.model.to(self.device)
        self.model.eval()  # 确保模型处于评估模式
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        # 初始化存储梯度的字典
        grad_total = {name: torch.zeros_like(param, device=self.device)
                      for name, param in self.model.named_parameters()}

        batch_count = 0
        for batch_idx, (x, target) in enumerate(test_data):
            x, target = x.to(self.device), target.to(self.device).long()
            x.requires_grad_(True)  # 为输入数据设置requires_grad以便计算梯度
            self.model.zero_grad()  # 清除现有梯度

            # 前向传播
            pred = self.model(x)
            loss = self.criterion(pred, target)
            loss.backward()  # 反向传播计算梯度

            # 累加梯度
            for name, parameter in self.model.named_parameters():
                if parameter.grad is not None:
                    grad_total[name] += parameter.grad.detach()

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()
            metrics["test_correct"] += correct.item()
            metrics["test_loss"] += loss.item() * target.size(0)
            metrics["test_total"] += target.size(0)
            batch_count += 1

        # # 计算梯度平均值
        # for name in grad_total:
        #     grad_total[name] /= batch_count

        average_test_loss = metrics["test_loss"] / metrics["test_total"]
        accuracy = metrics["test_correct"] / metrics["test_total"]

        return accuracy, average_test_loss, grad_total

    def test_norm(self, test_data):
        self.model.to(self.device)
        self.model.eval()
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0, "grad_norm": 0.0}
        # 假设所有样本的类别数为C
        num_classes = self.model.out_size  # 假设模型有一个属性output_size定义了输出类别的数量
        # 提前生成均匀分布，所有值都是1/C
        uniform_dist = torch.full((1, num_classes), 1.0 / num_classes).to(self.device)
        for batch_idx, (x, target) in enumerate(test_data):
            x, target = x.to(self.device), target.to(self.device)
            # 开启梯度计算
            x.requires_grad_(True)
            pred = self.model(x)
            loss = self.criterion(pred, target)
            # 计算KL散度
            self.model.zero_grad()
            kl_divergence = F.kl_div(F.log_softmax(pred, dim=1), uniform_dist.expand_as(pred), reduction='batchmean')
            # 计算KL散度的梯度
            self.model.zero_grad()
            kl_divergence.backward(retain_graph=True)
            # 提取参数梯度范数
            grad_norm = 0.0
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    grad_norm += parameter.grad.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            # 累计梯度范数
            metrics["grad_norm"] += grad_norm
            # 清空梯度
            self.model.zero_grad()
            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()
            metrics["test_correct"] += correct.item()
            metrics["test_loss"] += loss.item() * target.size(0)
            metrics["test_total"] += target.size(0)
        # 计算平均损失和平均梯度范数
        average_test_loss = metrics["test_loss"] / metrics["test_total"]
        average_grad_norm = metrics["grad_norm"] / metrics["test_total"]
        # 计算正确率
        accuracy = metrics["test_correct"] / metrics["test_total"]
        return accuracy, average_test_loss, average_grad_norm

    def test_loss(self, test_data):
        self.model.to(self.device)
        self.model.eval()
        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        loss_distribution = []  # 用于收集每个样本的损失值
        with torch.no_grad():  # 在测试过程中，我们不需要计算梯度
            for batch_idx, (x, target) in enumerate(test_data):
                x, target = x.to(self.device), target.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
                loss_distribution.append(loss.item() * target.size(0))  # 收集每个batch上的损失
        # 计算平均损失
        average_test_loss = metrics["test_loss"] / metrics["test_total"]
        # 计算正确率
        accuracy = metrics["test_correct"] / metrics["test_total"]
        return accuracy, average_test_loss, loss_distribution
