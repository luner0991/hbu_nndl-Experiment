import torch
class RunnerV3(object):
    def __init__(self, model, optimizer, loss_fn, metric, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric  # 只用于计算评价指标

        # 记录训练过程中的评价指标变化情况
        self.dev_scores = []

        # 记录训练过程中的损失函数变化情况
        self.train_epoch_losses = []  # 一个epoch记录一次loss
        self.train_step_losses = []  # 一个step记录一次loss
        self.dev_losses = []

        # 记录全局最优指标
        self.best_score = 0

    def train(self, train_loader, dev_loader=None, **kwargs):
        # 将模型切换为训练模式
        self.model.train()

        # 传入训练轮数，如果没有传入值则默认为0
        num_epochs = kwargs.get("num_epochs", 0)
        # 传入log打印频率，如果没有传入值则默认为100
        log_steps = kwargs.get("log_steps", 100)
        # 评价频率
        eval_steps = kwargs.get("eval_steps", 0)

        # 传入模型保存路径，如果没有传入值则默认为"best_model.pdparams"
        save_path = kwargs.get("save_path", "best_model.pdparams")

        custom_print_log = kwargs.get("custom_print_log", None)

        # 训练总的步数
        num_training_steps = num_epochs * len(train_loader)

        if eval_steps:
            if self.metric is None:
                raise RuntimeError('Error: Metric can not be None!')
            if dev_loader is None:
                raise RuntimeError('Error: dev_loader can not be None!')

        # 运行的step数目
        global_step = 0

        # 进行num_epochs轮训练
        for epoch in range(num_epochs):
            # 用于统计训练集的损失
            total_loss = 0
            for step, data in enumerate(train_loader):
                X, y = data
                # 获取模型预测
                logits = self.model(X)
                loss = self.loss_fn(logits, y.long())  # 默认求mean
                total_loss += loss

                # 训练过程中，每个step的loss进行保存
                self.train_step_losses.append((global_step, loss.item()))

                if log_steps and global_step % log_steps == 0:
                    print(
                        f"[Train] epoch: {epoch}/{num_epochs}, step: {global_step}/{num_training_steps}, loss: {loss.item():.5f}")

                # 梯度反向传播，计算每个参数的梯度值
                loss.backward()

                if custom_print_log:
                    custom_print_log(self)

                # 小批量梯度下降进行参数更新
                self.optimizer.step()
                # 梯度归零
                self.optimizer.zero_grad()

                # 判断是否需要评价
                if eval_steps > 0 and global_step > 0 and \
                        (global_step % eval_steps == 0 or global_step == (num_training_steps - 1)):

                    dev_score, dev_loss = self.evaluate(dev_loader, global_step=global_step)
                    print(f"[Evaluate]  dev score: {dev_score:.5f}, dev loss: {dev_loss:.5f}")

                    # 将模型切换为训练模式
                    self.model.train()

                    # 如果当前指标为最优指标，保存该模型
                    if dev_score > self.best_score:
                        self.save_model(save_path)
                        print(
                            f"[Evaluate] best accuracy performence has been updated: {self.best_score:.5f} --> {dev_score:.5f}")
                        self.best_score = dev_score

                global_step += 1

            # 当前epoch 训练loss累计值
            trn_loss = (total_loss / len(train_loader)).item()
            # epoch粒度的训练loss保存
            self.train_epoch_losses.append(trn_loss)

        print("[Train] Training done!")

    # 模型评估阶段，使用'torch.no_grad()'控制不计算和存储梯度
    @torch.no_grad()
    def evaluate(self, dev_loader, **kwargs):
        assert self.metric is not None

        # 将模型设置为评估模式
        self.model.eval()

        global_step = kwargs.get("global_step", -1)

        # 用于统计训练集的损失
        total_loss = 0

        # 重置评价
        self.metric.reset()

        # 遍历验证集每个批次
        for batch_id, data in enumerate(dev_loader):
            X, y = data

            # 计算模型输出
            logits = self.model(X)

            # 计算损失函数
            loss = self.loss_fn(logits, y.long()).item()
            # 累积损失
            total_loss += loss

            # 累积评价
            self.metric.update(logits, y)

        dev_loss = (total_loss / len(dev_loader))
        dev_score = self.metric.accumulate()

        # 记录验证集loss
        if global_step != -1:
            self.dev_losses.append((global_step, dev_loss))
            self.dev_scores.append(dev_score)

        return dev_score, dev_loss

    # 模型评估阶段，使用'torch.no_grad()'控制不计算和存储梯度
    @torch.no_grad()
    def predict(self, x, **kwargs):
        # 将模型设置为评估模式
        self.model.eval()
        # 运行模型前向计算，得到预测值
        logits = self.model(x)
        return logits

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, model_path):
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)



class Accuracy():
    def __init__(self):
        """
        输入：
           - is_logist: outputs是logist还是激活后的值
        """
        # 用于统计正确的样本个数
        self.num_correct = 0
        # 用于统计样本的总数
        self.num_count = 0

        self.is_logist = True

    def update(self, outputs, labels):
        """
        输入：
           - outputs: 预测值, shape=[N,class_num]
           - labels: 标签值, shape=[N,1]
        """
        # 判断是二分类任务还是多分类任务，shape[1]=1时为二分类任务，shape[1]>1时为多分类任务
        if outputs.shape[1] == 1:  # 二分类
            outputs = torch.squeeze(outputs, dim=-1)
            if self.is_logist:
                # logist判断是否大于0
                preds = (outputs >= 0).float()
            else:
                # 如果不是logist，判断每个概率值是否大于0.5，当大于0.5时，类别为1，否则类别为0
                preds = (outputs >= 0.5).float()
        else:
            # 多分类时，使用'torch.argmax'计算最大元素索引作为类别
            preds = torch.argmax(outputs, dim=1).int()

        # 获取本批数据中预测正确的样本个数
        labels = torch.squeeze(labels, dim=-1)
        batch_correct = torch.sum((preds == labels).clone().detach()).cpu().numpy()  # 将张量转移到CPU
        batch_count = len(labels)

        # 更新num_correct 和 num_count
        self.num_correct += batch_correct
        self.num_count += batch_count

    def accumulate(self):
        # 使用累计的数据，计算总的指标
        if self.num_count == 0:
            return 0
        return self.num_correct / self.num_count

    def reset(self):
        # 重置正确的数目和总数
        self.num_correct = 0
        self.num_count = 0

    def name(self):
        return "Accuracy"
import matplotlib.pyplot as plt

# 可视化
def plot(runner, fig_name):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    train_items = runner.train_step_losses[::30]
    train_steps = [x[0] for x in train_items]
    train_losses = [x[1] for x in train_items]

    plt.plot(train_steps, train_losses, color='#8E004D', label="Train loss")
    if runner.dev_losses[0][0] != -1:
        dev_steps = [x[0] for x in runner.dev_losses]
        dev_losses = [x[1] for x in runner.dev_losses]
        plt.plot(dev_steps, dev_losses, color='#E20079', linestyle='--', label="Dev loss")
    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='x-large')
    plt.xlabel("step", fontsize='x-large')
    plt.legend(loc='upper right', fontsize='x-large')

    plt.subplot(1, 2, 2)
    # 绘制评价准确率变化曲线
    if runner.dev_losses[0][0] != -1:
        plt.plot(dev_steps, runner.dev_scores,
                 color='#E20079', linestyle="--", label="Dev accuracy")
    else:
        plt.plot(list(range(len(runner.dev_scores))), runner.dev_scores,
                 color='#E20079', linestyle="--", label="Dev accuracy")
    # 绘制坐标轴和图例
    plt.ylabel("score", fontsize='x-large')
    plt.xlabel("step", fontsize='x-large')
    plt.legend(loc='lower right', fontsize='x-large')

    plt.savefig(fig_name)
    plt.show()

def plot_training_loss(runner, fig_name, sample_step):
    plt.figure()
    train_items = runner.train_step_losses[::sample_step]
    train_steps = [x[0] for x in train_items]
    train_losses = [x[1] for x in train_items]
    plt.plot(train_steps, train_losses, color='#e4007f', label="Train loss")

    dev_steps = [x[0] for x in runner.dev_losses]
    dev_losses = [x[1] for x in runner.dev_losses]
    plt.plot(dev_steps, dev_losses, color='#f19ec2', linestyle='--', label="Dev loss")

    # 绘制坐标轴和图例
    plt.ylabel("loss", fontsize='large')
    plt.xlabel("step", fontsize='large')
    plt.legend(loc='upper right', fontsize='x-large')

    plt.savefig(fig_name)
    plt.show()