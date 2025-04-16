import os
import torch

class RunnerV2_1(object):
    def __init__(self, model, optimizer, metric, loss_fn, **kwargs):
        # 初始化 RunnerV2_1 类的实例
        self.model = model  # 存储模型
        self.optimizer = optimizer  # 存储优化器
        self.loss_fn = loss_fn  # 存储损失函数
        self.metric = metric  # 存储评估指标

        # 记录训练过程中的评估指标变化情况
        self.train_scores = []  # 用于存储每个训练轮次的得分
        self.dev_scores = []    # 用于存储每个验证轮次的得分

        # 记录训练过程中的损失变化情况
        self.train_loss = []    # 用于存储每个训练轮次的损失
        self.dev_loss = []      # 用于存储每个验证轮次的损失

    def train(self, train_set, dev_set, **kwargs):
        # 传入训练轮数，如果没有传入值则默认为0
        num_epochs = kwargs.get("num_epochs", 0)
        # 传入日志打印频率，如果没有传入值则默认为100
        log_epochs = kwargs.get("log_epochs", 100)

        # 传入模型保存路径
        save_dir = kwargs.get("save_dir", None)

        # 记录全局最优指标
        best_score = 0
        # 进行 num_epochs 轮训练
        for epoch in range(num_epochs):
            X, y = train_set  # 解包训练集数据
            # 获取模型预测
            logits = self.model(X)  # 模型输出

            # 计算交叉熵损失
            trn_loss = self.loss_fn(logits, y)  # 返回一个张量

            # 记录当前训练损失
            self.train_loss.append(trn_loss.item())
            # 计算评估指标
            trn_score = self.metric(logits, y)  # 计算训练集得分
            self.train_scores.append(trn_score)  # 记录得分

            # 反向传播计算梯度
            self.loss_fn.backward()

            # 参数更新
            self.optimizer.step()  # 使用优化器更新模型参数

            # 评估验证集
            dev_score, dev_loss = self.evaluate(dev_set)
            # 如果当前指标为最优指标，保存该模型
            if dev_score > best_score:
                print(f"[Evaluate] best accuracy performance has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score  # 更新最佳得分
                if save_dir:  # 如果指定了保存路径
                    self.save_model(save_dir)  # 保存模型

            # 打印训练过程中的损失和状态
            if log_epochs and epoch % log_epochs == 0:
                print(f"[Train] epoch: {epoch}/{num_epochs}, loss: {trn_loss.item()}")

    def evaluate(self, data_set):
        # 解包验证集数据
        X, y = data_set
        # 计算模型输出
        logits = self.model(X)  # 模型预测
        # 计算损失函数
        loss = self.loss_fn(logits, y).item()  # 计算验证集损失
        self.dev_loss.append(loss)  # 记录验证损失
        # 计算评估指标
        score = self.metric(logits, y)  # 计算验证集得分
        self.dev_scores.append(score)  # 记录得分
        return score, loss  # 返回得分和损失

    def predict(self, X):
        # 对输入数据进行预测
        return self.model(X)

    def save_model(self, save_dir):
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 对模型每层参数分别进行保存，保存文件名称与该层名称相同
        for layer in self.model.layers:  # 遍历所有层
            if isinstance(layer.params, dict):  # 检查层参数是否为字典
                torch.save(layer.params, os.path.join(save_dir, layer.name + ".pdparams"))  # 保存参数

    def load_model(self, model_dir):
        # 获取所有层参数名称和保存路径之间的对应关系
        model_file_names = os.listdir(model_dir)  # 列出模型目录下的所有文件
        name_file_dict = {}
        for file_name in model_file_names:
            name = file_name.replace(".pdparams", "")  # 提取层名称
            name_file_dict[name] = os.path.join(model_dir, file_name)  # 构建名称与路径的字典

        # 加载每层参数
        for layer in self.model.layers:  # 遍历所有层
            if isinstance(layer.params, dict):  # 检查层参数是否为字典
                name = layer.name  # 获取层名称
                file_path = name_file_dict[name]  # 获取对应的文件路径
                layer.params = torch.load(file_path)  # 加载参数
