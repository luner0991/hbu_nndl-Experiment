# runner.py
import os
import torch
class Runner(object):
    class Runner(object):
        def __init__(self, model, optimizer, loss_fn, metric):
            # 优化器和损失函数为None,不再关注

            # 模型
            self.model = model
            # 评估指标
            self.metric = metric
            # 优化器
            self.optimizer = optimizer

        def train(self, dataset, reg_lambda, model_dir):
            X, y = dataset
            self.optimizer(self.model, X, y, reg_lambda)

            # 保存模型
            self.save_model(model_dir)

        def evaluate(self, dataset, **kwargs):
            X, y = dataset

            y_pred = self.model(X)
            result = self.metric(y_pred, y)

            return result

        def predict(self, X, **kwargs):
            return self.model(X)

        def save_model(self, model_dir):
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            params_saved_path = os.path.join(model_dir, 'params.pdtensor')
            torch.save(model.params, params_saved_path)

        def load_model(self, model_dir):
            params_saved_path = os.path.join(model_dir, 'params.pdtensor')
            self.model.params = torch.load(params_saved_path)


