
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tensorflow import keras
from sklearn.metrics import roc_auc_score, ndcg_score

import pandas as pd
import numpy as np
import tqdm
import matplotlib
matplotlib.use('Agg')
import torch
torch.backends.cudnn.enabled = False
import torch.optim as optim
import torch.nn as nn

from attmodel import MFBasedModel, GMFBasedModel, DNNBasedModel, Generator, Discriminator, Generatore, Discriminatore


class Run():
    def __init__(self,config):
        self.use_cuda = config['use_cuda']
        self.config = config
        self.base_model = config['base_model']
        self.root = config['root']
        self.ratio = config['ratio']
        self.task = config['task']
        self.src = config['src_tgt_pairs'][self.task]['src']
        self.tgt = config['src_tgt_pairs'][self.task]['tgt']
        self.uid_all = config['src_tgt_pairs'][self.task]['uid']
        self.iid_all = config['src_tgt_pairs'][self.task]['iid']
        self.batchsize_src = config['src_tgt_pairs'][self.task]['batchsize_src']
        self.batchsize_tgt = config['src_tgt_pairs'][self.task]['batchsize_tgt']
        self.batchsize_meta = config['src_tgt_pairs'][self.task]['batchsize_meta']
        self.batchsize_map = config['src_tgt_pairs'][self.task]['batchsize_map']
        self.batchsize_test = config['src_tgt_pairs'][self.task]['batchsize_test']
        self.batchsize_aug = self.batchsize_src

        self.epoch = config['epoch']
        self.emb_dim = config['emb_dim']
        self.meta_dim = config['meta_dim']
        self.num_fields = config['num_fields']
        self.lr = config['lr']
        self.wd = config['wd']

        self.generator = Generator(self.emb_dim, self.meta_dim)
        self.discriminator = Discriminator(self.emb_dim)

        self.discriminatorelec = Discriminatore(self.emb_dim, self.meta_dim)
        self.loss_fn = nn.BCELoss()

        self.input_root = './data/ready/_' + str(int(self.ratio[0] * 10)) + '_' + str(int(self.ratio[1] * 10)) + \
            '/tgt_' + self.tgt + '_src_' + self.src
        self.src_path = self.input_root + '/train_src.csv'
        self.tgt_path = self.input_root + '/train_tgt.csv'
        self.meta_path = self.input_root + '/train_meta.csv'
        self.test_path = self.input_root + '/test.csv'
        self.supp_path = self.input_root + '/train_supp.csv'
        self.quer_path = self.input_root + '/train_quer.csv'

        self.results = {'tgt_mae': 10, 'tgt_rmse': 10,
                        'aug_mae': 10, 'aug_rmse': 10,
                        'emcdr_mae': 10, 'emcdr_rmse': 10,
                        'ptupcdr_mae': 10, 'ptupcdr_rmse': 10,
                        'mafcdr_mae': 10, 'mafcdr_rmse': 10}

    def seq_extractor(self, x):
        x = x.rstrip(']').lstrip('[').split(', ')
        for i in range(len(x)):
            try:
                x[i] = int(x[i])
            except:
                x[i] = self.iid_all
        return np.array(x)

    def read_log_data(self, path, batchsize, history=False):
        if not history:
            cols = ['uid', 'iid', 'y']
            x_col = ['uid', 'iid']
            y_col = ['y']
            data = pd.read_csv(path, header=None)
            data.columns = cols
            X = torch.tensor(data[x_col].values, dtype=torch.long)
            y = torch.tensor(data[y_col].values, dtype=torch.long)

            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True)
            return data_iter
        else:
            data = pd.read_csv(path, header=None)
            cols = ['uid', 'iid', 'y', 'pos_seq']
            x_col = ['uid', 'iid']
            y_col = ['y']

            data.columns = cols

            # pos_seq_list = [self.seq_extractor(seq) for seq in data.pos_seq]
            #
            # pos_seq_tensor = [torch.tensor(seq) for seq in pos_seq_list]
            # pos_seq = pad_sequence(pos_seq_tensor, batch_first=True, padding_value=0)
            pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20,
                                                                 padding='post')

            pos_seq = torch.tensor(pos_seq, dtype=torch.long)
            id_fea = torch.tensor(data[x_col].values, dtype=torch.long)
            X = torch.cat([id_fea, pos_seq], dim=1)
            y = torch.tensor(data[y_col].values, dtype=torch.long)
            if self.use_cuda:
                X = X.cuda()
                y = y.cuda()
            dataset = TensorDataset(X, y)
            data_iter = DataLoader(dataset, batchsize, shuffle=True, drop_last=True)
            return data_iter

    # def read_e_data(self):
    #     data = pd.read_csv(self.meta_path, header=None)
    #     cols = ['uid', 'iid', 'y', 'pos_seq']
    #     x_col = ['uid', 'iid']
    #     s_col = ['pos_seq']
    #     y_col = ['y']
    #     data.columns = cols
    #     pos_seq = keras.preprocessing.sequence.pad_sequences(data.pos_seq.map(self.seq_extractor), maxlen=20,
    #                                                          padding='post')
    #
    #     pos_seq = torch.tensor(pos_seq, dtype=torch.long)
    #     x = torch.tensor(data[x_col].values, dtype=torch.long)
    #     s = torch.tensor(data[s_col].values, dtype=torch.long)
    #     y = torch.tensor(data[y_col].values, dtype=torch.long)
    #     if self.use_cuda:
    #         X = x.cuda()
    #         s = s.cuda()
    #         y = y.cuda()
    #
    #     dataset = TensorDataset(X, s, y)
    #     date_iter = DataLoader(dataset, self.batchsize_meta, shuffle=True, drop_last=True)
    #     return date_iter


    def read_map_data(self):
        cols = ['uid', 'iid', 'y', 'pos_seq']

        data = pd.read_csv(self.meta_path, header=None)
        data.columns = cols
        X = torch.tensor(data['uid'].unique(), dtype=torch.long)
        y = torch.tensor(np.array(range(X.shape[0])), dtype=torch.long)
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_map, shuffle=True)
        return data_iter


    def read_aug_data(self):
        cols_train = ['uid', 'iid', 'y']
        x_col = ['uid', 'iid']
        y_col = ['y']
        src = pd.read_csv(self.src_path, header=None)
        src.columns = cols_train
        tgt = pd.read_csv(self.tgt_path, header=None)
        tgt.columns = cols_train

        X_src = torch.tensor(src[x_col].values, dtype=torch.long)
        y_src = torch.tensor(src[y_col].values, dtype=torch.long)
        X_tgt = torch.tensor(tgt[x_col].values, dtype=torch.long)
        y_tgt = torch.tensor(tgt[y_col].values, dtype=torch.long)
        X = torch.cat([X_src, X_tgt])
        y = torch.cat([y_src, y_tgt])
        if self.use_cuda:
            X = X.cuda()
            y = y.cuda()
        dataset = TensorDataset(X, y)
        data_iter = DataLoader(dataset, self.batchsize_aug, shuffle=True)

        return data_iter

    def get_data(self):
        print('========Reading data========')
        data_src = self.read_log_data(self.src_path, self.batchsize_src)
        print('src {} iter / batchsize = {} '.format(len(data_src), self.batchsize_src))

        data_tgt = self.read_log_data(self.tgt_path, self.batchsize_tgt)
        print('tgt {} iter / batchsize = {} '.format(len(data_tgt), self.batchsize_tgt))

        data_meta = self.read_log_data(self.meta_path, self.batchsize_meta, history=True)
        print('meta {} iter / batchsize = {} '.format(len(data_meta), self.batchsize_meta))

        data_map = self.read_map_data()
        print('map {} iter / batchsize = {} '.format(len(data_map), self.batchsize_map))

        data_aug = self.read_aug_data()
        print('aug {} iter / batchsize = {} '.format(len(data_aug), self.batchsize_aug))

        data_test = self.read_log_data(self.test_path, self.batchsize_test, history=True)
        print('test {} iter / batchsize = {} '.format(len(data_test), self.batchsize_test))

        data_supp = self.read_log_data(self.supp_path, self.batchsize_meta, history=True)
        print('supp {} iter / batchsize = {} '.format(len(data_supp), self.batchsize_meta))

        data_quer = self.read_log_data(self.quer_path, self.batchsize_meta, history=True)
        print('quer {} iter / batchsize = {} '.format(len(data_quer), self.batchsize_meta))

        return data_src, data_tgt, data_meta, data_map, data_aug, data_test, data_supp, data_quer,

    def get_model(self):
        if self.base_model == 'MF':

            model = MFBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        elif self.base_model == 'DNN':
            model = DNNBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        elif self.base_model == 'GMF':
            model = GMFBasedModel(self.uid_all, self.iid_all, self.num_fields, self.emb_dim, self.meta_dim)
        else:
            raise ValueError('Unknown base model: ' + self.base_model)
        return model.cuda() if self.use_cuda else model

    def get_optimizer(self, model):
        optimizer_src = torch.optim.Adam(params=model.src_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_tgt = torch.optim.Adam(params=model.tgt_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_meta = torch.optim.Adam(params=model.meta_net.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_aug = torch.optim.Adam(params=model.aug_model.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_map = torch.optim.Adam(params=model.mapping.parameters(), lr=self.lr, weight_decay=self.wd)
        optimizer_outloop = torch.optim.Adam(params=model.meta_net.parameters(), lr=self.lr, weight_decay=self.wd)
        generator_optimizer = torch.optim.Adam(params=model.generator.parameters(), lr=1e-3, weight_decay=self.wd)
        discriminator_optimizer = torch.optim.Adam(params=self.discriminator.parameters(), lr=1e-3, weight_decay=self.wd)
        optim_me = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        generator_optimizer_e = torch.optim.Adam(params=model.generatorelec.parameters(), lr=0.01, weight_decay=self.wd)
        discriminator_optimizer_e = torch.optim.Adam(params=model.discriminatorelec.parameters(), lr=0.01, weight_decay=self.wd)

        return optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map, optimizer_outloop, generator_optimizer, discriminator_optimizer, optim_me, generator_optimizer_e, discriminator_optimizer_e

    def eval_mae(self, model, data_loader, stage):
        print('Evaluating MAE:')
        model.eval()
        targets, predicts = list(), list()
        loss = torch.nn.L1Loss()
        mse_loss = torch.nn.MSELoss()
        with torch.no_grad():
            for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
                if stage == 'test_e':
                    seq_out, ufea, pred = model(X.cuda(), stage)
                elif stage == 'test_meta':
                    pred, uid = model(X.cuda(), stage)
                else:
                    pred = model(X.cuda(), stage)
                targets.extend(y.squeeze(1).tolist())
                predicts.extend(pred.tolist())
        targets = torch.tensor(targets).float()
        predicts = torch.tensor(predicts)

        return loss(targets, predicts).item(), torch.sqrt(mse_loss(targets, predicts)).item()


    def train(self, data_loader, model, criterion, optimizer, epoch, stage, mapping=False):
        print('Training Epoch {}:'.format(epoch + 1))
        model.cuda()
        model.train()
        # for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
        for X, y in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):

            if mapping:
                src_emb, tgt_emb = model(X.cuda(), stage)
                loss = criterion(src_emb.cuda(), tgt_emb.cuda())
            else:
                pred = model(X.cuda(), stage)
                loss = criterion(pred.cuda(), y.squeeze().float().cuda())
            model.zero_grad()
            loss.backward()
            optimizer.step()



    def train_MAN(self, model, data_supp, data_quer, generator_optimizer, discriminator_optimizer, optim_me, i, stage):
        print('Training Epoch {}:'.format(i + 1))
        model.cuda()
        model.train()
        loss_fn = nn.HingeEmbeddingLoss()
        # X和Y分别是元任务的输入和输出
        for (supp_x, supp_y), (quer_x, quer_y) in tqdm.tqdm(zip(data_supp, data_quer), smoothing=0, mininterval=1.0):
            losses_q = []
            for i in range(1):
                loss_list = []
                preds, uid_emb_tgt = model(supp_x, stage)
                loss = F.mse_loss(preds.unsqueeze(1), supp_y.squeeze().float())
                loss_list.append(loss)
                loss = torch.stack(loss_list).mean(0)
                # model.zero_grad()
                # loss.backward()
                # generator_optimizer.step()
                # 计算元判别器的损失
                fake_params = model.generator.fc1.weight
                real_params = uid_emb_tgt
                real_preds, c = model(supp_x, stage)
                fake_preds, d = model(supp_x, stage)
                real_labels = torch.ones_like(real_preds)
                fake_labels = torch.zeros_like(fake_preds)
                a = self.discriminator(real_params).squeeze(1) #判别器判别为真样本
                b = self.discriminator(fake_params).squeeze(1) #判别器判别为假样本
                real_loss = self.loss_fn(a, real_labels) #判别器判别为真样本的损失
                fake_loss = self.loss_fn(b, fake_labels) #判别器判别为假样本的损失
                discriminator_loss = real_loss + fake_loss  #判别器一共的损失和
                # 更新元生成器和元判别器的参数
                generator_optimizer.zero_grad()
                discriminator_optimizer.zero_grad()
                generator_loss = self.loss_fn(self.discriminator(fake_params).squeeze(1), real_labels)
                #生成器损失，假样本和真实的1样本的损失

                gen_dis = generator_loss + discriminator_loss + loss
                gen_dis.backward()
                generator_optimizer.step()
                discriminator_optimizer.step()

            quer_y_pred, x = model(quer_x, stage)
            loss_q = F.mse_loss(quer_y_pred.unsqueeze(1), quer_y.squeeze().float())
            losses_q.append(loss_q)
            losses_q = torch.stack(losses_q).mean(0)
            optim_me.zero_grad()
            losses_q.backward()
            optim_me.step()


    def etrain(self, model, data_meta, generator_optimizer_elec, discriminator_optimizer_elec, i, stage):
        print('Training Epoch {}:'.format(i + 1))
        model.cuda()
        model.train()
        for X, y in tqdm.tqdm(data_meta, smoothing=0, mininterval=1.0):
            loss_list = []
            seq_fake, seq_real, pred = model(X, stage)
            loss = F.mse_loss(pred, y.float())
            loss_list.append(loss)
            loss = torch.stack(loss_list).mean(0)
            model.zero_grad()
            loss.backward(retain_graph=True)
            discriminator_optimizer_elec.step()
            seq = seq_real.clone()
            generated_seq_length = 10
            generated_seq_extracted = seq_fake[:, :generated_seq_length, :]
            start_index = 5  # 替换的起始位置
            seq_real[:, start_index: start_index + generated_seq_length, :] = generated_seq_extracted
            dis_output = self.discriminatorelec(torch.cat([seq, seq_real], dim=0))
            fake = self.discriminatorelec(seq_fake)
            # 创建判别器和生成器的标签
            real_labels = torch.ones_like(fake)  # 真实序列的标签为1
            fake_labels = torch.zeros_like(fake)  # 生成序列的标签为0
            dis_label = torch.cat([real_labels, fake_labels], dim=0)
            # 将真实序列和生成序列输入判别器进行预测

            # 计算生成器损失并进行反向传播和优化
            generator_loss = self.loss_fn(fake, real_labels)
            # 计算判别器损失并进行反向传播和优化
            discriminator_loss = self.loss_fn(dis_output, dis_label)
            loss_gan = generator_loss + discriminator_loss
            generator_optimizer_elec.zero_grad()
            discriminator_optimizer_elec.zero_grad()
            loss_gan.backward()
            generator_optimizer_elec.step()
            discriminator_optimizer_elec.step()


    def update_results(self, mae, rmse, phase):
        if mae < self.results[phase + '_mae']:
            self.results[phase + '_mae'] = mae
        if rmse < self.results[phase + '_rmse']:
            self.results[phase + '_rmse'] = rmse

    def TgtOnly(self, model, data_tgt, data_test, criterion, optimizer):
        print('=========TgtOnly========')
        for i in range(self.epoch):
            self.train(data_tgt, model, criterion, optimizer, i, stage='train_tgt')
            mae, rmse = self.eval_mae(model, data_test, stage='test_tgt')
            self.update_results(mae, rmse, 'tgt')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def DataAug(self, model, data_aug, data_test, criterion, optimizer):
        print('=========DataAug========')
        for i in range(self.epoch):
            self.train(data_aug, model, criterion, optimizer, i, stage='train_aug')
            mae, rmse = self.eval_mae(model, data_test, stage='test_aug')
            self.update_results(mae, rmse, 'aug')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

    def CDR(self, model, data_src, data_map, data_meta, data_test, data_supp, data_quer,
            criterion, optimizer_src, optimizer_map, optimizer_meta, generator_optimizer, discriminator_optimizer, optim_me):

        print('=====CDR Pretraining=====')
        for i in range(self.epoch):
            self.train(data_src, model, criterion, optimizer_src, i, stage='train_src')


        print('==========EMCDR==========')
        for i in range(self.epoch):
            self.train(data_map, model, criterion, optimizer_map, i, stage='train_map', mapping=True)
            mae, rmse = self.eval_mae(model, data_test, stage='test_map')
            self.update_results(mae, rmse, 'emcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))


        print('==========PTUPCDR==========')
        for i in range(self.epoch):
            # data_meta : support set Dataloader and query set Dataloader
            model.cuda()
            self.train(data_meta, model, criterion, optimizer_meta, i, stage='train_pre')
            mae, rmse = self.eval_mae(model, data_test, stage='test_pre')
            self.update_results(mae, rmse, 'ptupcdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))

        print('==========MAFCDR==========')
        for i in range(self.epoch):
            model.cuda()
            self.train_MAN(model, data_supp, data_quer, generator_optimizer, discriminator_optimizer, optim_me, i, stage = 'train_meta')
            mae, rmse = self.eval_mae(model, data_test, stage='test_meta')
            self.update_results(mae, rmse, 'mlucdr')
            print('MAE: {} RMSE: {}'.format(mae, rmse))


    def main(self):

        model = self.get_model().cuda()
        data_src, data_tgt, data_meta, data_map, data_aug, data_test, data_supp, data_quer = self.get_data()
        optimizer_src, optimizer_tgt, optimizer_meta, optimizer_aug, optimizer_map, optimizer_outloop, generator_optimizer, discriminator_optimizer, optim_me, generator_optimizer_e, discriminator_optimizer_e = self.get_optimizer(
            model)
        criterion = torch.nn.MSELoss().cuda()
        self.TgtOnly(model, data_tgt, data_test, criterion, optimizer_tgt)
        self.DataAug(model, data_aug, data_test, criterion, optimizer_aug)
        self.CDR(model, data_src, data_map, data_meta, data_test, data_supp, data_quer,
                 criterion, optimizer_src, optimizer_map, optimizer_meta, generator_optimizer, discriminator_optimizer, optim_me)
        print(self.results)
