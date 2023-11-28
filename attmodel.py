import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
import torch.optim as optim



class LookupEmbedding(torch.nn.Module):

    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1).cuda()).cuda()
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1).cuda()).cuda()
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb




class GRUModel():
    """
            Parameters：
            - input_size: feature size
            - hidden_size: number of hidden units
            - output_size: number of output
            - num_layers: layers of LSTM to stack
        """

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).cuda()
        self.batch_norm = nn.BatchNorm1d(20).cuda()
        self.dropout = nn.Dropout(p=0.5).cuda()

        self.fc = nn.Linear(hidden_size, output_size).cuda()
        self.l2_reg = nn.Linear(hidden_size, hidden_size, bias=False).cuda()

    def forward(self, x, h):
        # self.gru.cuda()
        out, h = self.gru(x, h)
        out1 = self.batch_norm(out.cuda()).cuda()
        out2 = self.dropout(out1)
        out3 = torch.relu(out2)
        out4 = self.l2_reg(out3)
        out5 = self.fc(out4)
        return out5

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden

class muAtt():
    def __init__(self, emb_dim, meta_dim):
        super(muAtt).__init__()
        self.attention1 = nn.Linear(emb_dim, meta_dim).cuda()
        self.attention2 = nn.Linear(emb_dim, meta_dim).cuda()
        self.attention3 = nn.Linear(meta_dim, 1).cuda()
        self.fc = nn.Linear(emb_dim, emb_dim).cuda()
        self.batch_norm = nn.BatchNorm1d(10).cuda()

    def forward(self, s, l):
        attn1_out = F.tanh(self.attention1(l))
        attn2_out = F.tanh(self.attention2(s))
        attn = attn1_out + attn2_out
        attn1 = F.tanh(attn)
        attn_out = self.attention3(attn1).squeeze(2)
        attn_weights = F.softmax(attn_out, dim = 1).unsqueeze(2)
        #  Multi-level feature fusion  
        fused_out = (attn_weights * l) + ((1 - attn_weights) * s)
        #  Output layer  
        output = self.fc(fused_out)
        output1 = torch.sum(output, 1)
        output2 = self.batch_norm(output1)
        return output2


class Fusion(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        #短期特征
        self.gru = GRUModel(input_size=10, hidden_size=50, num_layers=1, output_size=10)
        #多级注意力融合长短期特征
        self.mtt = muAtt(emb_dim, meta_dim)

    def forward(self, x):
        # 通过GRU提取短期特征
        h = self.gru.init_hidden(x.shape[0]).cuda()
        s = self.gru.forward(x.cuda(), h.cuda()).cuda()  # s:128*10
        user_fea = self.mtt.forward(s.cuda(), x.cuda())

        return user_fea  #128*10


# 定义元生成器和元判别器的网络结构
class Generator(nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super(Generator, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(emb_dim, 128)
        self.prelu = nn.PReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, emb_dim * emb_dim)


    def forward(self, x):


        x = self.flatten(x)
        x = self.prelu(self.fc1(x))
        x = self.prelu(self.fc2(x))
        uid_emb = self.fc3(x)

        return uid_emb

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256).cuda()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1).cuda()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a= self.fc1(x.cuda()).cuda()
        b = self.relu(a).cuda()
        c = self.fc2(b).cuda()
        d = self.sigmoid(c).cuda()
        return d


class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()

        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                            torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = nn.Sequential(nn.Linear(emb_dim, meta_dim), nn.ReLU(),
                                           nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, x, seq_index):

        mask = (seq_index == 0).float()
        event_K = self.event_K(x)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        a = att * x
        his_fea = torch.sum(a, 1)
        output_ptu = self.decoder(his_fea)

        return output_ptu   #128*100

class Generatore(nn.Module):
    def __init__(self, emb_dim, meta_dim_0, max_seq_length=50, num_layers=1, num_heads=5, dropout=0.1):
        super(Generatore, self).__init__()

        self.position_encoding = PositionalEncoding(max_seq_length)
        self.transformer = Transformer(num_layers, num_heads, dropout)
        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, input_seq):

        # Positional encoding
        positional_encoded = self.position_encoding(input_seq)

        # Transformer layers
        output = self.transformer(positional_encoded)

        # Linear layer
        output = self.fc(output)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_length):
        super(PositionalEncoding, self).__init__()

        self.hidden_size = 50

        # Compute the positional encodings in advance
        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 10, 2) * (-math.log(10000.0) / 50))
        pe = torch.zeros(max_seq_length, 10)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.hidden_size)
        seq_length = x.size(1)
        a = self.pe[:, :seq_length, :]
        x = x + self.pe[:, :seq_length, :]
        return x

class Transformer(nn.Module):
    def __init__(self, num_layers, num_heads, dropout):
        super(Transformer, self).__init__()

        self.transformer = nn.Transformer(d_model=10, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=50,
                                          dropout=dropout)

    def forward(self, x):
        output = self.transformer(x, x)
        return output

class Discriminatore(nn.Module):
    def __init__(self, emb_dim, meta_dim_0):
        super(Discriminatore, self).__init__()

        self.embedding = nn.Embedding(emb_dim, meta_dim_0)
        self.fc = nn.Linear(200, 10).cuda()

    def forward(self, input_seq):
        # Embedding
        # embedded = self.embedding(input_seq)

        # Flatten
        flattened = input_seq.view(input_seq.size(0), -1)

        # Linear layer
        output = self.fc(flattened.cuda()).cuda()

        # Sigmoid activation
        output = torch.sigmoid(output)

        return output


class GMFBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, 1, False)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = emb[:, 0, :] * emb[:, 1, :]
        x = self.linear(x)
        return x.squeeze(1)


class DNNBase(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embedding = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.linear = torch.nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        emb = self.embedding.forward(x)
        x = torch.sum(self.linear(emb[:, 0, :]) * emb[:, 1, :], 1)
        return x


class MFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim_0):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.fusion = Fusion(emb_dim, meta_dim_0)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, 50)
        self.generator = Generator(emb_dim, meta_dim_0)
        self.generatorelec = Generatore(emb_dim, meta_dim_0, max_seq_length=50, num_layers=1, num_heads=5, dropout=0.1)
        self.discriminatorelec = Discriminatore(emb_dim, meta_dim_0)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.batch_norm = nn.BatchNorm1d(100)
        self.Encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(emb_dim, meta_dim_0),
            nn.PReLU(),
            nn.Linear(meta_dim_0, meta_dim_0),
            nn.PReLU(),
            nn.Linear(meta_dim_0, emb_dim))

    def forward(self, x, stage):
        if stage in ['train_src', 'test_src']:
            emb = self.src_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ['train_aug', 'test_aug']:
            emb = self.aug_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ['train_e', 'test_e']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.iid_embedding(x[:, 2:])
            seq_out = self.generatorelec(ufea)
            uid = self.discriminatorelec(ufea).unsqueeze(1)
            emb = torch.cat([uid, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return seq_out, ufea, output
        elif stage in ['train_pre', 'test_pre']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.iid_embedding(x[:, 2:])
            mapping_ptu= self.meta_net.forward(ufea, x[:, 2:])
            mapping_ptu = mapping_ptu.view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping_ptu)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output
        elif stage in ['train_meta', 'test_meta']:
            iid_emb = self.tgt_model.iid_embedding(x[:, 1].unsqueeze(1))  #目标域项目
            uid_emb_src = self.src_model.uid_embedding(x[:, 0].unsqueeze(1))  # 源域用户
            uid_emb_tgt = self.tgt_model.uid_embedding(x[:, 0])
            ufea = self.src_model.iid_embedding(x[:, 2:])  # 128*20*10序列
            output1 = self.fusion.forward(ufea.cuda())
            uid_emb =self.generator(output1).view(-1, self.emb_dim, self.emb_dim)
            a2 = self.generator(uid_emb_src).squeeze(1).view(-1, self.emb_dim, self.emb_dim)
            a3 = torch.sum(a2, 1).unsqueeze(1)
            uid_emb1 = torch.bmm(a3, uid_emb)
            emb = torch.cat([uid_emb1, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return output, uid_emb_tgt
        elif stage == 'train_map':
            src_emb = self.src_model.uid_embedding(x.unsqueeze(1)).squeeze().cuda()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.uid_embedding(x.unsqueeze(1)).squeeze().cuda()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.uid_embedding(x[:, 0].unsqueeze(1)).squeeze())
            emb = self.tgt_model.forward(x)
            emb[:, 0, :] = uid_emb
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x


class GMFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = GMFBase(uid_all, iid_all, emb_dim)
        self.tgt_model = GMFBase(uid_all, iid_all, emb_dim)
        self.aug_model = GMFBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.fusion = Fusion(emb_dim, meta_dim)
        self.generator = Generator(emb_dim, meta_dim)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.batch_norm = nn.BatchNorm1d(10)
        self.Encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(emb_dim, meta_dim),
            nn.PReLU(),
            nn.Linear(meta_dim, meta_dim),
            nn.PReLU(),
            nn.Linear(meta_dim, emb_dim))

    def forward(self, x, stage):
        if stage == 'train_src':
            x = self.src_model.forward(x)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x)
            return x
        elif stage in ['train_aug', 'test_aug']:
            x = self.aug_model.forward(x)
            return x
        elif stage in ['test_pre', 'train_pre']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            mapping_ptu = self.meta_net.forward(ufea, x[:, 2:])
            mapping_ptu = mapping_ptu.view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping_ptu)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return output.squeeze(1)
        elif stage in ['train_meta', 'test_meta']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))
            uid_emb_tgt = self.tgt_model.embedding.uid_embedding(x[:, 0])
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            output1 = self.fusion.forward(ufea.cuda())
            uid_emb = self.generator(output1).view(-1, self.emb_dim, self.emb_dim)
            a2 = self.generator(uid_emb_src).squeeze(1).view(-1, self.emb_dim, self.emb_dim)
            a3 = torch.sum(a2, 1).unsqueeze(1)
            uid_emb1 = torch.bmm(a3, uid_emb)
            emb = torch.cat([uid_emb1, iid_emb], 1)
            output = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return output, uid_emb_tgt
        elif stage == 'train_map':
            src_emb = self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze().cuda()
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze().cuda()
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = self.tgt_model.linear(emb[:, 0, :] * emb[:, 1, :])
            return x.squeeze(1)


class DNNBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, num_fields, emb_dim, meta_dim):
        super().__init__()
        self.num_fields = num_fields
        self.emb_dim = emb_dim
        self.src_model = DNNBase(uid_all, iid_all, emb_dim)
        self.tgt_model = DNNBase(uid_all, iid_all, emb_dim)
        self.aug_model = DNNBase(uid_all, iid_all, emb_dim)
        self.meta_net = MetaNet(emb_dim, meta_dim)
        self.fusion = Fusion(emb_dim, meta_dim)
        self.generator = Generator(emb_dim, meta_dim)
        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.Encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(emb_dim, meta_dim),
            nn.PReLU(),
            nn.Linear(meta_dim, meta_dim),
            nn.PReLU(),
            nn.Linear(meta_dim, emb_dim))

    def forward(self, x, stage):
        if stage == 'train_src':
            x = self.src_model.forward(x)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            x = self.tgt_model.forward(x)
            return x
        elif stage in ['train_aug', 'test_aug']:
            x = self.aug_model.forward(x)
            return x
        elif stage in ['test_pre', 'train_pre']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            mapping_ptu = self.meta_net.forward(ufea, x[:, 2:])
            mapping_ptu = mapping_ptu.view(-1, self.emb_dim, self.emb_dim)
            uid_emb = torch.bmm(uid_emb_src, mapping_ptu)
            emb = torch.cat([uid_emb, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            return output
        elif stage in ['train_meta', 'test_meta']:
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            uid_emb_src = self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1)))
            uid_emb_tgt = self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))
            ufea = self.src_model.embedding.iid_embedding(x[:, 2:])
            output1 = self.fusion.forward(ufea.cuda())
            uid_emb = self.generator(output1).view(-1, self.emb_dim, self.emb_dim)
            a2 = self.generator(uid_emb_src).squeeze(1).view(-1, self.emb_dim, self.emb_dim)
            a3 = torch.sum(a2, 1).unsqueeze(1)
            uid_emb1 = torch.bmm(a3, uid_emb)
            emb = torch.cat([uid_emb1, iid_emb], 1)
            output = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            return output, uid_emb_tgt
        elif stage == 'train_map':
            src_emb = self.src_model.linear(self.src_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            src_emb = self.mapping.forward(src_emb)
            tgt_emb = self.tgt_model.linear(self.tgt_model.embedding.uid_embedding(x.unsqueeze(1)).squeeze())
            return src_emb, tgt_emb
        elif stage == 'test_map':
            uid_emb = self.mapping.forward(self.src_model.linear(self.src_model.embedding.uid_embedding(x[:, 0].unsqueeze(1))))
            iid_emb = self.tgt_model.embedding.iid_embedding(x[:, 1].unsqueeze(1))
            emb = torch.cat([uid_emb, iid_emb], 1)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], 1)
            return x
