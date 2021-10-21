import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder


class BASEModel(nn.Module):
    def __init__(self, hyp_params):
        super(BASEModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 60, 60, 60
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = (self.d_l + self.d_a + self.d_v)

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=4)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=4)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=4)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def backbone(self, x_l, x_a, x_v):
        """
         text, audio, and vision should have dimension [batch_size, seq_len, n_features]
         """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            h_ls = self.trans_l_mem(proj_x_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            h_as = self.trans_a_mem(proj_x_a)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            h_vs = self.trans_v_mem(proj_x_v)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        return last_h_a, last_h_l, last_h_v

    def forward(self, x_l, x_a, x_v):

        last_h_a, last_h_l, last_h_v = self.backbone(x_l, x_a, x_v)
        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output, last_hs


class CombModel(nn.Module):
    def __init__(self, hyp_params):
        super(CombModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 60, 60, 60
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_l

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=4)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=4)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=4)

        # 1. Temporal convolutional layers
        self.proj_ls = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_as = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_vs = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mems = self.get_network(self_type='l_mem', layers=4)
        self.trans_a_mems = self.get_network(self_type='a_mem', layers=4)
        self.trans_v_mems = self.get_network(self_type='v_mem', layers=4)

        # Projection layers
        self.proj_l_1 = nn.Linear(combined_dim, combined_dim)
        self.proj_l_2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer_l_mu = nn.Linear(combined_dim, output_dim)
        self.out_layer_l_v = nn.Linear(combined_dim, output_dim)
        self.out_layer_l_alpha = nn.Linear(combined_dim, output_dim)
        self.out_layer_l_beta = nn.Linear(combined_dim, output_dim)
        self.proj_a_1 = nn.Linear(combined_dim, combined_dim)
        self.proj_a_2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer_a_mu = nn.Linear(combined_dim, output_dim)
        self.out_layer_a_v = nn.Linear(combined_dim, output_dim)
        self.out_layer_a_alpha = nn.Linear(combined_dim, output_dim)
        self.out_layer_a_beta = nn.Linear(combined_dim, output_dim)
        self.proj_v_1 = nn.Linear(combined_dim, combined_dim)
        self.proj_v_2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer_v_mu = nn.Linear(combined_dim, output_dim)
        self.out_layer_v_v = nn.Linear(combined_dim, output_dim)
        self.out_layer_v_alpha = nn.Linear(combined_dim, output_dim)
        self.out_layer_v_beta = nn.Linear(combined_dim, output_dim)

        self.proj1 = nn.Linear(3*combined_dim, 3*combined_dim)
        self.proj2 = nn.Linear(3*combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        self.out_layer_v = nn.Linear(combined_dim, output_dim)
        self.out_layer_alpha = nn.Linear(combined_dim, output_dim)
        self.out_layer_beta = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def backbone(self, x_l, x_a, x_v):
        """
         text, audio, and vision should have dimension [batch_size, seq_len, n_features]
         """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            h_ls = self.trans_l_mem(proj_x_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            h_as = self.trans_a_mem(proj_x_a)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            h_vs = self.trans_v_mem(proj_x_v)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        return last_h_a, last_h_l, last_h_v

    def backbone2(self, x_l, x_a, x_v):
        """
         text, audio, and vision should have dimension [batch_size, seq_len, n_features]
         """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_ls(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_as(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_vs(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            h_ls = self.trans_l_mems(proj_x_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            h_as = self.trans_a_mems(proj_x_a)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            h_vs = self.trans_v_mems(proj_x_v)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        return last_h_a, last_h_l, last_h_v

    def evidence(self, x):
        # return tf.exp(x)
        return F.softplus(x)

    def split(self, mu, logv, logalpha, logbeta):
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta

    def moe_nig(self, u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
        u = ((la1+2*alpha1) * u1 + u2 * (la2+2*alpha2)) / ((la1+2*alpha1) + (la2+2*alpha2))
        la = la1 + la2
        alpha = alpha1 + alpha2 + 0.5
        beta = beta1 + beta2 + 0.5 * (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
        return u, la, alpha, beta

    def forward(self, x_l, x_a, x_v):

        last_h_a, last_h_l, last_h_v = self.backbone(x_l, x_a, x_v)
        last_h_as, last_h_ls, last_h_vs = self.backbone2(x_l, x_a, x_v)
        last_hs = torch.cat([last_h_ls, last_h_as, last_h_vs], dim=1)
        # if self.partial_mode == 3:
        #     last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj_l = self.proj_l_2(F.dropout(F.relu(self.proj_l_1(last_h_l)), p=self.out_dropout, training=self.training))
        # last_hs_proj_l += last_h_l
        last_hs_proj_a = self.proj_a_2(F.dropout(F.relu(self.proj_a_1(last_h_a)), p=self.out_dropout, training=self.training))
        # last_hs_proj_a += last_h_a
        last_hs_proj_v = self.proj_v_2(F.dropout(F.relu(self.proj_v_1(last_h_v)), p=self.out_dropout, training=self.training))
        # last_hs_proj_v += last_h_v

        output_l_mu = self.out_layer_l_mu(last_hs_proj_l)
        output_l_v = self.out_layer_l_v(last_hs_proj_l)
        output_l_alpha = self.out_layer_l_alpha(last_hs_proj_l)
        output_l_beta = self.out_layer_l_beta(last_hs_proj_l)
        output_a_mu = self.out_layer_a_mu(last_hs_proj_a)
        output_a_v = self.out_layer_a_v(last_hs_proj_a)
        output_a_alpha = self.out_layer_a_alpha(last_hs_proj_a)
        output_a_beta = self.out_layer_a_beta(last_hs_proj_a)
        output_v_mu = self.out_layer_v_mu(last_hs_proj_v)
        output_v_v = self.out_layer_v_v(last_hs_proj_v)
        output_v_alpha = self.out_layer_v_alpha(last_hs_proj_v)
        output_v_beta = self.out_layer_v_beta(last_hs_proj_v)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        output = self.out_layer(last_hs_proj)
        output_v = self.out_layer_v(last_hs_proj)
        output_alpha = self.out_layer_alpha(last_hs_proj)
        output_beta = self.out_layer_beta(last_hs_proj)
        output, output_v, output_alpha, output_beta = self.split(output, output_v, output_alpha, output_beta)

        mu_l, v_l, alpha_l, beta_l = self.split(output_l_mu, output_l_v, output_l_alpha, output_l_beta)
        mu_a, v_a, alpha_a, beta_a = self.split(output_a_mu, output_a_v, output_a_alpha, output_a_beta)
        mu_v, v_v, alpha_v, beta_v = self.split(output_v_mu, output_v_v, output_v_alpha, output_v_beta)

        return mu_l, v_l, alpha_l, beta_l, mu_a, v_a, alpha_a, beta_a, mu_v, v_v, alpha_v, beta_v, output, output_v,\
            output_alpha, output_beta


class MOEModel(nn.Module):
    def __init__(self, hyp_params):
        super(MOEModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 60, 60, 60
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_l

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=4)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=4)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=4)

        # Projection layers
        self.proj_l_1 = nn.Linear(combined_dim, combined_dim)
        self.proj_l_2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer_l_mu = nn.Linear(combined_dim, output_dim)
        self.out_layer_l_v = nn.Linear(combined_dim, output_dim)
        self.out_layer_l_alpha = nn.Linear(combined_dim, output_dim)
        self.out_layer_l_beta = nn.Linear(combined_dim, output_dim)
        self.proj_a_1 = nn.Linear(combined_dim, combined_dim)
        self.proj_a_2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer_a_mu = nn.Linear(combined_dim, output_dim)
        self.out_layer_a_v = nn.Linear(combined_dim, output_dim)
        self.out_layer_a_alpha = nn.Linear(combined_dim, output_dim)
        self.out_layer_a_beta = nn.Linear(combined_dim, output_dim)
        self.proj_v_1 = nn.Linear(combined_dim, combined_dim)
        self.proj_v_2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer_v_mu = nn.Linear(combined_dim, output_dim)
        self.out_layer_v_v = nn.Linear(combined_dim, output_dim)
        self.out_layer_v_alpha = nn.Linear(combined_dim, output_dim)
        self.out_layer_v_beta = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def backbone(self, x_l, x_a, x_v):
        """
         text, audio, and vision should have dimension [batch_size, seq_len, n_features]
         """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            h_ls = self.trans_l_mem(proj_x_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            h_as = self.trans_a_mem(proj_x_a)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            h_vs = self.trans_v_mem(proj_x_v)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        return last_h_a, last_h_l, last_h_v

    def evidence(self, x):
        # return tf.exp(x)
        return F.softplus(x)

    def split(self, mu, logv, logalpha, logbeta):
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta

    def moe_nig(self, u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
        u = ((la1+2*alpha1) * u1 + u2 * (la2+2*alpha2)) / ((la1+2*alpha1) + (la2+2*alpha2))
        la = la1 + la2
        alpha = alpha1 + alpha2 + 0.5
        beta = beta1 + beta2 + 0.5 * (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
        return u, la, alpha, beta

    def forward(self, x_l, x_a, x_v):

        last_h_a, last_h_l, last_h_v = self.backbone(x_l, x_a, x_v)
        # if self.partial_mode == 3:
        #     last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj_l = self.proj_l_2(F.dropout(F.relu(self.proj_l_1(last_h_l)), p=self.out_dropout, training=self.training))
        # last_hs_proj_l += last_h_l
        last_hs_proj_a = self.proj_a_2(F.dropout(F.relu(self.proj_a_1(last_h_a)), p=self.out_dropout, training=self.training))
        # last_hs_proj_a += last_h_a
        last_hs_proj_v = self.proj_v_2(F.dropout(F.relu(self.proj_v_1(last_h_v)), p=self.out_dropout, training=self.training))
        # last_hs_proj_v += last_h_v

        output_l_mu = self.out_layer_l_mu(last_hs_proj_l)
        output_l_v = self.out_layer_l_v(last_hs_proj_l)
        output_l_alpha = self.out_layer_l_alpha(last_hs_proj_l)
        output_l_beta = self.out_layer_l_beta(last_hs_proj_l)
        output_a_mu = self.out_layer_a_mu(last_hs_proj_a)
        output_a_v = self.out_layer_a_v(last_hs_proj_a)
        output_a_alpha = self.out_layer_a_alpha(last_hs_proj_a)
        output_a_beta = self.out_layer_a_beta(last_hs_proj_a)
        output_v_mu = self.out_layer_v_mu(last_hs_proj_v)
        output_v_v = self.out_layer_v_v(last_hs_proj_v)
        output_v_alpha = self.out_layer_v_alpha(last_hs_proj_v)
        output_v_beta = self.out_layer_v_beta(last_hs_proj_v)

        mu_l, v_l, alpha_l, beta_l = self.split(output_l_mu, output_l_v, output_l_alpha, output_l_beta)
        mu_a, v_a, alpha_a, beta_a = self.split(output_a_mu, output_a_v, output_a_alpha, output_a_beta)
        mu_v, v_v, alpha_v, beta_v = self.split(output_v_mu, output_v_v, output_v_alpha, output_v_beta)

        return mu_l, v_l, alpha_l, beta_l, mu_a, v_a, alpha_a, beta_a, mu_v, v_v, alpha_v, beta_v


class GAUSSIANModel(nn.Module):
    def __init__(self, hyp_params):
        super(GAUSSIANModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 60, 60, 60
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = (self.d_l + self.d_a + self.d_v)

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=4)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=4)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=4)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        self.proj1_sigma = nn.Linear(combined_dim, combined_dim)
        self.proj2_sigma = nn.Linear(combined_dim, combined_dim)
        self.out_layer_sigma = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def backbone(self, x_l, x_a, x_v):
        """
         text, audio, and vision should have dimension [batch_size, seq_len, n_features]
         """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            h_ls = self.trans_l_mem(proj_x_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            h_as = self.trans_a_mem(proj_x_a)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            h_vs = self.trans_v_mem(proj_x_v)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        return last_h_a, last_h_l, last_h_v

    def forward(self, x_l, x_a, x_v):

        last_h_a, last_h_l, last_h_v = self.backbone(x_l, x_a, x_v)
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        # last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        output_sigma = self.out_layer_sigma(last_hs_proj)
        output_sigma = output_sigma**2
        return output, output_sigma


class NIGModel(nn.Module):
    def __init__(self, hyp_params):
        super(NIGModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        self.d_l, self.d_a, self.d_v = 60, 60, 60
        self.vonly = hyp_params.vonly
        self.aonly = hyp_params.aonly
        self.lonly = hyp_params.lonly
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = self.d_l  # assuming d_l == d_a == d_v
        else:
            combined_dim = (self.d_l + self.d_a + self.d_v)

        output_dim = hyp_params.output_dim  # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=4)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=4)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=4)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        self.out_layer_v = nn.Linear(combined_dim, output_dim)
        self.out_layer_alpha = nn.Linear(combined_dim, output_dim)
        self.out_layer_beta = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def evidence(self, x):
        # return tf.exp(x)
        return F.softplus(x)

    def split(self, mu, logv, logalpha, logbeta):
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta

    def backbone(self, x_l, x_a, x_v):
        """
         text, audio, and vision should have dimension [batch_size, seq_len, n_features]
         """
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            h_ls = self.trans_l_mem(proj_x_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Take the last output for prediction

        if self.aonly:
            h_as = self.trans_a_mem(proj_x_a)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            h_vs = self.trans_v_mem(proj_x_v)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        return last_h_a, last_h_l, last_h_v

    def forward(self, x_l, x_a, x_v):

        last_h_a, last_h_l, last_h_v = self.backbone(x_l, x_a, x_v)
        last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))

        output = self.out_layer(last_hs_proj)
        output_v = self.out_layer_v(last_hs_proj)
        output_alpha = self.out_layer_alpha(last_hs_proj)
        output_beta = self.out_layer_beta(last_hs_proj)
        output, output_v, output_alpha, output_beta = self.split(output, output_v, output_alpha, output_beta)
        return output, output_v, output_alpha, output_beta


