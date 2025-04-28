import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.scale = 1.0 / (d_model ** 0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.softmax(scores)
        context = torch.bmm(attn, value)
        return context, attn

class CrossmodalNet(nn.Module):
    def __init__(self):
        super(CrossmodalNet, self).__init__()
        d_model = 64
        self.conv_e = nn.Conv1d(14, d_model, 1)
        self.conv_e_img = nn.Conv1d(64, d_model, 1)
        self.attn = Attention(d_model)
        self.attn_img = Attention(d_model)

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )

        self.conv_module = nn.Sequential(
            conv_block(128, 70), conv_block(70, 70),
            conv_block(70, 70), conv_block(70, 70)
        )
        self.conv_module_img = nn.Sequential(
            conv_block(128, 70), conv_block(70, 70),
            conv_block(70, 70), conv_block(70, 70)
        )
        self.generator = nn.Sequential(
            nn.Linear(18900, 1000),
            nn.Linear(1000, 300),
            nn.Linear(300, 9)
        )
        if use_cuda:
            self.cuda()

    def forward(self, x, x_img, mask):
        e_x = self.conv_e(x)
        e_x_img = self.conv_e_img(x_img)

        q1 = e_x.transpose(1, 2)
        k1 = e_x_img.transpose(1, 2)
        v1 = k1
        mask_exp1 = mask.repeat(1, 1, k1.size(1))
        context1, att_score1 = self.attn(q1, k1, v1, mask=mask_exp1)
        x_attn_img = context1.transpose(1, 2)

        q2 = e_x_img.transpose(1, 2)
        k2 = e_x.transpose(1, 2)
        v2 = k2
        mask_exp2 = mask_exp1.transpose(1, 2)
        context2, att_score2 = self.attn(q2, k2, v2, mask=mask_exp2)
        x_attn = context2.transpose(1, 2)

        x_fused = torch.cat([e_x, x_attn_img], dim=1)
        x_img_fused = torch.cat([e_x_img, x_attn], dim=1)

        out_entropy = self.conv_module(x_fused)
        out_image = self.conv_module_img(x_img_fused)

        out_entropy = torch.flatten(out_entropy, 1)
        out_image = torch.flatten(out_image, 1)

        out = torch.cat([out_entropy, out_image], dim=-1)
        y = self.generator(out)
        return y, att_score1