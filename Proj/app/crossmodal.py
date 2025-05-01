import torch
import torch.nn as nn

use_cuda = torch.cuda.is_available()

class Attention(nn.Module):
    def __init__(self, d_model):
        super(Attention, self).__init__()
        self.scale = 1.0 / (d_model ** 0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        # query: [B, Tq, D], key/value: [B, Tk, D]
        scores = torch.bmm(query, key.transpose(1, 2)) * self.scale  # [B, Tq, Tk]
        if mask is not None:
            # Expect mask of shape [B, 1, Tk]
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.softmax(scores)
        context = torch.bmm(attn, value)  # [B, Tq, D]
        return context, attn

class CrossmodalNet(nn.Module):
    def __init__(self):
        super(CrossmodalNet, self).__init__()
        d_model = 64
        # Embedding layers for each modality:
        # - Entropy: input [B, 14, L_entropy]
        # - Image: input [B, 64, L_img]
        # - Byte histogram: input [B, 256]
        self.conv_e = nn.Conv1d(14, d_model, 1)
        self.conv_e_img = nn.Conv1d(64, d_model, 1)
        self.conv_e_bytes = nn.Conv1d(256, d_model, 1)

        # Downsampling layers (applied before attention)
        self.pool_e = nn.MaxPool1d(kernel_size=4, stride=4)   # For entropy: L=3600 -> 900
        self.pool_i = nn.MaxPool1d(kernel_size=4, stride=4)   # For image: L=784  -> 196

        # Attention blocks for fusion
        self.attn_e2i = Attention(d_model)  # Entropy attends to image
        self.attn_i2e = Attention(d_model)  # Image attends to entropy
        self.attn_b2i = Attention(d_model)  # Byte histogram attends to image

        # Convolutional modules after fusion (for entropy and image branches)
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
        self.conv_mod_e = nn.Sequential(
            conv_block(d_model*2, 70), conv_block(70, 70),
            conv_block(70, 70), conv_block(70, 70)
        )
        self.conv_mod_i = nn.Sequential(
            conv_block(d_model*2, 70), conv_block(70, 70),
            conv_block(70, 70), conv_block(70, 70)
        )
        # For the byte branch (which is a single token), we use Identity.
        self.conv_mod_b = nn.Identity()

        # Classifier head.
        # Expected dimensions based on using downsampled features:
        # Entropy branch: original length 3600 downsampled by pool_e -> 900
        #   After 4 conv blocks, time dimension becomes: 900 / (2^4) = 900 // 16 = 56.
        # Image branch: original length 784 downsampled by pool_i -> 196
        #   After 4 conv blocks, time dimension becomes: 196 // 16 = 12.
        # Byte branch: remains a single token.
        #
        # With conv_mod_e and conv_mod_i, outputs will have 70 channels.
        # So, flattened dimensions: entropy: 70*56 = 3920, image: 70*12 = 840, byte: 128 (from concatenation)
        # Total = 3920 + 840 + 128 = 4888.
        seq_len = 3600
        red_len = (seq_len // 4) // 16    # 3600/4 = 900; 900//16 = 56
        img_len = 784
        red_img = (img_len // 4) // 16      # 784/4 = 196; 196//16 = 12
        byte_dim = 128  # Byte branch fused feature dimension.
        flat_size = 70 * (red_len + red_img) + byte_dim  # 70*(56+12) + 128 = 70*68 + 128 = 4760 + 128 = 4888
        self.generator = nn.Sequential(
            nn.Linear(flat_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 300),
            nn.ReLU(),
            nn.Linear(300, 9)
        )
        if use_cuda:
            self.cuda()

    def forward(self, x_e, x_i, x_b, mask):
        # x_e: [B,14,L_entropy] (L_entropy = 3600)
        # x_i: [B,64,L_img]      (L_img = 784)
        # x_b: [B,256]
        B, _, L_entropy = x_e.size()
        _, _, L_img = x_i.size()

        # Compute initial embeddings
        e = self.conv_e(x_e)         # [B,64,L_entropy]
        i = self.conv_e_img(x_i)     # [B,64,L_img]
        b = self.conv_e_bytes(x_b.unsqueeze(-1))  # [B,64,1]

        # Downsample entropy and image embeddings
        e_ds = self.pool_e(e)        # [B,64,3600/4 = 900]
        i_ds = self.pool_i(i)        # [B,64,784/4 = 196]

        # Downsample the entropy mask.
        # Assume input mask shape is [B, L_entropy, 1]
        mask_ds_entropy = nn.functional.max_pool1d(mask.transpose(1,2).float(), kernel_size=4, stride=4)
        mask_ds_entropy = mask_ds_entropy.transpose(1,2).bool()  # [B,900,1]
        mask_ds_entropy = mask_ds_entropy.transpose(1,2)         # [B,1,900]

        # For image, create a mask of ones.
        img_mask = torch.ones(B, L_img, 1, device=x_i.device, dtype=torch.bool)  # [B,784,1]
        mask_ds_img = nn.functional.max_pool1d(img_mask.transpose(1,2).float(), kernel_size=4, stride=4)
        mask_ds_img = mask_ds_img.transpose(1,2).bool()  # [B,196,1]
        mask_ds_img = mask_ds_img.transpose(1,2)         # [B,1,196]

        # Prepare for attention: transpose to [B, T, D]
        qe = e_ds.transpose(1,2)  # [B,900,64]
        ki = i_ds.transpose(1,2)  # [B,196,64]

        # Attention: entropy attends to image (using image mask)
        ci_ds, _ = self.attn_e2i(qe, ki, ki, mask=mask_ds_img)
        # Attention: image attends to entropy (using entropy mask)
        ce_ds, _ = self.attn_i2e(ki, qe, qe, mask=mask_ds_entropy)

        # Byte branch:
        # Use a global image representation by averaging i_ds.
        global_img = i_ds.mean(dim=2, keepdim=True)  # [B,64,1]
        qb = b.transpose(1,2)  # [B,1,64]
        cbi, _ = self.attn_b2i(qb, global_img.transpose(1,2), global_img.transpose(1,2), mask=None)  # [B,1,64]

        # Do not upsample—use the downsampled features for fusion.
        # Fuse features:
        # For entropy branch: concatenate downsampled entropy and the attended output
        ci = ci_ds.transpose(1,2)  # [B,64,900]
        f_e = torch.cat([e_ds, ci], dim=1)  # [B,128,900]
        # For image branch: concatenate downsampled image and the attended output
        ce = ce_ds.transpose(1,2)  # [B,64,196]
        f_i = torch.cat([i_ds, ce], dim=1)  # [B,128,196]
        # For byte branch: concatenate original byte embedding and its attended output
        f_b = torch.cat([b, cbi.transpose(1,2)], dim=1)  # [B,128,1]

        # Process through convolutional modules
        o_e = self.conv_mod_e(f_e)  # Expected output: [B,70,900/16], roughly [B,70,56]
        o_i = self.conv_mod_i(f_i)  # Expected output: [B,70,196/16], roughly [B,70,12]
        o_b = self.conv_mod_b(f_b)  # Identity; stays [B,128,1]

        # Flatten outputs
        o_e = o_e.flatten(1)  # 70 * 56 = 3920 features (approx)
        o_i = o_i.flatten(1)  # 70 * 12 = 840 features (approx)
        o_b = o_b.flatten(1)  # 128 features
        o = torch.cat([o_e, o_i, o_b], dim=1)  # Total = 3920 + 840 + 128 = 4888 features

        return self.generator(o)
