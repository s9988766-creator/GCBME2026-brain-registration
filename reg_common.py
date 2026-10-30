"""
reg_common.py
---------------------------------------------------------
共用模組：空間式 3D VAE、速度場網路、微分同胚積分 (scaling & squaring)、
空間變換 (grid_sample warp)、NCC / 平滑正則損失、Jacobian 評估。

核心觀念（跟舊 code 的差別）：
  * VAE latent 保留「空間維度」  z ∈ [B, C, 16, 20, 16]，不再壓成 256 維向量。
  * 網路輸出的是「速度場」，沿時間積分（scaling & squaring）得到「形變場」。
  * 形變場用 grid_sample 施加在原始影像上 → 這才是真正的 registration。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FULL_SHAPE   = (128, 160, 128)   # 影像 (Z, Y, X)
LATENT_SHAPE = (16, 20, 16)      # 3 次 stride-2 下採樣後
LATENT_CH    = 8                 # 空間 latent 的通道數
UP           = 8                 # 128/16 = 160/20 = 128/16 = 8（各軸一致）


# =========================================================
# 空間式 3D VAE：影像 <-> 空間 latent [B, C, 16, 20, 16]
# =========================================================
class SpatialEncoder(nn.Module):
    def __init__(self, latent_ch=LATENT_CH):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, 2, 1), nn.BatchNorm3d(32), nn.LeakyReLU(0.2, True),   # 64x80x64
            nn.Conv3d(32, 64, 3, 2, 1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2, True),  # 32x40x32
            nn.Conv3d(64, 128, 3, 2, 1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2, True) # 16x20x16
        )
        self.conv_mu     = nn.Conv3d(128, latent_ch, 1)   # 保留空間維度
        self.conv_logvar = nn.Conv3d(128, latent_ch, 1)

    def forward(self, x):
        h = self.conv(x)
        return self.conv_mu(h), self.conv_logvar(h)        # [B, C, 16, 20, 16]

    def features(self, x):
        """取出多尺度特徵供 U-Net skip connection 用（不改權重，現有 VAE 照用）。
           回傳 mu 與 [f0(32ch,1/2), f1(64ch,1/4), f2(128ch,1/8)]。"""
        f0 = self.conv[0:3](x)     # Conv+BN+ReLU -> 1/2
        f1 = self.conv[3:6](f0)    # -> 1/4
        f2 = self.conv[6:9](f1)    # -> 1/8
        mu = self.conv_mu(f2)
        return mu, [f0, f1, f2]


class SpatialDecoder(nn.Module):
    def __init__(self, latent_ch=LATENT_CH):
        super().__init__()
        self.conv_in = nn.Conv3d(latent_ch, 128, 1)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 4, 2, 1), nn.BatchNorm3d(64), nn.LeakyReLU(0.2, True),  # 32x40x32
            nn.ConvTranspose3d(64, 32, 4, 2, 1), nn.BatchNorm3d(32), nn.LeakyReLU(0.2, True),   # 64x80x64
            nn.ConvTranspose3d(32, 1, 4, 2, 1),                                                 # 128x160x128
        )

    def forward(self, z):
        return self.deconv(self.conv_in(z))


class SpatialVAE(nn.Module):
    def __init__(self, latent_ch=LATENT_CH):
        super().__init__()
        self.encoder = SpatialEncoder(latent_ch)
        self.decoder = SpatialDecoder(latent_ch)

    def encode(self, x):
        return self.encoder(x)                 # mu, logvar (皆為空間 latent)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# =========================================================
# 空間變換：用位移場 flow 對 src 做 grid_sample（VoxelMorph 慣例）
# flow 單位 = voxel，順序 [B, 3(=z,y,x), D, H, W]
# =========================================================
class SpatialTransformer(nn.Module):
    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.stack(torch.meshgrid(vectors, indexing="ij"))  # [3, D, H, W]
        grid = grid.unsqueeze(0).float()
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        new_locs = self.grid + flow                    # [B, 3, D, H, W]
        shape = flow.shape[2:]
        # 正規化到 [-1, 1]
        for i in range(len(shape)):
            new_locs[:, i] = 2 * (new_locs[:, i] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)     # [B, D, H, W, 3]
        new_locs = new_locs[..., [2, 1, 0]]            # grid_sample 要 (x, y, z) 順序
        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True)


# =========================================================
# 微分同胚積分：scaling & squaring
# 把「速度場」積分成「形變場」，保證平滑、可逆 (diffeomorphic)。
# 這就是「flow」的時間積分：dφ/dt = v(φ)。
# =========================================================
class VecInt(nn.Module):
    def __init__(self, inshape, nsteps=7):
        super().__init__()
        assert nsteps >= 0
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


# =========================================================
# 速度場網路：吃 (moving latent, fixed latent) -> 速度場（latent 解析度）
# =========================================================
class VelocityNet(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            nn.Conv3d(64, 32, 3, 1, 1), nn.LeakyReLU(0.2, True),
        )
        self.flow = nn.Conv3d(32, 3, 3, 1, 1)
        # 初始化成接近 0：一開始等於「不動」，訓練才穩
        nn.init.normal_(self.flow.weight, 0, 1e-5)
        nn.init.zeros_(self.flow.bias)

    def forward(self, x):
        return self.flow(self.net(x))            # [B, 3, 16, 20, 16]


# =========================================================
# 完整對位模型：VAE(凍結) -> 速度場 -> 上採樣 -> 積分 -> warp
# =========================================================
class LatentRegNet(nn.Module):
    def __init__(self, vae, latent_ch=LATENT_CH, up=UP,
                 full_shape=FULL_SHAPE, int_steps=7):
        super().__init__()
        self.vae = vae
        for p in self.vae.parameters():          # 凍結 VAE
            p.requires_grad = False
        self.vae.eval()
        self.velnet = VelocityNet(latent_ch * 2)
        self.up = up
        self.vecint = VecInt(full_shape, nsteps=int_steps)
        self.stn = SpatialTransformer(full_shape)

    def forward(self, moving_img, fixed_img):
        with torch.no_grad():
            zm, _ = self.vae.encode(moving_img)      # [B, C, 16, 20, 16]
            zf, _ = self.vae.encode(fixed_img)
        v = self.velnet(torch.cat([zm, zf], dim=1))  # latent 解析度速度場
        # 上採樣到全解析度；位移量隨解析度放大 up 倍
        v = F.interpolate(v, scale_factor=self.up, mode="trilinear",
                          align_corners=True) * self.up
        flow = self.vecint(v)                        # 微分同胚形變場（全解析度）
        warped = self.stn(moving_img, flow)          # 施加在原始影像上
        return warped, flow, v


# =========================================================
# U-Net 版速度場解碼器：用 VAE encoder 的多尺度 skip 特徵，
# 在 1/2 解析度預測速度場（比 1/8 保留更多細節 → 更高 Dice）。
# =========================================================
class RegUNet(nn.Module):
    def __init__(self, latent_ch=LATENT_CH):
        super().__init__()
        # 1/8: cat(mu_m, mu_f, f2_m, f2_f) = C+C+128+128
        self.c3 = nn.Sequential(nn.Conv3d(latent_ch * 2 + 256, 64, 3, 1, 1),
                                nn.LeakyReLU(0.2, True))
        # 1/4: cat(up, f1_m, f1_f) = 64+64+64
        self.c2 = nn.Sequential(nn.Conv3d(64 + 128, 32, 3, 1, 1), nn.LeakyReLU(0.2, True))
        # 1/2: cat(up, f0_m, f0_f) = 32+32+32
        self.c1 = nn.Sequential(nn.Conv3d(32 + 64, 16, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.head = nn.Conv3d(16, 3, 3, 1, 1)         # 速度場 @ 1/2 解析度
        nn.init.normal_(self.head.weight, 0, 1e-5)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)

    def forward(self, mu_m, mu_f, sk_m, sk_f):
        f0m, f1m, f2m = sk_m
        f0f, f1f, f2f = sk_f
        x = torch.cat([mu_m, mu_f, f2m, f2f], dim=1)          # @1/8
        x = self.c3(x)
        x = torch.cat([self._up(x), f1m, f1f], dim=1)         # @1/4
        x = self.c2(x)
        x = torch.cat([self._up(x), f0m, f0f], dim=1)         # @1/2
        x = self.c1(x)
        return self.head(x)                                   # [B,3,1/2 解析度]


class LatentRegUNet(nn.Module):
    """VAE(凍結) + U-Net 速度場解碼器（含 skip）+ 積分 + warp。
       介面與 LatentRegNet 相同：forward(moving, fixed) -> (warped, flow, v)。"""
    def __init__(self, vae, latent_ch=LATENT_CH, full_shape=FULL_SHAPE, int_steps=7):
        super().__init__()
        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
        self.dec = RegUNet(latent_ch)
        self.vecint = VecInt(full_shape, nsteps=int_steps)
        self.stn = SpatialTransformer(full_shape)

    def forward(self, moving_img, fixed_img):
        with torch.no_grad():
            mu_m, sk_m = self.vae.encoder.features(moving_img)
            mu_f, sk_f = self.vae.encoder.features(fixed_img)
        v_half = self.dec(mu_m, mu_f, sk_m, sk_f)             # 速度場 @ 1/2
        v = F.interpolate(v_half, scale_factor=2, mode="trilinear",
                          align_corners=True) * 2             # → 全解析度，量值 ×2
        flow = self.vecint(v)
        warped = self.stn(moving_img, flow)
        return warped, flow, v


# =========================================================
# Flow-matching 版（v6）：在「形變場空間」流。
#   速度網路 v_θ(φ_t, t | z_M, z_F) 吃「當下形變場 φ_t + 時間 t + 條件」→ 速度。
#   從 φ_0(雜訊/0)沿時間積分到 φ_1。生成式:不同雜訊 → 不同 φ_1 → 不確定性。
#   靠 reg loss 訓練(繞過「需要目標」);積分用 gradient checkpointing 控記憶體。
# =========================================================
class FlowVelocityNet(nn.Module):
    """輸入:當下形變場 φ(全解析度)、時間 t、M/F 多尺度特徵。輸出:速度場(全解析度)。"""
    def __init__(self, latent_ch=LATENT_CH, up=UP):
        super().__init__()
        self.up = up
        in3 = latent_ch * 2 + 256 + 3 + 1            # mu_m,mu_f,f2_m,f2_f,φ_low(3),t(1)
        self.c3 = nn.Sequential(nn.Conv3d(in3, 64, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.c2 = nn.Sequential(nn.Conv3d(64 + 128, 32, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.c1 = nn.Sequential(nn.Conv3d(32 + 64, 16, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.head = nn.Conv3d(16, 3, 3, 1, 1)
        nn.init.normal_(self.head.weight, 0, 1e-5)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _up2(x):
        return F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)

    def forward(self, phi, t, feats_m, feats_f):
        mu_m, sk_m = feats_m
        mu_f, sk_f = feats_f
        f0m, f1m, f2m = sk_m
        f0f, f1f, f2f = sk_f
        phi_low = F.avg_pool3d(phi, kernel_size=self.up, stride=self.up)   # φ 下採樣到 1/8
        tch = torch.full_like(phi_low[:, :1], float(t))                    # t 當一個通道
        x = torch.cat([mu_m, mu_f, f2m, f2f, phi_low, tch], dim=1)
        x = self.c3(x)
        x = torch.cat([self._up2(x), f1m, f1f], dim=1); x = self.c2(x)
        x = torch.cat([self._up2(x), f0m, f0f], dim=1); x = self.c1(x)
        v_half = self.head(x)                          # 速度 @ 1/2
        return self._up2(v_half) * 2                   # 全解析度,量值 ×2


class LatentFlowReg(nn.Module):
    """VAE(凍結)+ 帶雜訊起點的形變場流 + checkpointed 積分 + warp。
       forward(moving, fixed, noise_std) -> (warped, flow, flow) 介面與其他 Reg 相容。"""
    def __init__(self, vae, latent_ch=LATENT_CH, full_shape=FULL_SHAPE, n_steps=5, use_ckpt=True):
        super().__init__()
        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
        self.vnet = FlowVelocityNet(latent_ch)
        self.stn = SpatialTransformer(full_shape)
        self.full_shape = full_shape
        self.n_steps = n_steps
        self.use_ckpt = use_ckpt

    def forward(self, moving, fixed, noise_std=0.0):
        with torch.no_grad():
            feats_m = self.vae.encoder.features(moving)
            feats_f = self.vae.encoder.features(fixed)
        B = moving.shape[0]
        if noise_std > 0:
            phi = torch.randn(B, 3, *self.full_shape, device=moving.device) * noise_std
        else:
            phi = torch.zeros(B, 3, *self.full_shape, device=moving.device)
        dt = 1.0 / self.n_steps
        for k in range(self.n_steps):
            t = k * dt
            if self.use_ckpt and self.training:
                v = checkpoint(lambda p, tt=t: self.vnet(p, tt, feats_m, feats_f),
                               phi, use_reentrant=False)
            else:
                v = self.vnet(phi, t, feats_m, feats_f)
            phi = phi + v * dt
        warped = self.stn(moving, phi)
        return warped, phi, phi


# =========================================================
# 損失
# =========================================================
class NCC:
    """局部正規化交叉相關（VoxelMorph 版），同模態對位標準損失。"""
    def __init__(self, win=9):
        self.win = win

    def loss(self, y_true, y_pred):
        Ii, Ji = y_true, y_pred
        win = [self.win] * 3
        sum_filt = torch.ones([1, 1, *win], device=Ii.device)
        pad = self.win // 2
        I2, J2, IJ = Ii * Ii, Ji * Ji, Ii * Ji
        conv = lambda a: F.conv3d(a, sum_filt, stride=1, padding=pad)
        I_sum, J_sum = conv(Ii), conv(Ji)
        I2_sum, J2_sum, IJ_sum = conv(I2), conv(J2), conv(IJ)
        wsz = float(np.prod(win))
        u_I, u_J = I_sum / wsz, J_sum / wsz
        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * wsz
        # 變異數鉗制 >=0：平坦/背景區數值誤差會算出負值 -> 分母翻負 -> loss 爆掉
        I_var = torch.clamp(I2_sum - 2 * u_I * I_sum + u_I * u_I * wsz, min=0.0)
        J_var = torch.clamp(J2_sum - 2 * u_J * J_sum + u_J * u_J * wsz, min=0.0)
        cc = cross * cross / (I_var * J_var + 1e-5)
        cc = torch.clamp(cc, max=1.0)      # 理論上界 1（Cauchy–Schwarz），防數值溢位
        return -torch.mean(cc)


class Grad:
    """位移場平滑正則。"""
    def __init__(self, penalty="l2"):
        self.penalty = penalty

    def loss(self, flow):
        dz = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dy = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dx = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])
        if self.penalty == "l2":
            dz, dy, dx = dz * dz, dy * dy, dx * dx
        return (dz.mean() + dy.mean() + dx.mean()) / 3.0


# =========================================================
# 評估：Jacobian 行列式（形變合理性）
# 回傳 (負值比例, log-Jacobian 標準差)
# =========================================================
def jacobian_stats(flow):
    """flow: numpy [3, D, H, W]（voxel 位移）。"""
    disp = flow.transpose(1, 2, 3, 0)             # [D, H, W, 3]
    grid = np.stack(np.meshgrid(
        np.arange(disp.shape[0]), np.arange(disp.shape[1]),
        np.arange(disp.shape[2]), indexing="ij"), axis=-1).astype(np.float32)
    coords = grid + disp
    gz = np.gradient(coords, axis=0)
    gy = np.gradient(coords, axis=1)
    gx = np.gradient(coords, axis=2)
    J = np.stack([gz, gy, gx], axis=-1)           # [D, H, W, 3, 3]
    det = np.linalg.det(J)
    neg_frac = float((det <= 0).mean())
    log_std = float(np.std(np.log(np.clip(det, 1e-6, None))))
    return neg_frac, log_std


# =========================================================
# 影像讀取 / 正規化（與 train_vae 一致：z-score）
# =========================================================
def load_norm(path):
    import nibabel as nib
    img = nib.load(path).get_fdata().astype(np.float32)
    m, s = img.mean(), img.std() + 1e-6
    return (img - m) / s                          # [Z, Y, X]


def load_pkl(path):
    """讀 TransMorph 版 IXI 的 .pkl，回傳 (image, seg)。
       影像已正規化 [0,1]；seg 為整數標籤（無標籤則為 None）。"""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, (tuple, list)):
        img = np.asarray(data[0], dtype=np.float32)
        seg = np.asarray(data[1]) if len(data) > 1 else None
    else:
        img, seg = np.asarray(data, dtype=np.float32), None
    return img, seg


def dice_score(seg_a, seg_b, labels=None):
    """多標籤平均 Dice（排除背景 0）。seg_a/seg_b: 整數標籤陣列。"""
    if labels is None:
        labels = np.unique(seg_a)
        labels = labels[labels != 0]
    dices = []
    for l in labels:
        a, b = (seg_a == l), (seg_b == l)
        denom = a.sum() + b.sum()
        if denom == 0:
            continue
        dices.append(2.0 * np.logical_and(a, b).sum() / denom)
    return float(np.mean(dices)) if dices else 0.0


def one_hot_seg(seg, labels):
    """seg: [B,1,D,H,W] 整數張量;labels: 標籤值 list。→ [B,K,D,H,W] one-hot float。"""
    return torch.cat([(seg == int(l)).float() for l in labels], dim=1)


def soft_dice_loss(p, g, eps=1e-5):
    """可微分 soft-Dice 損失 = 1 − soft-Dice。
       p = warp 後的軟標籤(雙線性,連續 0~1),g = atlas one-hot,皆 [B,K,D,H,W]。
       用連續值算重疊 → 可微分 → 能當訓練損失,直接優化解剖對齊。"""
    dims = (2, 3, 4)
    inter = (p * g).sum(dims)
    denom = p.sum(dims) + g.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)       # [B,K] 每個腦區各一個
    return 1.0 - dice.mean()
