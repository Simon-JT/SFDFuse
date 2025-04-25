import torch
import torch.nn as nn
import torch.nn.functional as F

#Stage-1
class HighFrequencyEnergyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = 0.01  

    def forward(self, high_freq):
        energy = torch.mean(high_freq ** 2)
        return -self.weight * torch.log(energy + 1e-8)

class FrequencyDomainContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = 0.01  
        
    def forward(self, high_freq, low_freq):
        high_energy = torch.mean(high_freq ** 2)
        low_energy = torch.mean(low_freq ** 2)
        
        energy_ratio = high_energy / (low_energy + 1e-8)

        target_ratio = torch.tensor(2.0).to(high_freq.device)
        
        return self.weight * F.smooth_l1_loss(energy_ratio, target_ratio)

class FrequencySeparationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_freq_loss = HighFrequencyEnergyLoss()
        self.contrast_loss = FrequencyDomainContrastLoss()
        self.rgb_weight = 1.0
        self.ir_weight = 1.0

    def forward(self, rgb_high_freq, rgb_low_freq, ir_high_freq, ir_low_freq):
        rgb_hf_loss = self.high_freq_loss(rgb_high_freq)
        rgb_contrast = self.contrast_loss(rgb_high_freq, rgb_low_freq)

        ir_hf_loss = self.high_freq_loss(ir_high_freq)
        ir_contrast = self.contrast_loss(ir_high_freq, ir_low_freq)
        
        total_loss = (self.rgb_weight * (rgb_hf_loss + rgb_contrast) + 
                     self.ir_weight * (ir_hf_loss + ir_contrast))
        
        return total_loss

#Stage-2
class Sobelxy(nn.Module):
    def __init__(self):
        super().__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.register_buffer("weightx", kernelx)  
        self.register_buffer("weighty", kernely)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, vis_weight=0.8):
        super().__init__()
        self.window_size = window_size
        self.vis_weight = vis_weight
        self.register_buffer("window", self.create_window(window_size))

    def create_window(self, window_size):
        def gaussian(window_size, sigma=1.5):
            gauss = torch.exp(-((torch.arange(window_size).float() - window_size // 2) ** 2) / (2 * sigma ** 2))
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window

    def calc_ssim(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=1)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=1) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, fused, visible, infrared):
        fused = torch.clamp(fused, 0, 1)
        visible = torch.clamp(visible, 0, 1)
        infrared = torch.clamp(infrared, 0, 1)
        
        ssim_vis = self.calc_ssim(fused, visible)
        ssim_ir = self.calc_ssim(fused, infrared)
        
        return self.vis_weight * ssim_vis + (1 - self.vis_weight) * ssim_ir

class MaskFusionL1Loss(nn.Module):
    def __init__(self, intensity_weight=15.0, gradient_weight=12.0, 
                 target_weight=5.0, back_weight=2.0):
        super().__init__()
        self.sobelconv = Sobelxy()
        self.intensity_weight = intensity_weight
        self.gradient_weight = gradient_weight
        self.target_weight = target_weight
        self.back_weight = back_weight

    def pixel_grad_loss(self, image_vis, image_ir, fus_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, fus_img)
        
        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        fus_img_grad = self.sobelconv(fus_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, fus_img_grad)
        
        return self.intensity_weight * loss_in + self.gradient_weight * loss_grad

    def saliency_loss(self, fus, ir, vi, mask):
        loss_tar = F.l1_loss(fus * mask, ir * mask)
        loss_back = F.l1_loss(fus * (1 - mask), vi * (1 - mask))
        return self.target_weight * loss_tar + self.back_weight * loss_back

    def forward(self, fused, ir, vi, mask):
        fused = torch.clamp(fused, 0, 1)
        ir = torch.clamp(ir, 0, 1)
        vi = torch.clamp(vi, 0, 1)
        mask = torch.clamp(mask, 0, 1)

        sal_loss = self.saliency_loss(fused, ir, vi, mask)
        grad_loss = self.pixel_grad_loss(vi, ir, fused)

        total_loss = sal_loss + grad_loss

        return total_loss, {
            'saliency_loss': sal_loss.item(),
            'gradient_loss': grad_loss.item()
        }

class DetailAwareLoss(nn.Module):
    def __init__(self, lambda_edge=1.4, lambda_ssim=0.2, lambda_mask=0.6,
                 lambda_freq=0.4, lambda_phase=0.2, mask_fusion_l1=None):
        super().__init__()
        self.mask_loss = mask_fusion_l1 if mask_fusion_l1 is not None else MaskFusionL1Loss()
        self.ssim_loss = SSIMLoss()
        
        self.lambda_edge = lambda_edge
        self.lambda_ssim = lambda_ssim
        self.lambda_mask = lambda_mask
        self.lambda_freq = lambda_freq
        self.lambda_phase = lambda_phase
        
    def forward(self, fused, ir, vi, fused_freq, ir_freq, vi_freq, mask_vi):
        fused_amp = fused_freq['amp']
        fused_phase = fused_freq['phase']
        ir_amp = ir_freq['low_freq'] + ir_freq['high_freq']
        ir_phase = ir_freq['phase']
        vi_amp = vi_freq['low_freq'] + vi_freq['high_freq']
        vi_phase = vi_freq['phase']
        
        mask_loss, loss_details = self.mask_loss(fused, ir, vi, mask_vi)
        
        ssim_loss = self.ssim_loss(fused, vi, ir)
        
        freq_loss = F.l1_loss(fused_amp, torch.maximum(ir_amp, vi_amp)) 
        phase_loss = F.l1_loss(fused_phase, (ir_phase + vi_phase) / 2)
        
        total_loss = (
            self.lambda_edge * loss_details['gradient_loss'] +
            self.lambda_mask * mask_loss +
            self.lambda_ssim * ssim_loss +
            self.lambda_freq * freq_loss +
            self.lambda_phase * phase_loss
        )
        
        loss_details.update({
            'freq_loss': freq_loss.item(),
            'phase_loss': phase_loss.item()
        })
        
        return total_loss, loss_details
