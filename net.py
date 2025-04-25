import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

#===============================================================================
# Frequency Domain Components
#===============================================================================

class LFSM(nn.Module):
    """Learning Frequency Separation Module"""
    def __init__(self):
        super().__init__()
        # Initialize convolution kernel parameters
        self.rgb_filter = nn.Parameter(torch.randn(16, 1, 3, 3) * 0.01)
        self.ir_filter = nn.Parameter(torch.randn(16, 1, 3, 3) * 0.01)
        
        # Frequency adaptation network
        self.freq_adapt = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1), 
            nn.Sigmoid()
        )
        
        # Learnable threshold parameters
        self.learnable_threshold = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = nn.Parameter(torch.tensor(5.0))
        
    def get_freq_components(self, x, filter, threshold_net):
        eps = 1e-7
        original_size = x.shape[-2:]

        # Compute Fourier transform
        fft = torch.fft.rfftn(x, dim=(-2, -1))
        magnitude = torch.abs(fft) + eps
        phase = torch.angle(fft)

        # Handle dimensions
        if magnitude.dim() == 3:
            magnitude = magnitude.unsqueeze(1)
        if phase.dim() == 3:
            phase = phase.unsqueeze(1)

        # Frequency filtering and adaptive processing
        filtered = F.conv2d(magnitude, filter, padding=1)
        filtered = self.freq_adapt(filtered)

        # Generate frequency mask
        freq_mask = torch.sigmoid(
            (filtered - self.learnable_threshold) * self.scale_factor
        )
        
        # Frequency separation
        high_freq = filtered * freq_mask
        low_freq = filtered * (1 - freq_mask)
        
        return {
            'high_freq': high_freq.clamp(0, 5.0),
            'low_freq': low_freq.clamp(0, 5.0),
            'phase': phase,
            'spatial_size': original_size,
            'freq_size': magnitude.shape[-2:]
        }
        
    def forward(self, vis_img, ir_img):
        # Process visible light image
        rgb_components = self.get_freq_components(
            vis_img, self.rgb_filter, self.freq_adapt
        )
        
        # Process infrared image
        ir_components = self.get_freq_components(
            ir_img, self.ir_filter, self.freq_adapt
        )
        
        # Return processing results
        return {
            'rgb': {
                'low_freq': rgb_components['low_freq'],
                'high_freq': rgb_components['high_freq'],
                'phase': rgb_components['phase'],
                'spatial_size': rgb_components['spatial_size'],
                'freq_size': rgb_components['freq_size']
            },
            'ir': {
                'low_freq': ir_components['low_freq'],
                'high_freq': ir_components['high_freq'],
                'phase': ir_components['phase'],
                'spatial_size': ir_components['spatial_size'],
                'freq_size': ir_components['freq_size']
            }
        }

class FAEM(nn.Module):
    """Frequency Adaptive Enhancement Module"""
    def __init__(self):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.scale2 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        self.combine = nn.Conv2d(2, 1, kernel_size=1)
        
        self.lower_bound = nn.Parameter(torch.tensor(1.5))
        self.upper_bound = nn.Parameter(torch.tensor(4.0))
        
    def forward(self, high_freq):
        enhance1 = self.scale1(high_freq)
        enhance2 = self.scale2(high_freq)
        
        enhance_coeff = self.combine(torch.cat([enhance1, enhance2], dim=1))
        enhance_coeff = torch.sigmoid(enhance_coeff) 
        
        enhance_coeff = (self.upper_bound - self.lower_bound) * enhance_coeff + self.lower_bound
        
        enhanced_freq = high_freq * enhance_coeff
        return enhanced_freq

#===============================================================================
# Spatial Domain Components
#===============================================================================

class MSAA(nn.Module):
    """Multi-Scale Attention Aggregation Module"""
    def __init__(self, in_channels, reduction=8):
        super(MSAA, self).__init__()
        
        # Ensure three branches sum up to input channels
        self.mid_channels_1 = in_channels // 2  # 4 channels
        self.mid_channels_2 = in_channels - self.mid_channels_1  # 4 channels
        
        # Multi-scale convolution kernels
        self.conv_3x3 = nn.Conv2d(in_channels, self.mid_channels_1, kernel_size=3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels, self.mid_channels_2 // 2, kernel_size=5, padding=2)
        self.conv_7x7 = nn.Conv2d(in_channels, self.mid_channels_2 // 2, kernel_size=7, padding=3)

        # Channel attention mechanism
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention mechanism
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Multi-scale convolution
        feat_3x3 = self.conv_3x3(x)          # [B, 4, H, W]
        feat_5x5 = self.conv_5x5(x)          # [B, 2, H, W]
        feat_7x7 = self.conv_7x7(x)          # [B, 2, H, W]

        # Concatenate multi-scale features, maintain 8 channels
        multi_scale_features = torch.cat([feat_3x3, feat_5x5, feat_7x7], dim=1)  # [B, 8, H, W]

        # Channel attention
        channel_weights = self.channel_att(multi_scale_features)
        channel_refined = multi_scale_features * channel_weights

        # Spatial attention
        max_pool = torch.max(channel_refined, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(channel_refined, dim=1, keepdim=True)
        spatial_weights = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        output = channel_refined * spatial_weights

        return output

class ISIM(nn.Module):
    """Intensity-Structure Interaction Module"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.ir_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )
        self.vis_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )
        
        self.msaa = MSAA(out_channels)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, ir, vis):
        # Feature embedding
        ir_feat = self.ir_embed(ir)
        vis_feat = self.vis_embed(vis)
        
        # MSAA enhancement
        ir_att = self.msaa(ir_feat)
        vis_att = self.msaa(vis_feat)
        
        # Feature fusion
        fused = self.fusion(torch.cat([ir_att, vis_att], dim=1))
        
        return ir_att + fused, vis_att + fused

#===============================================================================
# Fusion Components
#===============================================================================

class AmpFuse(nn.Module):
    """Amplitude Fusion Module"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 8, kernel_size=3, padding=1), 
            nn.LeakyReLU(0.1),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)
        return self.conv1(x)

class PhaFuse(nn.Module):
    """Phase Fusion Module"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1), 
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
    def forward(self, f1, f2): 
        x = torch.cat([f1, f2], dim=1)
        return self.conv1(x)

class DDFM(nn.Module):
    """Dual Domain Fusion Module - combines frequency and spatial information"""
    def __init__(self):
        super().__init__()
        self.amp_fuse = AmpFuse()
        self.phase_fuse = PhaFuse()
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.interaction_module = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 9, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_freq_info, ir_freq_info, spatial_ir, spatial_vis):
        fused_amplitude = self.amp_fuse(
            rgb_freq_info['low_freq'] + rgb_freq_info['high_freq'],
            ir_freq_info['low_freq'] + ir_freq_info['high_freq']
        )
        
        rgb_phase = rgb_freq_info['phase']
        ir_phase = ir_freq_info['phase']
        fused_phase = self.phase_fuse(rgb_phase, ir_phase)
        
        real = fused_amplitude * torch.cos(fused_phase)
        imag = fused_amplitude * torch.sin(fused_phase)
        freq_complex = torch.complex(real.squeeze(1), imag.squeeze(1))
        
        spatial_size = rgb_freq_info['spatial_size']
        
        freq_output = torch.fft.irfftn(
            freq_complex,
            s=spatial_size,
            dim=(-2, -1)
        ).unsqueeze(1)
        
        spatial_features = self.spatial_fusion(
            torch.cat([spatial_ir, spatial_vis], dim=1)
        )
        
        combined_features = torch.cat([freq_output, spatial_features], dim=1)
        
        interaction_weights = self.interaction_module(combined_features)
        
        fused_features = combined_features * interaction_weights
        output = self.final_conv(fused_features)
        
        output = (output - output.min()) / (output.max() - output.min() + 1e-6)
        
        fused_freq_info = {
            'amp': fused_amplitude,
            'phase': fused_phase
        }
        
        return output, fused_freq_info

#===============================================================================
# Main Network
#===============================================================================

class UnifiedFusionNetwork(nn.Module):
    """Main network architecture combining all components"""
    def __init__(self):
        super().__init__()
        self.freq_separator = LFSM()
        self.freq_enhancement = FAEM()
        self.isim = ISIM(1, 8)
        self.final_fusion = DDFM()
        
    def forward(self, vis_img, ir_img):
        if len(vis_img.shape) == 3:
            vis_img = vis_img.unsqueeze(1)
        if len(ir_img.shape) == 3:
            ir_img = ir_img.unsqueeze(1)
            
        freq_components = self.freq_separator(vis_img, ir_img)
        
        freq_components['rgb']['high_freq'] = self.freq_enhancement(freq_components['rgb']['high_freq'])
        freq_components['ir']['high_freq'] = self.freq_enhancement(freq_components['ir']['high_freq'])
        
        spatial_ir, spatial_vis = self.isim(ir_img, vis_img)
        
        output, freq_info = self.final_fusion(
            freq_components['rgb'],
            freq_components['ir'],
            spatial_ir,
            spatial_vis
        )
        
        return output, freq_info



