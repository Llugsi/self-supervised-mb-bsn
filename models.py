# ===================================================================================================
# GITHUB REPOSITORY MODULE: PARAMETRIC FRAMEWORK MODELS
# CORE DESIGN ARCHITECTURES WITH INTEGRATED STRUCTURAL ABLATION MATRIX (IEEE TNNLS COMPLIANT)
# ===================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class TNNLS_BlindSpotNet(nn.Module):
    """
    Multi-directional Multi-Branch Blind-Spot Network aggregating spatial insights across 
    concurrent local, global, and dilated streams. Preserves true architectural blind spots.
    Includes parameter configuration to support dynamic channel-wise ablation matrices.
    """
    def __init__(self, in_channels=1, out_channels=1, ablation_mode="full"):
        super(TNNLS_BlindSpotNet, self).__init__()
        # Almacena el modo seleccionado en el submenú ('full', 'local_only', 'dilated_only', 'dense_mix')
        self.ablation_mode = str(ablation_mode).lower()
        
        # RAMA A: Convolución local estándar 3x3 (Padding=1)
        self.conv_branch_a = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        
        # RAMA B: Convolución de mediano alcance 5x5 (Padding=2)
        self.conv_branch_b = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        
        # RAMA C: Convolución dilatada 3x3 (Dilation=2, Padding=2)
        self.conv_branch_c = nn.Conv2d(in_channels, 32, kernel_size=3, padding=2, dilation=2)
        
        # RED DE FUSIÓN LOCAL 1x1
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 16, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        # STREAM A: Ultra-Local continuous context mapping (3x3 dense receptive field)
        mask_a = torch.ones_like(self.conv_branch_a.weight)
        mask_a[:, :, 1, 1] = 0.0
        weight_a = self.conv_branch_a.weight * mask_a
        h_a = F.conv2d(x, weight_a, bias=self.conv_branch_a.bias, padding=1)
        
        # STREAM B: Meso-Scale global continuous context mapping (5x5 extended receptive field)
        mask_b = torch.ones_like(self.conv_branch_b.weight)
        mask_b[:, :, 2, 2] = 0.0
        weight_b = self.conv_branch_b.weight * mask_b
        h_b = F.conv2d(x, weight_b, bias=self.conv_branch_b.bias, padding=2)
        
        # STREAM C: Macro-Scale sparse context mapping (3x3 Atrous convolution, dilation = 2)
        mask_c = torch.ones_like(self.conv_branch_c.weight)
        mask_c[:, :, 1, 1] = 0.0
        weight_c = self.conv_branch_c.weight * mask_c
        h_c = F.conv2d(x, weight_c, bias=self.conv_branch_c.bias, padding=2, dilation=2)
        
        # --- DYNAMIC STRUCTURAL FEATURE SUPPRESSION CONTROLLER ---
        # Maps precisely to manuscript Experiments I, II, III, and IV via software parameterization
        if self.ablation_mode == "local_only":
            # Experiment I: Suppress Global and Dilated feature tensors using analytical zeroing
            h_b = torch.zeros_like(h_b)
            h_c = torch.zeros_like(h_c)
        elif self.ablation_mode == "dilated_only":
            # Experiment II: Suppress Ultra-Local and Global continuous context matrices
            h_a = torch.zeros_like(h_a)
            h_b = torch.zeros_like(h_b)
        elif self.ablation_mode == "dense_mix":
            # Experiment III: Suppress exclusively the macro-scale sparse atrous stream
            h_c = torch.zeros_like(h_c)
        # If 'full', parameters remain unconstrained (Experiment IV - Proposed Method)
            
        # Manifold aggregation via channel concatenation
        multi_scale_features = torch.cat([h_a, h_b, h_c], dim=1)
        return torch.sigmoid(self.fusion(multi_scale_features))


class N2V_BaselineRegressor(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_features=32):
        super(N2V_BaselineRegressor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(hidden_features, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return torch.sigmoid(self.decoder(self.encoder(x)))
