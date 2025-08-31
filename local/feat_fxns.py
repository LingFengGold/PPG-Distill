import os
import numpy as np
import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
# from momentfm import MOMENTPipeline

from local.supp_fxns import *

# from model.ppg2ecgmetrics_net import PPG2ECGmetricsNet


class LastMinMaxPool(nn.Module):
    """Combines last element, min, and max pooling along time dimension"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # x: (B, T, D)
        last = x[:, -1, :]      # Last element pooling
        min_pool = x.min(dim=1).values  # Min pooling
        max_pool = x.max(dim=1).values  # Max pooling
        return torch.cat([last, min_pool, max_pool], dim=1)


class EnhancedUniversalPool(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_heads)
        ])
        self.final_attn = nn.Linear(2 * hidden_size, 1)  # For cross-layer attention
        
    def forward(self, layer_outputs):
        # layer_outputs: List[(B,T,D)]
        all_stats = []
        for i, layer in enumerate(layer_outputs):
            # Per-layer attentive stats using head i
            weights = F.softmax(self.heads[i](layer), dim=1)  # (B,T,1)
            mean = torch.sum(weights * layer, dim=1)  # (B,D)
            std = torch.sqrt(torch.sum(weights * (layer ** 2), dim=1) - mean ** 2 + 1e-6)
            all_stats.append(torch.cat([mean, std], dim=-1))  # (B,2D)
            
        # Cross-layer attention
        stacked = torch.stack(all_stats, dim=1)  # (B,N,2D)
        attn_weights = F.softmax(self.final_attn(stacked).squeeze(-1), dim=1)  # (B,N)
        return torch.sum(attn_weights.unsqueeze(-1) * stacked, dim=1)  # (B,2D)


class UniversalPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.layer_attn = nn.Linear(hidden_size, 1)
        
    def forward(self, layer_outputs):
        # layer_outputs: List[(B, T, D)]
#        print([_.shape for _ in layer_outputs])
        stacked = torch.stack(layer_outputs, dim=3)  # (B, T, D, L)
        layer_means = stacked.mean(dim=1)  # (B, D, L)
        
        # Permute to (B, L, D) for linear layer compatibility
        attn_input = layer_means.permute(0, 2, 1)  # (B, L, D)
        
        # Compute attention weights
        weights = F.softmax(self.layer_attn(attn_input), dim=1)  # (B, L, 1)
        
        # Aggregate layers
#        print(layer_means.shape, weights.shape)
        out = torch.einsum('bld,bl->bd', 
                          layer_means.permute(0, 2, 1),  # (B, L, D)
                          weights.squeeze(-1))  # (B, L)
#        print(out.shape)
        return out


class AttentiveStatsPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, D)
        weights = F.softmax(self.attention(x).squeeze(2), dim=1)
        mean = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        var = torch.sum(weights.unsqueeze(-1) * (x - mean.unsqueeze(1))**2, dim=1)
        return torch.cat([mean, var], dim=1)  # (B, 2D)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, D)
        weights = F.softmax(self.attention(x).squeeze(2), dim=1)  # (B, T)
        mean = torch.sum(weights.unsqueeze(-1) * x, dim=1)  # (B, D)
        return mean #.unsqueeze(1)


class EnhancedLDEPooling(nn.Module):
    """ useless
    Optimized Learnable Dictionary Encoding (LDE) pooling with:
    - Learnable temperature parameter
    - Center normalization
    - Combined mean + variance statistics
    - Efficient distance computation
    """
    def __init__(self, in_dim, num_centers=8, use_stats='mean_var'):
        super().__init__()
        self.num_centers = num_centers
        self.use_stats = use_stats
        
        # Learnable parameters with better initialization
        self.centers = nn.Parameter(torch.empty(num_centers, in_dim))
        self.scale = nn.Parameter(torch.ones(num_centers))
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Statistics type configuration
        self.stat_dim = {
            'mean': in_dim,
            'var': in_dim,
            'mean_var': 2*in_dim
        }[use_stats]

        self._init_parameters()

    def _init_parameters(self):
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        nn.init.constant_(self.scale, 1.0)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        
        # Compute distances using einsum for efficiency
        residuals = x.unsqueeze(2) - self.centers.view(1, 1, -1, D)  # (B, T, K, D)
        distances = torch.einsum('btkd,btkd->btk', residuals, residuals)  # (B, T, K)
        
        # Soft assignment with learned temperature
        assignment = F.softmax(-self.temperature * self.scale * distances, dim=-1)
        
        # Statistics computation
        weights = assignment.unsqueeze(-1)  # (B, T, K, 1)
        
        if self.use_stats in ['mean', 'mean_var']:
            mean = torch.sum(weights * residuals, dim=1)  # (B, K, D)
            
        if self.use_stats in ['var', 'mean_var']:
            var = torch.sum(weights * (residuals ** 2), dim=1) - (mean ** 2)
            
        # Combine statistics
        if self.use_stats == 'mean_var':
            stats = torch.cat([mean, var], dim=-1)
        elif self.use_stats == 'mean':
            stats = mean
        else:
            stats = var

        # Normalize and flatten
        stats = F.layer_norm(stats, [self.stat_dim])
        return stats.view(B, -1)  # (B, K*stat_dim)


class QuartilePooling(nn.Module):
    """
    Flexible quantile pooling layer with configurable quantile points
    Computes multiple quantiles along temporal dimension
    """
    def __init__(self, q_points=[0.1, 0.25, 0.5, 0.75, 0.9], keepdim=False):
        super().__init__()
        self.q_points = nn.Parameter(torch.tensor(q_points), requires_grad=False)
        self.keepdim = keepdim
        
    def forward(self, x):
        # x: (B, T, D)
        quantiles = torch.quantile(
            x, 
            self.q_points.to(x.device), 
            dim=1, 
            keepdim=self.keepdim
        )  # (num_q, B, 1, D) if keepdim else (num_q, B, D)
        
        # Rearrange dimensions
        if self.keepdim:
            quantiles = quantiles.permute(1, 2, 0, 3)  # (B, 1, num_q, D)
            return quantiles.flatten(2, 3)  # (B, 1, num_q*D)
        else:
            quantiles = quantiles.permute(1, 0, 2)  # (B, num_q, D)
            return quantiles.flatten(1, 2)  # (B, num_q*D)


class LDEPooling(nn.Module):
    """
    Learnable Dictionary Encoding (LDE) pooling layer.
    Learns dictionary centers and aggregates frame-level features
    using soft-assignment based on learned scales.
    """
    def __init__(self, in_dim, num_centers=8):
        super().__init__()
        self.num_centers = num_centers
        # Initialize dictionary centers and scale parameters
        self.centers = nn.Parameter(torch.randn(num_centers, in_dim))
        self.scale = nn.Parameter(torch.ones(num_centers))

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.size()
        # Expand centers to (1, 1, K, D)
        centers = self.centers.unsqueeze(0).unsqueeze(0)
        # Compute residuals: (B, T, K, D)
        residuals = x.unsqueeze(2) - centers
        # Squared Euclidean distances: (B, T, K)
        distances = (residuals ** 2).sum(dim=3)
        # Compute soft-assignment weights using learned scales
        scaled_distances = - self.scale.unsqueeze(0).unsqueeze(0) * distances
        assignment = F.softmax(scaled_distances, dim=2)
        # Aggregate weighted residuals over time: (B, K, D)
        aggregated = torch.sum(assignment.unsqueeze(3) * residuals, dim=1)
        # Flatten to (B, K*D)
        return aggregated.view(B, -1)


class PIIFeatureExtractor(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        normalize: bool = True,
        use_e: bool = True,
        fusion_method: str = 'concat',
        m_fc: int = 2,
        device: torch.device = torch.device('cuda'),
        model_path: str = '/scratch/skatar6/PPG_FieldStudy/results/modelv2_nocutmix.pth'
    ):
        super(PIIFeatureExtractor, self).__init__()
        self.d_model = d_model
        self.normalize = normalize
        self.use_e = use_e
        self.fusion_method = fusion_method
        self.m_fc = m_fc
        self.device = device

        # Load the trained PIINet model
        self.piinet = PIINet(
            segment_length=40,
            embedding_dim=256,
            hidden_dim=256,
            num_layers=1,
            m_fc=self.m_fc,
            num_subjects=12,
            loss_combination='learnable'
        )
        self.piinet.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=False)['model_state_dict'],
            strict=False
        )
        self.piinet.to(self.device)
        self.piinet.eval()

        # Size of shared representation
        self.feature_dim = 64 * self.m_fc

        # Linear layer to map features to d_model
        self.feature_linear = nn.Linear(self.feature_dim, self.d_model)

        # Initialize separator token
        self.sep_pii = nn.Parameter(torch.randn(self.d_model))

        # Define fusion-specific layers
        if self.fusion_method == 'concat' and self.use_e:
            self.projection = nn.Linear(2 * self.d_model, self.d_model)
        elif self.fusion_method in ['gated', 'improvedgated', 'mhgated',
                                    'attgated', 'mhattgated']:
            fusion_class = {
                'gated': GatedFusion,
            }[self.fusion_method]
            self.gated_fusion = fusion_class(self.d_model)
        elif self.fusion_method in ['add', 'mul']:
            pass
        elif self.fusion_method == 'compact_bilinear':
            self.output_dim = 1024
            self.compact_bilinear = CompactBilinearPooling(
                self.d_model, self.d_model, self.output_dim)
            self.projection = nn.Linear(self.output_dim, self.d_model)
        elif self.fusion_method == 'se':
            self.se_fusion = SEFusion(self.d_model)
        elif self.fusion_method == 'attn':
            self.attn_fusion = AttentionFusion(self.d_model)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, e: torch.Tensor = None) -> torch.Tensor:
        B, S, D = x.shape
        x = x.to(self.device)

        # Extract shared embeddings using PIINet
        shared_embeddings = self.piinet.extract_shared_embedding(x)

        # Normalize features if required
        if self.normalize:
            shared_embeddings_mean = shared_embeddings.mean(dim=1, keepdim=True)
            shared_embeddings_std = shared_embeddings.std(dim=1, keepdim=True)
            shared_embeddings = (shared_embeddings - shared_embeddings_mean) / (shared_embeddings_std + 1e-6)

        # Map features to embeddings of size d_model
        features_emb = self.feature_linear(shared_embeddings)  # Shape: (B, d_model)
        features_emb = self.activation(features_emb)

        # Include separator token
        sep = self.sep_pii.unsqueeze(0).repeat(B, 1)  # Shape: (B, d_model)

        if self.use_e:
            assert e is not None, "Embeddings 'e' must be provided when 'use_e=True'."
            seq_length = e.shape[1]

            # Repeat features_emb to match seq_length - 1
            features_emb_repeated = features_emb.unsqueeze(1).repeat(1, seq_length - 1, 1)  # Shape: (B, seq_length - 1, d_model)

            # Concatenate separator and features_emb_repeated
            sep = sep.unsqueeze(1)  # Shape: (B, 1, d_model)
            features_emb_with_sep = torch.cat([sep, features_emb_repeated], dim=1)  # Shape: (B, seq_length, d_model)

            # Now, features_emb_with_sep has shape (B, seq_length, d_model)
            features_emb_expanded = features_emb_with_sep

            # Fusion
            if self.fusion_method == 'concat':
                combined = torch.cat([e, features_emb_expanded], dim=2)  # Shape: (B, seq_length, 2*d_model)
                output = self.projection(combined)  # Shape: (B, seq_length, d_model)
            elif self.fusion_method in ['gated', 'improvedgated', 'mhgated',
                                        'attgated', 'mhattgated']:
                output = self.gated_fusion(e, features_emb_expanded)
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        else:
            # If not using 'e', output the features embedding with separator
            features_emb_expanded = torch.cat([sep.unsqueeze(1), features_emb.unsqueeze(1)], dim=1)  # Shape: (B, 2, d_model)
            output = features_emb_expanded

        output = self.activation(output)
        return output


class PIINet(nn.Module):
    def __init__(self, segment_length, embedding_dim, hidden_dim, num_layers, m_fc=1, num_subjects=1000, loss_combination='sum'):
        super(PIINet, self).__init__()
        self.segment_length = segment_length
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  # Dummy parameter for compatibility
        self.m_fc = m_fc              # Multiplier variable to control capacity of shared fc layers
        self.loss_combination = loss_combination

        # 1D Dilated CNN layers operating on raw signals
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, dilation=4, padding=4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Shared fully connected layer
        self.shared_fc = nn.Linear(hidden_dim, 64 * self.m_fc)

        # Task-specific output layers
        self.fc_age = nn.Linear(64 * self.m_fc, 1)
        self.fc_height = nn.Linear(64 * self.m_fc, 1)
        self.fc_weight = nn.Linear(64 * self.m_fc, 1)
        self.fc_skin = nn.Linear(64 * self.m_fc, 1)
        self.fc_fitness = nn.Linear(64 * self.m_fc, 1)
        # Enhanced classification head for gender
        self.fc_gender = nn.Sequential(
            nn.Linear(64 * self.m_fc, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

        # Embedding Head for Triplet Loss
        self.embedding_head = nn.Sequential(
            nn.Linear(64 * self.m_fc, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Initialize learnable weights or uncertainty parameters based on loss_combination
        if self.loss_combination == 'learnable':
            # Initialize a parameter vector for the losses (7 components)
            self.alpha = nn.Parameter(torch.ones(7))
        elif self.loss_combination == 'uncertainty':
            # Initialize uncertainty parameters
            self.log_var_age = nn.Parameter(torch.zeros(1))
            self.log_var_height = nn.Parameter(torch.zeros(1))
            self.log_var_weight = nn.Parameter(torch.zeros(1))
            self.log_var_skin = nn.Parameter(torch.zeros(1))
            self.log_var_fitness = nn.Parameter(torch.zeros(1))
            self.log_var_gender = nn.Parameter(torch.zeros(1))
            self.log_var_triplet = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x shape: [batch_size, seq_len, segment_length]
        batch_size, seq_len, segment_length = x.size()

        # Flatten seq_len and segment_length to create the input_length
        input_length = seq_len * segment_length
        x = x.view(batch_size, 1, input_length)  # Shape: [batch_size, 1, input_length]

        # Apply dilated convolutions
        x = self.relu(self.conv1(x))          # Shape: [batch_size, hidden_dim, input_length]
        x = self.relu(self.conv2(x))          # Shape: [batch_size, hidden_dim, input_length]
        x = self.relu(self.conv3(x))          # Shape: [batch_size, hidden_dim, input_length]

        x = self.dropout(x)

        # Global average pooling over the input_length dimension
        x = self.global_avg_pool(x)           # Shape: [batch_size, hidden_dim, 1]
        x = x.squeeze(2)                      # Shape: [batch_size, hidden_dim]

        # Shared fully connected layer
        shared_representation = torch.relu(self.shared_fc(x))  # Shape: [batch_size, 64 * m_fc]

        # Task-specific outputs
        age_out = self.fc_age(shared_representation).squeeze()
        height_out = self.fc_height(shared_representation).squeeze()
        weight_out = self.fc_weight(shared_representation).squeeze()
        skin_out = self.fc_skin(shared_representation).squeeze()
        fitness_out = self.fc_fitness(shared_representation).squeeze()
        gender_out = self.fc_gender(shared_representation).squeeze()

        # Embedding for Triplet Loss
        embedding = self.embedding_head(shared_representation)  # Shape: [batch_size, embedding_dim]

        # Stack regression outputs
        regression_out = torch.stack([age_out, height_out, weight_out, skin_out, fitness_out], dim=1)

        if self.loss_combination == 'learnable':
            # Apply softmax to alpha to obtain positive, normalized weights
            weights = F.softmax(self.alpha, dim=0)  # Shape: [7]
            return regression_out, gender_out, embedding, weights
        elif self.loss_combination == 'uncertainty':
            # Uncertainty-based weighted sum
            loss_age_weighted = torch.exp(-self.log_var_age) * age_out + self.log_var_age
            loss_height_weighted = torch.exp(-self.log_var_height) * height_out + self.log_var_height
            loss_weight_weighted = torch.exp(-self.log_var_weight) * weight_out + self.log_var_weight
            loss_skin_weighted = torch.exp(-self.log_var_skin) * skin_out + self.log_var_skin
            loss_fitness_weighted = torch.exp(-self.log_var_fitness) * fitness_out + self.log_var_fitness
            loss_gender_weighted = torch.exp(-self.log_var_gender) * gender_out + self.log_var_gender
            loss_triplet_weighted = torch.exp(-self.log_var_triplet) * embedding.mean() + self.log_var_triplet  # Placeholder for triplet loss weighting

            return regression_out, gender_out, embedding, \
                   loss_age_weighted, loss_height_weighted, loss_weight_weighted, \
                   loss_skin_weighted, loss_fitness_weighted, loss_gender_weighted, loss_triplet_weighted
        else:
            return regression_out, gender_out, embedding

    def extract_shared_embedding(self, x):
        """
        Extracts the shared representation from the input x without computing gradients.

        Parameters:
        - x: torch.Tensor of shape [batch_size, seq_len, segment_length]

        Returns:
        - shared_representation: torch.Tensor of shape [batch_size, 64 * m_fc]
        """
        with torch.no_grad():
            batch_size, seq_len, segment_length = x.size()

            # Flatten seq_len and segment_length to create the input_length
            input_length = seq_len * segment_length
            x = x.view(batch_size, 1, input_length)  # Shape: [batch_size, 1, input_length]

            # Apply dilated convolutions
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))

            x = self.dropout(x)

            # Global average pooling
            x = self.global_avg_pool(x).squeeze(2)  # Shape: [batch_size, hidden_dim]

            # Shared fully connected layer
            shared_representation = torch.relu(self.shared_fc(x))  # Shape: [batch_size, 64 * m_fc]

            return shared_representation

class MomentFeatureExtractor(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        normalize: bool = True,
        use_e: bool = True,
        fusion_method: str = 'concat',
        device: torch.device = torch.device('cuda'),
        model_name: str = "AutonLab/MOMENT-1-large",
    ):
        super(MomentFeatureExtractor, self).__init__()
        self.d_model = d_model
        self.normalize = normalize
        self.use_e = use_e
        self.fusion_method = fusion_method
        self.device = device
        if 'large' in model_name:
            self.d_moment = 1024
        elif 'base' in model_name:
            self.d_moment = 768

        # Load the MOMENTPipeline
        self.pipeline = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={"task_name": "embedding"},
        )
        self.pipeline.init()
        self.pipeline.to(device)
        self.pipeline.eval()

        # If d_model != 768, define feature_linear
        if self.d_model != self.d_moment:
            self.feature_linear = nn.Linear(self.d_moment, self.d_model).to(self.device)

        # Initialize separator token
        self.sep_moment = nn.Parameter(torch.randn(self.d_model).to(self.device))

        # Fusion-specific layers
        if self.fusion_method == 'concat' and self.use_e:
            self.projection = nn.Linear(2 * self.d_model, self.d_model).to(self.device)
        elif self.fusion_method in [
            'gated', 'improvedgated', 'mhgated',
            'attgated', 'mhattgated'
        ]:
            fusion_class = {
                'gated': GatedFusion,
            }[self.fusion_method]
            self.gated_fusion = fusion_class(self.d_model).to(self.device)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, e: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for MomentFeatureExtractor.

        Parameters:
        - x: torch.Tensor of shape (B, S, P)
        - e: torch.Tensor of shape (B, S+1, d_model), optional embeddings to fuse with

        Returns:
        - output: torch.Tensor, the fused embeddings
        """
        B, S, P = x.shape  # x shape: (B, S, P)

        # Reshape x to (B*S, 1, P)
        x_reshaped = x.view(B * S, 1, P).to(self.device)

        with torch.no_grad():
            # Pass x through MOMENT model
            outputs = self.pipeline(x_enc=x_reshaped)
            embeddings = outputs.embeddings  # Shape: (B*S, 768)
            embeddings = embeddings.to(self.device)

        # Reshape embeddings to (B, S, 768)
        embeddings = embeddings.view(B, S, -1)  # Now embeddings shape: (B, S, 768)

        # Map embeddings to d_model if necessary
        if self.d_model != self.d_moment:
            embeddings = self.feature_linear(embeddings)

        # Apply activation
        embeddings = self.activation(embeddings)

        # Normalize if required
        if self.normalize:
            embeddings_mean = embeddings.mean(dim=2, keepdim=True)
            embeddings_std = embeddings.std(dim=2, keepdim=True)
            embeddings = (embeddings - embeddings_mean) / (embeddings_std + 1e-6)

        # Prepend the separator token to match e's sequence length (S + 1)
        sep = self.sep_moment.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1)  # Shape: (B, 1, d_model)
        features_emb = torch.cat([sep, embeddings], dim=1)  # Shape: (B, S+1, d_model)              # c2

        if self.use_e:
            assert e is not None, "Embeddings 'e' must be provided when 'use_e=True'."
            e = e.to(self.device)
            # Now features_emb and e have the same shape: (B, S+1, d_model)

            # Fusion
            if self.fusion_method == 'concat':
                combined = torch.cat([e, features_emb], dim=2)  # Shape: (B, S+1, 2*d_model)
                output = self.projection(combined)  # Shape: (B, S+1, d_model)
            elif self.fusion_method in [
                'gated', 'improvedgated', 'mhgated',
                'attgated', 'mhattgated'
            ]:
                output = self.gated_fusion(e, features_emb)
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        else:
            output = features_emb  # Shape: (B, S+1, d_model)

        output = self.activation(output)
        return output


class MultiScaleSTFTPhaseFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_ffts=[16, 32, 64, 128],
        d_model=768,  # Ensure this matches your model's d_model
        normalize=True,
        use_e=True,
        fusion_method='gated',
        window_function='hamming',
        learnable_window=False
    ):
        super(MultiScaleSTFTPhaseFeatureExtractor, self).__init__()
        self.n_ffts = n_ffts
        self.d_model = d_model
        self.normalize = normalize
        self.use_e = use_e
        self.fusion_method = fusion_method
        self.window_function = window_function
        self.learnable_window = learnable_window

        # Initialize window functions for each n_fft
        self.windows = nn.ParameterDict()
        for n_fft in self.n_ffts:
            # Create initial window
            if self.window_function == 'hann':
                window = torch.hann_window(n_fft)
            elif self.window_function == 'hamming':
                window = torch.hamming_window(n_fft)
            elif self.window_function == 'blackman':
                window = torch.blackman_window(n_fft)
            else:
                raise ValueError(f"Unsupported window function: {self.window_function}")

            if self.learnable_window:
                # Make the window a learnable parameter
                self.windows[str(n_fft)] = nn.Parameter(window)
            else:
                # Register window as buffer (non-learnable)
                self.register_buffer(f'window_{n_fft}', window)

        # Linear layers for magnitude and phase for each scale
        self.magnitude_linears = nn.ModuleDict()
        self.phase_linears = nn.ModuleDict()
        for n_fft in self.n_ffts:
            P2 = n_fft // 2 + 1
            self.magnitude_linears[str(n_fft)] = nn.Linear(P2, self.d_model)
            self.phase_linears[str(n_fft)] = nn.Linear(P2, self.d_model)

        # Initialize separator token
        self.sep_spectral = nn.Parameter(torch.randn(1, 1, len(self.n_ffts) * self.d_model))

        # Fusion-specific layers
        if self.fusion_method == 'concat':
            if self.use_e:
                # Adjusted input dimension based on the number of scales
                self.projection = nn.Linear((len(self.n_ffts) + 1) * self.d_model, self.d_model)
        elif self.fusion_method in ['gated', 'improvedgated', 'mhgated', 'attgated', 'mhattgated', 'add', 'mul']:
            # Add projection layer to match dimensions
            if self.use_e:
                self.projection_spectral = nn.Linear(len(self.n_ffts) * self.d_model, self.d_model)
            fusion_class = {
                'gated': GatedFusion,
            }[self.fusion_method]
            self.gated_fusion = fusion_class(self.d_model)
        elif self.fusion_method == 'compact_bilinear':
            self.output_dim = 1024  # Ensure output_dim is a power of two
            self.compact_bilinear = CompactBilinearPooling(len(self.n_ffts) * self.d_model, self.d_model, self.output_dim)
            self.projection = nn.Linear(self.output_dim, self.d_model)
        elif self.fusion_method == 'se':
            self.se_fusion = SEFusion(self.d_model)
        elif self.fusion_method == 'attn':
            self.attn_fusion = AttentionFusion(self.d_model)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, e: torch.Tensor = None) -> torch.Tensor:
        B, S, P = x.shape

        if self.use_e:
            assert e is not None, "Patch embeddings 'e' must be provided when 'use_e=True'."

        spectral_embs = []
        for n_fft in self.n_ffts:
            # Adjust x to match n_fft
            if P < n_fft:
                # Zero-pad to the right
                pad_amount = n_fft - P
                x_padded = nn.functional.pad(x, (0, pad_amount))
            elif P > n_fft:
                # Truncate x to match n_fft
                x_padded = x[:, :, :n_fft]
            else:
                x_padded = x

            # Apply window function
            if self.learnable_window:
                window = self.windows[str(n_fft)].to(x.device)
            else:
                window = getattr(self, f'window_{n_fft}').to(x.device)
            x_windowed = x_padded * window  # Shape: (B, S, n_fft)

            # Compute Real FFT
            fft = torch.fft.rfft(x_windowed, n=n_fft)  # Shape: (B, S, P2), complex tensor
            magnitude = fft.abs()                      # Shape: (B, S, P2)
            phase = torch.angle(fft)                   # Shape: (B, S, P2)

            if self.normalize:
                # Normalize magnitude
                magnitude_mean = magnitude.mean(dim=-1, keepdim=True)
                magnitude_std = magnitude.std(dim=-1, keepdim=True)
                magnitude = (magnitude - magnitude_mean) / (magnitude_std + 1e-6)

                # Normalize phase to range [-1, 1]
                phase = phase / np.pi  # Since phase ranges from -π to π

            # Map magnitude and phase to embeddings
            magnitude_linear = self.magnitude_linears[str(n_fft)]
            phase_linear = self.phase_linears[str(n_fft)]
            magnitude_emb = magnitude_linear(magnitude)  # Shape: (B, S, d_model)
            phase_emb = phase_linear(phase)              # Shape: (B, S, d_model)

            # Combine magnitude and phase embeddings (element-wise addition)
            spectral_emb = magnitude_emb + phase_emb
            spectral_emb = self.activation(spectral_emb)

            spectral_embs.append(spectral_emb)  # List of tensors of shape (B, S, d_model)

        # Combine embeddings from different scales
        combined_spectral_emb = torch.cat(spectral_embs, dim=-1)  # Shape: (B, S, len(self.n_ffts) * d_model)

        # Prepend the learnable separator token
        sep = self.sep_spectral.repeat(B, 1, 1)  # Shape: (B, 1, len(self.n_ffts) * d_model)
        combined_spectral_emb = torch.cat([sep, combined_spectral_emb], dim=1)  # Shape: (B, S+1, len(self.n_ffts) * d_model)   # c1

        # Project combined_spectral_emb back to d_model if needed
        if self.use_e and self.fusion_method in ['gated', 'improvedgated', 'mhgated', 'attgated', 'mhattgated', 'add', 'mul']:
            combined_spectral_emb = self.projection_spectral(combined_spectral_emb)  # Shape: (B, S+1, d_model)

        if self.use_e:
            if self.fusion_method == 'concat':
                # Concatenate e and combined_spectral_emb along feature dimension
                combined = torch.cat([e, combined_spectral_emb], dim=-1)  # Shape: (B, S+1, total_d_model)
                # Apply linear projection to map back to d_model
                output = self.projection(combined)  # Shape: (B, S+1, d_model)
            elif self.fusion_method in ['gated', 'improvedgated', 'mhgated', 'attgated', 'mhattgated']:
                output = self.gated_fusion(e, combined_spectral_emb)  # Both inputs have shape (B, S+1, d_model)
            elif self.fusion_method == 'add':
                output = e + combined_spectral_emb  # Element-wise addition
            elif self.fusion_method == 'mul':
                output = e * combined_spectral_emb  # Element-wise multiplication
            elif self.fusion_method == 'compact_bilinear':
                cbp_output = self.compact_bilinear(combined_spectral_emb, e)  # Shape: (B, S+1, output_dim)
                output = self.projection(cbp_output)  # Map back to (B, S+1, d_model)
            elif self.fusion_method == 'se':
                output = self.se_fusion(e, combined_spectral_emb)  # Shape: (B, S+1, d_model)
            elif self.fusion_method == 'attn':
                output = self.attn_fusion(e, combined_spectral_emb)  # Shape: (B, S+1, d_model)
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
        else:
            # If use_e=False, we only have combined_spectral_emb
            output = combined_spectral_emb  # Shape: (B, S+1, len(self.n_ffts) * d_model)

        output = self.activation(output)  # Shape: (B, S+1, d_model)
        return output


class SpectralPhaseFeatureExtractor(nn.Module):
    def __init__(
        self,
        n_fft: int,
        d_model: int,
        normalize: bool = True,
        use_e: bool = True,
        fusion_method: str = 'concat',
        window_function: str = 'hamming'
    ):
        super(SpectralPhaseFeatureExtractor, self).__init__()
        self.n_fft = n_fft
        self.d_model = d_model
        self.normalize = normalize
        self.use_e = use_e
        self.fusion_method = fusion_method
        self.window_function = window_function

        # Calculate the number of frequency bins for real FFT
        self.P2 = self.n_fft // 2 + 1

        # Linear layers to map magnitude and phase features from P2 to d_model
        self.magnitude_linear = nn.Linear(self.P2, self.d_model)
        self.phase_linear = nn.Linear(self.P2, self.d_model)

        # Initialize sep_spectral with d_model dimensions
        self.sep_spectral = nn.Parameter(torch.randn(self.d_model))

        # Define window function and register it as a buffer
        if self.window_function == 'hann':
            window = torch.hann_window(self.n_fft)
        elif self.window_function == 'hamming':
            window = torch.hamming_window(self.n_fft)
        elif self.window_function == 'blackman':
            window = torch.blackman_window(self.n_fft)
        else:
            raise ValueError(f"Unsupported window function: {self.window_function}")
        self.register_buffer('window', window)

        # Define fusion-specific layers
        if self.fusion_method == 'concat':
            if self.use_e:
                self.projection = nn.Linear(3 * self.d_model, self.d_model)  # Adjusted for concatenated embeddings
        elif self.fusion_method in ['gated', 'improvedgated', 'mhgated', 'attgated', 'mhattgated']:
            fusion_class = {
                'gated': GatedFusion,
            }[self.fusion_method]
            self.gated_fusion = fusion_class(self.d_model)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, e: torch.Tensor = None) -> torch.Tensor:
        B, S, P = x.shape
        assert P == self.n_fft, f"Patch size P ({P}) must equal n_fft ({self.n_fft})."

        if self.use_e:
            assert e is not None, "Patch embeddings 'e' must be provided when 'use_e=True'."

        # Apply window function to each patch
        window = self.window.to(x.device)
        x_windowed = x * window  # Shape: (B, S, P)

        # Compute Real FFT
        fft = torch.fft.rfft(x_windowed, n=self.n_fft)  # Shape: (B, S, P2), complex tensor
        magnitude = fft.abs()                           # Shape: (B, S, P2)
        phase = torch.angle(fft)                        # Shape: (B, S, P2)

        if self.normalize:
            # Normalize magnitude
            magnitude_mean = magnitude.mean(dim=-1, keepdim=True)  # Shape: (B, S, 1)
            magnitude_std = magnitude.std(dim=-1, keepdim=True)    # Shape: (B, S, 1)
            magnitude = (magnitude - magnitude_mean) / (magnitude_std + 1e-6)

            # Normalize phase to range [-1, 1]
            phase = phase / np.pi  # Since phase ranges from -π to π

        # Map magnitude and phase to embeddings
        magnitude_emb = self.magnitude_linear(magnitude)  # Shape: (B, S, d_model)
        phase_emb = self.phase_linear(phase)              # Shape: (B, S, d_model)

        # Combine magnitude and phase embeddings
        # Option 1: Element-wise addition
        spectral_emb = magnitude_emb + phase_emb

        # Option 2: Concatenate and project back to d_model
        # spectral_emb = torch.cat([magnitude_emb, phase_emb], dim=-1)  # Shape: (B, S, 2*d_model)
        # spectral_emb = self.projection_spectral(spectral_emb)          # Requires projection layer adjusted for input size

        spectral_emb = self.activation(spectral_emb)  # Shape: (B, S, d_model)

        # Prepend the learnable separator token
        sep = self.sep_spectral.unsqueeze(0).unsqueeze(1).repeat(B, 1, 1)  # Shape: (B, 1, d_model)
        spectral_emb = torch.cat([sep, spectral_emb], dim=1)               # Shape: (B, S+1, d_model)

        if self.use_e:
            if self.fusion_method == 'concat':
                # Concatenate e and spectral_emb along feature dimension
                combined = torch.cat([e, spectral_emb], dim=2)  # Shape: (B, S+1, 2*d_model)
                # Adjust projection layer for input size
                output = self.projection(combined)              # Shape: (B, S+1, d_model)
            else:
                output = self.gated_fusion(e, spectral_emb)     # Shape: (B, S+1, d_model)
        else:
            # If use_e=False, we only have spectral_emb
            output = spectral_emb                               # Shape: (B, S+1, d_model)

        output = self.activation(output)  # Shape: (B, S+1, d_model)
        return output


class SequenceAttnFusion(nn.Module):
    """
    Cross-attention fusion:
      - e is the query sequence (B, S+1, d_model).
      - ppg_seq is the key/value sequence (B, S, d_model).
    We'll do a skip connection + LayerNorm on 'e' after multi-head attention.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, e: torch.Tensor, ppg_seq: torch.Tensor) -> torch.Tensor:
        """
        :param e:       shape (B, S+1, d_model)
        :param ppg_seq: shape (B, S, d_model)
        :return:        shape (B, S+1, d_model)
        """
        # MultiheadAttention expects:
        #    query: (B, seq_len_q, d_model)
        #    key:   (B, seq_len_k, d_model)
        #    value: (B, seq_len_k, d_model)
        # We'll let 'e' be the query, and 'ppg_seq' be the key/value.
        attn_out, _ = self.multihead_attn(query=e, key=ppg_seq, value=ppg_seq)

        # Add skip connection + layernorm
        out = self.layernorm(e + attn_out)

        return out

class PPG2ECGmetricsFuser(nn.Module):
    def __init__(
        self,
        d_model: int,
        ppg2ecg_model_path: str = 'pretrained_models/best_model_ppg2ecgmetrics.pth'
    ):
        """
        :param d_model:            dimension of the embedding in 'e' (and final fused output)
        :param ppg2ecg_model_path: path to your pretrained PPG2ECGmetricsNet
        """
        super().__init__()

        # 1) Load your PPG2ECGmetricsNet, which returns a sequence if return_sequence=True.
        self.ppg2ecgmetricsnet = PPG2ECGmetricsNet()
        # Suppose out_dim=256 if the net's each time-step embedding is 256
        self.out_dim = 256  
        
        state_dict = torch.load(ppg2ecg_model_path, map_location='cpu')
        self.ppg2ecgmetricsnet.load_state_dict(state_dict)
        self.ppg2ecgmetricsnet.eval()

        # 2) Map (B, S, out_dim) -> (B, S, d_model)
        self.proj_to_dmodel = nn.Linear(self.out_dim, d_model)

        # 3) Cross-attention fusion module
        self.attn_fusion = SequenceAttnFusion(d_model=d_model, n_heads=4, dropout=0.1)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        :param x: PPG patches, shape (B, S, P).
                  We'll pass it to ppg2ecgmetricsnet(..., return_sequence=True)
                  => (B, S, out_dim).
        :param e: Lower-level embedding, shape (B, S+1, d_model).
        :return:  Fused output, shape (B, S+1, d_model).
        """
        B, S, P = x.shape

        # 1) Reshape x to (B, 1, S*P) or whatever shape your model expects
        x_in = x.view(B, 1, S*P)
        
        # 2) ppg2ecgmetricsnet now returns a sequence: shape (B, S, out_dim)
        #    using something like: net(..., return_sequence=True)
        ppg_seq = self.ppg2ecgmetricsnet(x_in, return_sequence=True)
        # ppg_seq: (B, S, out_dim)

        # 3) Project to d_model
        ppg_seq = self.proj_to_dmodel(ppg_seq)  # (B, S, d_model)

        # 4) Cross-attention: e attends to ppg_seq
        out = self.attn_fusion(e, ppg_seq)      # (B, S+1, d_model)

        return out


#-------------------------------------------------
# 1) A small cross-attention module to fuse
#    a local sequence (e) with a single global token (f).
#-------------------------------------------------
class AttnFusion(nn.Module):
    """
    Allows each position in e to attend to the global context f.
    We'll do a skip connection + LayerNorm so the fused output
    remains in the same dimension/shape as e.
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # MultiheadAttention in PyTorch: 
        #    query, key, value shapes: (B, seq_len, d_model) if batch_first=True
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, e: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        :param e: shape (B, S+1, d_model)   -- local/low-level sequence
        :param f: shape (B, 1, d_model)     -- single global embedding
        :return:  shape (B, S+1, d_model)   -- fused output
        """
        # We let e be the "query", and f be the "key" and "value".
        # That means each position in e can attend to f.
        attn_out, _ = self.multihead_attn(query=e, key=f, value=f)
        
        # Add skip connection + LayerNorm for stability
        out = self.layernorm(e + attn_out)
        return out


#-------------------------------------------------
# 2) The actual fuser that uses the cross-attention block
#-------------------------------------------------
class PPG2ECGmetricsFuser_v2(nn.Module):
    def __init__(
        self,
        d_model: int,
        ppg2ecg_model_path: str = 'pretrained_models/best_model_ppg2ecgmetrics.pth'
    ):
        """
        :param d_model: dimension of each token in e (and final fused output)
        :param ppg2ecg_model_path: path to your pretrained PPG2ECGmetricsNet
        """
        super().__init__()

        # A) Load your ppg2ecgmetricsnet (outputs 256 or 512, etc.)
        self.out_dim = 256  # If PPG2ECGmetricsNet outputs 256
        self.ppg2ecgmetricsnet = PPG2ECGmetricsNet()
        self.ppg2ecgmetricsnet.load_state_dict(
            torch.load(ppg2ecg_model_path, map_location='cpu')
        )
        self.ppg2ecgmetricsnet.eval()

        # B) Map from the net's out_dim to d_model
        self.feature_linear = nn.Linear(self.out_dim, d_model)

        # C) Define an attention-based fusion module
        self.attn_fusion = AttnFusion(d_model)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        :param x: PPG patches, shape (B, S, P)
        :param e: External embedding, shape (B, S+1, d_model)
        :return:  Fused output, shape (B, S+1, d_model)
        """
        B, S, P = x.shape

        # 1) Reshape x to (B, 1, S*P), then extract a single global embedding
        x_in = x.view(B, 1, S * P)
        # ppg2ecgmetricsnet(...) => shape (B, self.out_dim), e.g. (B,256)
        feat = self.ppg2ecgmetricsnet(x_in, return_context=True)

        # 2) Map from out_dim to d_model
        feat_emb = self.feature_linear(feat)     # (B, d_model)

        # 3) Reshape feat_emb => (B, 1, d_model), i.e. a single global "token"
        feat_emb = feat_emb.unsqueeze(1)

        # 4) Fuse: let e attend to feat_emb
        out = self.attn_fusion(e, feat_emb)      # (B, S+1, d_model)

        return out


class PPG2ECGmetricsFuser_v1(nn.Module):
    def __init__(self, d_model, ppg2ecg_model_path='pretrained_models/best_model_ppg2ecgmetrics.pth'):
        super().__init__()

        # 1) ppg2ecgmetricsnet outputs a 512-dimensional feature
        self.out_dim = 256
        self.ppg2ecgmetricsnet = PPG2ECGmetricsNet()
        self.ppg2ecgmetricsnet.load_state_dict(
            torch.load(ppg2ecg_model_path, map_location='cpu')
        )
        self.ppg2ecgmetricsnet.eval()

        # 2) Map 512-d feature to d_model
        self.feature_linear = nn.Linear(self.out_dim, d_model)

        # 3) Learnable separator token for the first sequence index
        self.sep = nn.Parameter(torch.randn(d_model))

        # 4) Gated fusion (use your own GatedFusion implementation here)
        self.gated_fusion = GatedFusion(d_model)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        :param x: PPG patches, shape (B, S, P)
        :param e: External embedding, shape (B, S+1, d_model)
        :return:  Fused output, shape (B, S+1, d_model)
        """
        B, S, P = x.shape

        # 1) Reshape x for ppg2ecgmetricsnet: (B, 1, S*P)
        x_in = x.reshape(B, 1, S * P)

        # 2) Extract 512-d features
        feat = self.ppg2ecgmetricsnet(x_in, return_context=True)  # shape: (B, 512)

        # 3) Map features to d_model
        feat_emb = self.feature_linear(feat) # shape: (B, d_model)

        # 4) Insert a learnable separator token for the first index,
        #    then repeat the feature for the next S positions
        sep_token = self.sep.unsqueeze(0).expand(B, -1)       # (B, d_model)
        sep_token = sep_token.unsqueeze(1)                    # (B, 1, d_model)
        repeated_feats = feat_emb.unsqueeze(1).expand(-1, S, -1) # (B, S, d_model)
        feat_with_sep = torch.cat([sep_token, repeated_feats], dim=1) # (B, S+1, d_model)

        # 5) Gated fusion with e
        out = self.gated_fusion(e, feat_with_sep)  # (B, S+1, d_model)

        return out


class MultiGatedFusion(nn.Module):
    def __init__(self, d_model, num_sources=4):
        """
        d_model:    dimension of each token embedding
        num_sources: how many input tensors we want to fuse simultaneously
        """
        super().__init__()
        # You can make these layers deeper or add nonlinearities, etc.
        # The key idea is that the gating subnetwork sees the concatenated embeddings
        # and then produces element-wise gates for each source.
        self.gate_network = nn.Sequential(
            nn.Linear(num_sources * d_model, num_sources * d_model),
            nn.ReLU(),
            nn.Linear(num_sources * d_model, num_sources * d_model),
            nn.Sigmoid()
        )
        self.num_sources = num_sources
        self.d_model = d_model

    def forward(self, x_list):
        """
        x_list: a list of length N (== self.num_sources),
                each element is (B, S, d_model)
        Returns:
            fused: (B, S, d_model) fused representation
        """
        # 1) Concatenate all sources along the last dimension -> (B, S, N*d_model)
        x_cat = torch.cat(x_list, dim=-1)

        # 2) Pass through gate network -> (B, S, N*d_model)
        #    We want one learned gate per (source x embedding-dim), so shape is [B, S, N*d_model].
        gates = self.gate_network(x_cat)

        # 3) Reshape gates to split out the 'num_sources' dimension -> (B, S, N, d_model)
        gates = gates.view(x_list[0].size(0), x_list[0].size(1), self.num_sources, self.d_model)

        # 4) Stack the inputs so we can multiply elementwise with the gates -> (B, S, N, d_model)
        x_stack = torch.stack(x_list, dim=2)

        # 5) Element‐wise gating and sum across sources -> (B, S, d_model)
        fused = (gates * x_stack).sum(dim=2)
        return fused


class GatedFusion(nn.Module):   # combine two embeddings in key query style
    def __init__(self, d_model):
        super(GatedFusion, self).__init__()
        self.fc_x1 = nn.Linear(d_model, d_model)
        self.fc_x2 = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # x1 and x2: (B, S+1, d_model)
        x1_proj = self.fc_x1(x1)
        x2_proj = self.fc_x2(x2)
        combined = torch.cat([x1_proj, x2_proj], dim=2)  # (B, S+1, 2*d_model)
        gate = self.gate(combined)  # (B, S+1, d_model)
        fused = gate * x1_proj + (1 - gate) * x2_proj  # (B, S+1, d_model)
        return fused
