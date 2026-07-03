import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSReweightingPooler(nn.Module):
    """
    Combines the CLS token with attention-pooled tokens.
    Output: a single pooled vector per sequence.
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Attention for token-level importance
        self.attention = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)

        # Learnable mixing of CLS and attention-pooled vector
        self.mix = nn.Linear(hidden_size * 2, hidden_size)

        # Optional nonlinearity
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None, return_scores=False):
        """
        hidden_states: [B, T, H]
        mask (optional): [B, T] (1 = keep token, 0 = ignore)
        """

        # ---- 1. CLS embedding ----
        cls = hidden_states[:, 0]  # [B, H]

        # ---- 2. Attention scores for each token ----
        scores = self.attention(hidden_states).squeeze(-1)  # [B, T]
        
        # Mask out CLS token
        scores[:, 0] = -1e4

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e4)

        weights = torch.softmax(scores, dim=-1)  # [B, T]

        # ---- 3. Attention-based pooled vector ----
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # [B, H]

        # # ---- 4. Concatenate CLS + attention-pooled ----
        combined = torch.cat([cls, pooled], dim=-1)  # [B, 2H]        
        combined = self.dropout(combined) 

        # ---- 5. Learnable mixing ----
        pooled = self.activation(self.mix(combined))  # [B, H]
        
        #pooled = cls + attn_pooled  # [B, H]

        if return_scores:
            return pooled, weights  # return per-token weights
        return pooled   


def mean_pooling(token_embeddings, attention_mask):
    # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    # Sum of the attention mask
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9).unsqueeze(1)
    # Mean Pooling
    return sum_embeddings / sum_mask
    
class MeanReweightingPooler(nn.Module):
    """
    Combines the Mean token with attention-pooled tokens.
    Output: a single pooled vector per sequence.
    """

    def __init__(self, hidden_size):
        super().__init__()

        # Attention for token-level importance
        self.attention = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.1)

        # Learnable mixing of CLS and attention-pooled vector
        self.mix = nn.Linear(hidden_size * 2, hidden_size)

        # Optional nonlinearity
        self.activation = nn.Tanh()

    def forward(self, hidden_states, mask=None, return_scores=False):
        """
        hidden_states: [B, T, H]
        mask (optional): [B, T] (1 = keep token, 0 = ignore)
        """

        # ---- 1. CLS embedding ----
        cls = mean_pooling(hidden_states, mask)  # [B, H]        

        # ---- 2. Attention scores for each token ----
        scores = self.attention(hidden_states).squeeze(-1)  # [B, T]
        
        # Mask out CLS token
        scores[:, 0] = -1e4

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e4)

        weights = torch.softmax(scores, dim=-1)  # [B, T]

        # ---- 3. Attention-based pooled vector ----
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)  # [B, H]

        # # ---- 4. Concatenate CLS + attention-pooled ----
        combined = torch.cat([cls, pooled], dim=-1)  # [B, 2H]        
        combined = self.dropout(combined) 

        # ---- 5. Learnable mixing ----
        pooled = self.activation(self.mix(combined))  # [B, H]
        
        #pooled = cls + attn_pooled  # [B, H]

        if return_scores:
            return pooled, weights  # return per-token weights
        return pooled   


class TextGatedAttentionPooler_1(nn.Module):
    def __init__(self, hidden_dim=768, output_dim=1024, gated_weight=0.5):
        super().__init__()
        self.gated_weight = gated_weight  # Controls landmark injection intensity
        
        # Gating network for raw token sequences
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Projection heads to map elements to your 1024 footprint
        self.base_projection = nn.Linear(hidden_dim, output_dim)
        self.gated_projection = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, last_hidden_states, attention_mask, original_pooler_output):
        """
        Args:
            last_hidden_states: [B, Seq_Len, 768] (Raw token states)
            attention_mask: [B, Seq_Len]
            original_pooler_output: [B, 768] (CLIP's native EOS token output)
        """
        # 1. Project the foundational, pre-aligned CLIP representation
        base_clip_emb = self.base_projection(original_pooler_output) # Preserves Obj 1
        
        # 2. Extract your clean landmark features via Gated Attention
        gate_scores = torch.sigmoid(self.gate_net(last_hidden_states))
        transformed_features = self.feature_net(last_hidden_states)
        
        mask_expanded = attention_mask.unsqueeze(-1)
        gated_features = transformed_features * gate_scores * mask_expanded
        active_gates = gate_scores * mask_expanded
        
        gated_context = gated_features.sum(dim=1) / (active_gates.sum(dim=1) + 1e-5)
        transformed_gated = self.gated_projection(gated_context) # Captures Obj 2
        
        # 3. Residual Blending: Base Bridge + Gated Landmark Modifier
        final_embeddings = base_clip_emb + self.gated_weight * transformed_gated
        
        return self.layer_norm(final_embeddings)
    

class TextGatedAttentionPooler(nn.Module):
    """
    Hybrid Order-Aware Gated Attention Pooler for Cross-Modal VPR.
    Combines CLIP's native global semantic anchor (CLS/EOS) with local 
    1D-Convolutional spatial layouts to preserve left-to-right modifiers.
    """
    def __init__(self, hidden_dim=768, output_dim=512, kernel_size=3, gated_weight=0.4):
        super().__init__()
        self.gated_weight = gated_weight # Controls directional spatial injection intensity
        
        # 1. Global Space Alignment Branch (Objective 1)
        self.base_projection = nn.Linear(hidden_dim, output_dim)
        
        # 2. Local Attribute Binding Branch (Objective 2)
        self.local_context_net = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.gated_projection = nn.Linear(hidden_dim, output_dim)
        
        # Final normalization layer
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, last_hidden_states, attention_mask, original_pooler_output):
        """
        Args:
            last_hidden_states (Tensor): Raw token states from CLIP text tower [Batch, 77, 768]
            attention_mask (Tensor): Binary token mask [Batch, 77]
            original_pooler_output (Tensor): CLIP's native sentence embedding [Batch, 768]
        """
        # =====================================================================
        # BRANCH A: NATIVE GLOBAL SEMANTIC ANCHOR
        # =====================================================================
        # Project the pre-aligned foundational CLIP representation
        base_clip_emb = self.base_projection(original_pooler_output)
        
        # =====================================================================
        # BRANCH B: ORDER-AWARE LOCAL LANDMARK EXTRACTION
        # =====================================================================
        mask_expanded = attention_mask.unsqueeze(-1)
        masked_states = last_hidden_states * mask_expanded
        
        # Permute to [Batch, 768, 77] for Conv1D phrase grouping
        x = masked_states.permute(0, 2, 1)
        x_context = F.gelu(self.local_context_net(x))
        
        # Permute back to sequence format [Batch, 77, 768]
        context_states = x_context.permute(0, 2, 1) * mask_expanded
        
        # Compute independent Sigmoid gate scores across contextualized words
        gate_scores = torch.sigmoid(self.gate_net(context_states))
        transformed_features = self.feature_net(context_states)
        
        gated_features = transformed_features * gate_scores * mask_expanded
        active_gates = gate_scores * mask_expanded
        
        # Normalized Gated Pooling (Weighted Average over sequence landmarks)
        gated_context = gated_features.sum(dim=1) / (active_gates.sum(dim=1) + 1e-5)
        transformed_gated = self.gated_projection(gated_context)
        
        # =====================================================================
        # COMBINATION: GLOBAL BASELINE + LOCAL SEQUENTIAL RESIDUAL
        # =====================================================================
        final_embeddings = base_clip_emb + self.gated_weight * transformed_gated
        
        return self.layer_norm(final_embeddings)
    

class GeMPooling1D(nn.Module):
    """
    Generalized Mean Pooling for ViT patch tokens.
    Expects input shape: [Batch, Num_Patches, Vision_Dim] (e.g., [B, 196, 768])
    Outputs shape: [Batch, Vision_Dim]
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        # Make p trainable so the network can find the optimal softness between AVG and MAX
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: [B, 196, 768]
        # Permute to [B, 768, 196] to pool across the patch dimension
        x = x.permute(0, 2, 1)
        
        # Apply the GeM equation safely to prevent negative bases or NaNs
        pooled = torch.mean(x.clamp(min=self.eps).pow(self.p), dim=2)
        pooled = pooled.pow(1.0 / self.p)
        
        return pooled # Returns [B, 768]
    
    
class AttentionGatedPatchPooler(nn.Module):
    """
    Learns a spatial scalar attention mask over the 196 patches.
    Transforms [B, 196, 768] -> [B, 768]
    """
    def __init__(self, vision_dim=768):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 4),
            nn.Tanh(),
            nn.Linear(vision_dim // 4, 1)
        )

    def forward(self, x):
        # x shape: [B, 196, 768]
        attn_logits = self.attention_net(x) # [B, 196, 1]
        attn_weights = F.softmax(attn_logits, dim=1) # Normalize across patches
        
        # Weighted sum across the patch dimension
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled # Returns [B, 768]
    
    
class SpatialLayoutPooler(nn.Module):
    """
    Preserves the 14x14 grid structure, projects channels down, 
    and flattens the spatial layout directly.
    """
    def __init__(self, vision_dim=768, grid_size=14, bottleneck_dim=768):
        super().__init__()
        self.grid_size = grid_size
        
        # Compress channels first to avoid a massive linear layer later
        self.channel_compress = nn.Sequential(
            nn.Conv2d(vision_dim, 64, kernel_size=1),
            nn.ReLU()
        )
        # Final projection to map back to your required vision output dimension
        self.final_projection = nn.Linear(64 * grid_size * grid_size, bottleneck_dim)

    def forward(self, x):
        B, N, C = x.shape # [B, 196, 768]
        
        # Reshape to 2D Image Space grid: [B, 768, 14, 14]
        x = x.permute(0, 2, 1).reshape(B, C, self.grid_size, self.grid_size)
        
        x = self.channel_compress(x) # [B, 64, 14, 14]
        x = x.reshape(B, -1) # Flatten spatial + channel layout: [B, 64 * 14 * 14]
        
        pooled = self.final_projection(x) # [B, bottleneck_dim] (e.g., 512 or 768)
        return pooled
    


class MultiLayerAttentionTextPooler(nn.Module):
    def __init__(
        self,
        text_dim=512,
        joint_dim=512,
        target_layers=(9, 10, 11, 12),
        num_heads=8,
        init_eot_bias=2.0,  # sigmoid(2)=0.88 => mostly EOT at start
    ):
        super().__init__()

        self.target_layers = target_layers
        num_layers = len(target_layers)

        #
        # Multi-layer fusion
        #
        self.layer_fusion = nn.Sequential(
            nn.Linear(text_dim * num_layers, text_dim),
            nn.GELU(),
        )

        #
        # Learnable query
        #
        self.pool_query = nn.Parameter(
            torch.randn(1, 1, joint_dim) * 0.02
        )

        #
        # QKV projections
        #
        self.k_proj = nn.Linear(text_dim, joint_dim)
        self.v_proj = nn.Linear(text_dim, joint_dim)

        #
        # MHA Pooling
        #
        self.mha_pool = nn.MultiheadAttention(
            embed_dim=joint_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        #
        # Residual normalization
        #
        self.norm = nn.LayerNorm(joint_dim)

        #
        # EOT vs Attention gate
        #
        self.alpha = nn.Parameter(
            torch.tensor(init_eot_bias)
        )

        #
        # Final projection
        #
        self.output_proj = nn.Sequential(
            nn.Linear(joint_dim, joint_dim),
            nn.LayerNorm(joint_dim),
        )

    def forward(self, text_all_layers, text_embeds, attention_mask, native_text_projection=None):
        B = attention_mask.size(0)

        # ==========================================================
        # 1. FIX: EXTRACT TRUE NATIVE CLIP EOT VECTOR
        # ==========================================================
        # In CLIP, the true global feature is at the EOT index of the LAST layer (12)
        
        eot_idx = (attention_mask.long().sum(dim=1) - 1)

        # ==========================================================
        # 2. HIERARCHICAL ATTENTION POOLING (MHA)
        # ==========================================================
        extracted_states = [text_all_layers[l] for l in self.target_layers]
        hierarchical_features = torch.cat(extracted_states, dim=-1)
        fused_text_tokens = self.layer_fusion(hierarchical_features) # [B, T, D]

        q = self.pool_query.expand(B, -1, -1) # [B, 1, D]
        k = self.k_proj(fused_text_tokens)
        v = self.v_proj(fused_text_tokens)

        key_padding_mask = (attention_mask == 0)
        pooled_features, _ = self.mha_pool(
            query=q, key=k, value=v, 
            key_padding_mask=key_padding_mask, 
            need_weights=False
        )
        pooled_features = pooled_features.squeeze(1)
        pooled_features = self.norm(pooled_features + q.squeeze(1))

        # ==========================================================
        # 3. FUSION GATE (True Global vs Learned Attention Local)
        # ==========================================================
        gate = torch.sigmoid(self.alpha)

        # Mix the TRUE pre-aligned global token with your custom attention pooler
        output = (
            gate * text_embeds 
            + (1.0 - gate) * self.output_proj(pooled_features)
        )

        # Normalization for Cosine Loss matching
        return F.normalize(output, p=2, dim=-1)
    
class ResidualTextPooler(nn.Module):
    def __init__(self, text_dim=512, joint_dim=512, target_layers=(9, 10, 11)):
        super().__init__()
        self.target_layers = target_layers
        num_layers = len(target_layers)
        
        # Fuse intermediate layers only (excluding the final layer to prevent duplication)
        self.layer_fusion = nn.Sequential(
            nn.Linear(text_dim * num_layers, text_dim),
            nn.GELU()
        )
        
        # Custom attention pooling head
        self.pool_query = nn.Parameter(torch.randn(1, 1, joint_dim) * 0.01)
        self.k_proj = nn.Linear(text_dim, joint_dim)
        self.v_proj = nn.Linear(text_dim, joint_dim)
        
        self.mha_pool = nn.MultiheadAttention(embed_dim=joint_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(joint_dim)
        
        # Scale block to shrink initialization impact close to zero
        self.adapter_scale = nn.Parameter(torch.tensor(0.001)) 
        self.output_proj = nn.Linear(joint_dim, joint_dim)

    def forward(self, text_all_layers, text_embeds, attention_mask):
        """
        Args:
            text_all_layers: text_output.hidden_states
            attention_mask: padding mask tensor
            native_clip_output: The exact pre-trained text embedding from your working script
        """
        B = attention_mask.size(0)
        
        # 1. Gather intermediate layers cleanly
        extracted_states = [text_all_layers[l] for l in self.target_layers]
        hierarchical_features = torch.cat(extracted_states, dim=-1)
        fused_text_tokens = self.layer_fusion(hierarchical_features)
        
        # 2. Run your Multi-Head Attention pooling routine
        q = self.pool_query.expand(B, -1, -1)
        k = self.k_proj(fused_text_tokens)
        v = self.v_proj(fused_text_tokens)
        
        pooled_features, _ = self.mha_pool(
            query=q, key=k, value=v, 
            key_padding_mask=(attention_mask == 0)
        )
        pooled_features = self.norm(pooled_features.squeeze(1) + q.squeeze(1))
        delta_text_features = self.output_proj(pooled_features)
        
        # 3. Apply the Residual Connection
        # This guarantees your model starts with your working baseline accuracy at step 0
        final_output = text_embeds + (self.adapter_scale * delta_text_features)
        
        return F.normalize(final_output, p=2, dim=-1)