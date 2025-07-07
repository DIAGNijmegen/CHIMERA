import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import Attn_Net, Attn_Net_Gated, create_mlp, process_clf, process_surv


class ABMIL(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        
        # Get fusion dimension first
        self.fusion_dim = getattr(config, 'fusion_dim', 0)  # Additional embeddings dimension (e.g., 320)
        
        # Debug: Print fusion_dim to verify it's being set correctly
        print(f"Model initialized with fusion_dim: {self.fusion_dim}")
        print(f"Config fusion_dim: {config.fusion_dim if hasattr(config, 'fusion_dim') else 'Not found'}")
        
        self.mlp = create_mlp(in_dim=config.in_dim,  # Input dimension: 1024
                              hid_dims=[config.embed_dim] * (config.n_fc_layers - 1),  # Hidden dimensions
                              dropout=config.dropout,
                              out_dim=config.embed_dim,  # Output dimension: embed_dim
                              end_with_fc=False)

        if config.gate:
            self.attention_net = Attn_Net_Gated(L=config.in_dim,  # Input dimension: in_dim (since MLP is commented out)
                                                D=config.attn_dim,  # Attention dimension
                                                dropout=config.dropout,
                                                n_classes=1)  # Single attention head
        else:
            self.attention_net = Attn_Net(L=config.in_dim,  # Input dimension: in_dim (since MLP is commented out)
                                          D=config.attn_dim,  # Attention dimension
                                          dropout=config.dropout,
                                          n_classes=1)  # Single attention head

        # Classifier input dimension accounts for potential fusion
        # If fusion is used: in_dim + additional_embeddings_dim (e.g., 1024 + 320 = 1344)
        # If no fusion: in_dim (e.g., 1024)
        classifier_input_dim = config.in_dim + self.fusion_dim  # Use in_dim instead of embed_dim
        print(f"Classifier input dimension: {classifier_input_dim} (in_dim: {config.in_dim} + fusion_dim: {self.fusion_dim})")
        self.classifier = nn.Linear(classifier_input_dim, config.n_classes)  # Updated input dimension
        print(f"Classifier created: input={self.classifier.in_features}, output={self.classifier.out_features}")
        self.n_classes = config.n_classes

    def forward_attention(self, h, attn_only=False):
        # h: Tensor of shape (batch_size, num_patches, in_dim) -> (B, N, 1024)
        # h = self.mlp(h)  # Uncomment if using MLP for feature transformation
        # Since MLP is commented out, h remains: (B, N, 1024)
        A = self.attention_net(h)  # Attention scores: (batch_size, num_patches, num_heads) -> (B, N, K)
        A = torch.transpose(A, -2, -1)  # Transpose attention scores: (batch_size, num_heads, num_patches) -> (B, K, N)
        if attn_only:
            return A  # Attention scores only: (B, K, N)
        else:
            return h, A  # Features and attention scores: (B, N, in_dim), (B, K, N)

    def forward_no_loss(self, h, additional_embeddings=None, attn_mask=None):
        h, A = self.forward_attention(h)  # Features: (B, N, in_dim), Attention scores: (B, K, N)
        A_raw = A  # Raw attention scores: (B, K, N)
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min  # Masked attention scores: (B, K, N)
        A = F.softmax(A, dim=-1)  # Normalized attention scores: (B, K, N)
        M = torch.bmm(A, h).squeeze(dim=1)  # Aggregated features: (B, K, in_dim) -> (B, in_dim)
        ######
        # FUSION OF MRI AND WSI FEATURES
        ######
        if additional_embeddings is not None:
            M = torch.cat([M, additional_embeddings], dim=-1)  # Concatenate MRI and WSI features: (B, in_dim + additional_embeddings_dim)
            # (M shape: (B, in_dim + additional_embeddings_dim)) --> (B, 1024 + 320)
        else:
            # If no additional embeddings, pad with zeros to match expected classifier input dimension
            if self.fusion_dim > 0:
                zeros = torch.zeros(M.size(0), self.fusion_dim, device=M.device, dtype=M.dtype)
                M = torch.cat([M, zeros], dim=-1)  # Pad with zeros: (B, in_dim + fusion_dim)
            # If fusion_dim is 0, M remains as is: (B, in_dim)




        logits = self.classifier(M)  # Logits: (B, n_classes)

        out = {'logits': logits,  # Final predictions: (B, n_classes)
               'attn': A,  # Normalized attention scores: (B, K, N)
               'feats': h,  # Input features: (B, N, in_dim)
               'feats_agg': M}  # Aggregated features: (B, in_dim) or (B, in_dim + fusion_dim)

        return out

    def forward(self, h, additional_embeddings=None, model_kwargs={}):
        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']  # Attention mask: (B, N)
            label = model_kwargs['label']  # Labels: (B,)
            loss_fn = model_kwargs['loss_fn']  # Loss function

            out = self.forward_no_loss(h, additional_embeddings=additional_embeddings, attn_mask=attn_mask)
            logits = out['logits']  # Logits: (B, n_classes)

            results_dict, log_dict = process_clf(logits, label, loss_fn)
        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']  # Attention mask: (B, N)
            label = model_kwargs['label']  # Labels: (B,)
            censorship = model_kwargs['censorship']  # Censorship flags: (B,)
            loss_fn = model_kwargs['loss_fn']  # Loss function

            out = self.forward_no_loss(h, additional_embeddings=additional_embeddings, attn_mask=attn_mask)
            logits = out['logits']  # Logits: (B, n_classes)

            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        else:
            raise NotImplementedError("Mode not implemented!")

        return results_dict, log_dict