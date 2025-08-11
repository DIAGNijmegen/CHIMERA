import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import Attn_Net, Attn_Net_Gated, create_mlp, process_clf, process_surv

class ABMIL(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        
        # --- Model dimensions from config ---
        self.fusion_dim = getattr(config, 'fusion_dim', 0)
        self.clinical_dim = getattr(config, 'clinical_dim', 0)
        self.clinical_hidden_dim = getattr(config, 'clinical_hidden_dim', 64)
        self.clinical_layers = getattr(config, 'clinical_layers', 2)
        self.post_attention_dim = getattr(config, 'post_attention_dim', 128)  # NEW: output dim after attention

        print(f"Model initialized with fusion_dim: {self.fusion_dim}, clinical_dim: {self.clinical_dim}, post_attention_dim: {self.post_attention_dim}")

        # --- Clinical MLP branch ---
        if self.clinical_dim > 0:
            clinical_layer_dims = []
            current_dim = self.clinical_dim
            for i in range(self.clinical_layers - 1):
                next_dim = max(self.clinical_hidden_dim // (2**i), 32)
                clinical_layer_dims.append(next_dim)
                current_dim = next_dim
            self.clinical_mlp = create_mlp(
                in_dim=self.clinical_dim,
                hid_dims=clinical_layer_dims,
                dropout=config.dropout,
                out_dim=self.clinical_hidden_dim,
                end_with_fc=True
            )
            print(f"Clinical MLP created with layers: {self.clinical_dim} -> {clinical_layer_dims} -> {self.clinical_hidden_dim}")
        else:
            self.clinical_mlp = None

        # --- (Optional) MLP for WSI features before attention ---
        self.mlp = create_mlp(
            in_dim=config.in_dim,
            hid_dims=[config.embed_dim] * (config.n_fc_layers - 1),
            dropout=config.dropout,
            out_dim=config.embed_dim,
            end_with_fc=False
        )

        # --- Attention network ---
        if config.gate:
            self.attention_net = Attn_Net_Gated(
                L=config.in_dim,
                D=config.attn_dim,
                dropout=config.dropout,
                n_classes=1
            )
        else:
            self.attention_net = Attn_Net(
                L=config.in_dim,
                D=config.attn_dim,
                dropout=config.dropout,
                n_classes=1
            )
        
        # --- NEW: Linear layer to reduce attention output dimension ---
        self.post_attention_fc = nn.Linear(config.in_dim, self.post_attention_dim)

        # --- Update classifier input dimension for fusion ---
        classifier_input_dim = self.post_attention_dim + self.fusion_dim
        if self.clinical_dim > 0:
            classifier_input_dim += self.clinical_hidden_dim

        print(f"Classifier input dimension: {classifier_input_dim} (post_attention_dim: {self.post_attention_dim} + fusion_dim: {self.fusion_dim} + clinical_hidden_dim: {self.clinical_hidden_dim if self.clinical_dim > 0 else 0})")
        self.classifier = nn.Linear(classifier_input_dim, config.n_classes)
        print(f"Classifier created: input={self.classifier.in_features}, output={self.classifier.out_features}")
        self.n_classes = config.n_classes

    def forward_attention(self, h, attn_only=False):
        # h: (B, N, in_dim)
        A = self.attention_net(h)  # (B, N, K)
        A = torch.transpose(A, -2, -1)  # (B, K, N)
        if attn_only:
            return A
        else:
            return h, A

    def forward_no_loss(self, h, additional_embeddings=None, clinical_features=None, attn_mask=None):
        h, A = self.forward_attention(h)
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)
        M = torch.bmm(A, h).squeeze(dim=1)  # (B, in_dim)
        # --- Reduce dimension after attention ---
        M = self.post_attention_fc(M)  # (B, post_attention_dim)

        # --- FUSION OF FEATURES (WSI + MRI + CLINICAL) ---
        fusion_features = [M]
        if additional_embeddings is not None:
            fusion_features.append(additional_embeddings)
        elif self.fusion_dim > 0:
            zeros = torch.zeros(M.size(0), self.fusion_dim, device=M.device, dtype=M.dtype)
            fusion_features.append(zeros)
        if clinical_features is not None and self.clinical_mlp is not None:
            clinical_embedding = self.clinical_mlp(clinical_features)
            fusion_features.append(clinical_embedding)
        elif self.clinical_dim > 0 and self.clinical_mlp is not None:
            zeros = torch.zeros(M.size(0), self.clinical_hidden_dim, device=M.device, dtype=M.dtype)
            fusion_features.append(zeros)
        M = torch.cat(fusion_features, dim=-1)
        logits = self.classifier(M)
        out = {
            'logits': logits,
            'attn': A,
            'feats': h,
            'feats_agg': M
        }
        return out

    def forward(self, h, additional_embeddings=None, clinical_features=None, model_kwargs={}):
        if self.mode == 'classification':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            loss_fn = model_kwargs['loss_fn']
            out = self.forward_no_loss(h, additional_embeddings=additional_embeddings, 
                                       clinical_features=clinical_features, attn_mask=attn_mask)
            logits = out['logits']
            results_dict, log_dict = process_clf(logits, label, loss_fn)
        elif self.mode == 'survival':
            attn_mask = model_kwargs['attn_mask']
            label = model_kwargs['label']
            censorship = model_kwargs['censorship']
            loss_fn = model_kwargs['loss_fn']
            out = self.forward_no_loss(h, additional_embeddings=additional_embeddings, 
                                       clinical_features=clinical_features, attn_mask=attn_mask)
            logits = out['logits']
            results_dict, log_dict = process_surv(logits, label, censorship, loss_fn)
        else:
            raise NotImplementedError("Mode not implemented!")
        return results_dict, log_dict