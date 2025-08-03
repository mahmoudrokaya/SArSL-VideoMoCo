import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class VideoMoCo(nn.Module):
    def __init__(self, base_encoder, feature_dim=256, K=4096, m=0.999, T=0.07):
        """
        Args:
            base_encoder: a 3D CNN encoder class (e.g., R3DBackbone)
            feature_dim: output feature dimension (for contrastive head)
            K: queue size (number of negative keys)
            m: momentum for updating key encoder
            T: softmax temperature
        """
        super(VideoMoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # Create encoders
        self.encoder_q = base_encoder(output_dim=feature_dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)

        # Freeze momentum encoder's gradients
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Create the queue
        self.register_buffer("queue", torch.randn(feature_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update the queue with new keys"""
        keys = concat_all_gather(keys)  # Handles multi-GPU if needed

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # Replace the keys at ptr
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: video batch for query encoder [B, 3, T, H, W]
            im_k: video batch for key encoder [B, 3, T, H, W]
        Output:
            contrastive loss
        """
        # Compute query features
        q = self.encoder_q(im_q)  # shape [B, D]
        q = nn.functional.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)

        # Compute logits: [B, 1+K]
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Concatenate logits and apply temperature
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.T

        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # Update queue
        self._dequeue_and_enqueue(k)

        return logits, labels
