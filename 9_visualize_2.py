def visualize_temporal_attention(attention_weights, save_path, title='Temporal Attention'):
    """
    attention_weights: Tensor of shape (T,) or (1, T)
    """
    if attention_weights.ndim == 2:
        attention_weights = attention_weights.squeeze(0)

    plt.figure(figsize=(8, 4))
    plt.plot(attention_weights.cpu().numpy(), marker='o')
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel("Attention Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
