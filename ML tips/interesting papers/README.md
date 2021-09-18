# CLIP
https://www.casualganpapers.com/zero-shot-contrastive-loss-image-text-pretraining/CLIP-explained.html

## Notes
- `n` image - text pairs (`n` refers to a batch of images)
- `I_f = image_encoder(I)` has shape `[n, d_i]` where `n` is batch size `d_i` is output dimension of the encoder (e.g. 512, etc.)
- `T_f = text_encoder(T)` has shape `[n, d_t]` where `d_t` is output dimension of the encoder (512, 768 etc.)
- image encoder has an embedding matrix `W_i` of shape `[d_i, d_e]` where `d_e` is dimension (col) of the embedding matrix
  - output `I_e = l2_norm(np.dot(I_f, W_i), axis=1)` has shape `[n, d_e]`
- similar, text encoder has embedding matrix `W_t` of shape `[d_t, d_e]`
  - output `T_e = l2_norm(np.dot(T_f, W_t), axis=1)` has shape `[n, d_e]`

- final scaled logits = `np.dot(I_e, T_e.T)*np.exp(t)` `t` is a learned temperature param
  - output is shape `[n, n]`
  
- loss func 
  - target matrix should be an identity matrix of shape `[n, n]` (1 on diag, 0 everywhere else)
  ```
  labels = np.arrange(n)# = [0, 1, ..., n-1]
  loss_i = cross_entropy(logits, labels, axis=0) # (image is columns)
  loss_t = cross_entropy(logits, labels, axis=0) # (text is rows)
  loss = (loss_i + loss_j)/2
  ```
