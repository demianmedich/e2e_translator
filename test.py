import numpy as np
import torch

from module import GruEncoder, GruDecoder

batch_size = 16
max_seq_len = 100
tgt_max_seq_len = 50
vocab_size = 300
embedding_dim = 100
encoder_bidirectional = True
encoder_hidden_size = 50
decoder_hidden_size = encoder_hidden_size

encoder = GruEncoder(vocab_size, embedding_dim, encoder_hidden_size,
                     bidirectional=encoder_bidirectional)

src_seqs = np.random.randint(vocab_size, size=(batch_size, max_seq_len))
src_seqs = torch.tensor(src_seqs, dtype=torch.long)
src_seq_lengths = np.ones(batch_size) * max_seq_len
src_seq_lengths = torch.tensor(src_seq_lengths, dtype=torch.int)

tgt_seqs = np.random.randint(vocab_size, size=(batch_size, tgt_max_seq_len))
tgt_seqs = torch.tensor(tgt_seqs, dtype=torch.long)
tgt_lengths = np.ones(batch_size) * tgt_max_seq_len
tgt_lengths = torch.tensor(tgt_lengths, dtype=torch.int)

print(f'src_seqs shape: {src_seqs.size()}, src_seq_lengths shape: {src_seq_lengths.size()}')
print(f'tgt_seqs shape: {tgt_seqs.size()}, tgt_lengths shape: {tgt_lengths.size()}')

output, hidden_state = encoder(src_seqs, src_seq_lengths)

decoder = GruDecoder(vocab_size, embedding_dim, decoder_hidden_size)
decoder.train()
decoder(output, hidden_state, tgt_seqs, tgt_lengths)
