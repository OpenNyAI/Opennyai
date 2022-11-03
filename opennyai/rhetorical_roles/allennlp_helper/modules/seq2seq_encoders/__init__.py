"""
Modules that transform a sequence of input vectors
into a sequence of output vectors.
Some are just basic wrappers around existing PyTorch modules,
others are AllenNLP modules.

The available Seq2Seq encoders are

- `"gru"` : allennlp.modules.seq2seq_encoders.GruSeq2SeqEncoder
- `"lstm"` : allennlp.modules.seq2seq_encoders.LstmSeq2SeqEncoder
- `"rnn"` : allennlp.modules.seq2seq_encoders.RnnSeq2SeqEncoder
- `"augmented_lstm"` : allennlp.modules.seq2seq_encoders.AugmentedLstmSeq2SeqEncoder
- `"alternating_lstm"` : allennlp.modules.seq2seq_encoders.StackedAlternatingLstmSeq2SeqEncoder
- `"pass_through"` : allennlp.modules.seq2seq_encoders.PassThroughEncoder
- `"feedforward"` : allennlp.modules.seq2seq_encoders.FeedForwardEncoder
- `"pytorch_transformer"` : allennlp.modules.seq2seq_encoders.PytorchTransformer
- `"compose"` : allennlp.modules.seq2seq_encoders.ComposeEncoder
- `"gated-cnn-encoder"` : allennlp.momdules.seq2seq_encoders.GatedCnnEncoder
- `"stacked_bidirectional_lstm"`: allennlp.modules.seq2seq_encoders.StackedBidirectionalLstmSeq2SeqEncoder
"""
