using Merlin
using Compat

include("birnn.jl")

type Encoder
  word_emb::Lookup
  bi_rnns::Vector{BiRnn}   # multi-layer bi-directional GRU
end

function Encoder(word_emb::Lookup, layer_num::Int, hidden_dim::Int)
  bi_rnns = map(n -> BiRnn(hidden_dim), 1:layer_num)
  Encoder(word_emb, bi_rnns)
end

@compat function (enc::Encoder)(xs::Vector{Var})
  # initialize hidden layers
  hs = map(enc.word_emb, xs)
  last_hs = Var[]   # hidden state at last step for each layer
  for bi_rnn in enc.bi_rnns
    hs = bi_rnn(hs)
    push!(last_hs, hs[end])
  end
  last_hs, hs
end

type Decoder
  word_emb::Lookup
  attentional_linear::Linear
  softmax_linear::Linear
  grus::Vector{Graph}
  input_feeding::Bool
end

function Decoder(word_emb::Lookup, layer_num::Int, hidden_dim::Int, attentional_dim::Int, out_vocab_size::Int, input_feeding::Bool)
  T = Float32
  grus = map(n -> GRU(T, hidden_dim), 1:layer_num)
  attentional_linear = Linear(T, hidden_dim * 2, attentional_dim)
  softmax_linear = Linear(T, attentional_dim, out_vocab_size)
  Decoder(word_emb, attentional_linear, softmax_linear, grus, input_feeding)
end

attention_model(e::Var, d::Var) = sum(e .* d, 1)    # inner-product

@compat function (dec::Decoder)(xs::Vector{Var}, h0s::Vector{Var}, encoder_states::Vector{Var})
  batch_size = size(h0s[1].value, 2)
  eos = Var(fill(EOS_ID, 1, batch_size))
  hs = h0s    # feed encoder states at last time step
  attentional = Var(zeros(Float32, 1, batch_size))
  ys = Var[]
  as = Var[]
  for x in [eos; xs]
    hs, a, attentional, y = step(dec, x, hs, encoder_states, attentional)
    push!(ys, y)
    push!(as, a)
  end
  ys, as
end

function generate(dec::Decoder, h0s::Vector{Var}, encoder_states::Vector{Var}, max_len::Int)
  batch_size = size(h0s[1].value, 2)
  hs = h0s    # feed encoder states at last time step
  attentional = Var(zeros(Float32, 1, batch_size))
  ids = Vector{Int}[]
  as = Var[]
  done = falses(batch_size)
  x = Var(fill(EOS_ID, 1, batch_size))
  for i in 1:max_len
    hs, a, attentional, y = step(dec, x, hs, encoder_states, attentional)
    # select most likely ID
    max_id = argmax(y.value, 1)   # TODO: modify this to allow for beam search
    x = Var(reshape(max_id, 1, batch_size))
    done |= max_id .== EOS_ID
    push!(ids, max_id)
    push!(as, a)
    if all(done)
      # all samples have reached <EOS>
      break
    end
  end
  ids, as
end

function step(dec::Decoder, x::Var, hs::Vector{Var}, encoder_states::Vector{Var}, attentional::Var)
  input_length = length(encoder_states)
  hidden_dim, batch_size = size(encoder_states[1].value)

  h_in = dec.word_emb(x)  # look up word embedding
  if dec.input_feeding
    # input-feeding
    h_in += attentional
  end
  next_hs = Var[]
  for l in 1:length(hs)
    h = hs[l]
    h_in = dec.grus[l](:x=>h_in, :h=>h)
    push!(next_hs, h_in)
  end

  # calculate attention weights
  d = next_hs[end]   # top layer (dec_hidden_size x batch_size)
  # TODO: is it more efficient to use matrix operations? (batch_matmul)
  unnormalized_weights = concat(1, map(e -> attention_model(e, d), encoder_states)...)   # (input_length x batch_size)
  a = softmax(unnormalized_weights)

  # compose context vector
  a_3d = reshape(a, 1, input_length, batch_size)  # (1 x input_length x batch_size)
  encoder_states_3d = concat(2, map(e -> reshape(e, hidden_dim, 1, batch_size), encoder_states)...)  # (hidden_dim x input_length x batch_size)
  context = reshape(sum(a_3d .* encoder_states_3d, 2), hidden_dim, batch_size)  # (hidden_dim x batch_size)

  # create attentional vector
  attentional = relu(dec.attentional_linear(concat(1, d, context)))   # (attentional_dim x batch_size)

  # predict next word
  #y = softmax(dec.softmax_linear(attentional))   # (out_vocab_size x batch_size)
  y = dec.softmax_linear(attentional)   # (out_vocab_size x batch_size)

  next_hs, a, attentional, y
end

type AttentionalEncoderDecoder
  encoder::Encoder
  decoder::Decoder
end

function AttentionalEncoderDecoder(vocab_size::Int, layer_num::Int, hidden_dim::Int, input_feeding::Bool)
  T = Float32
  word_emb = Lookup(T, vocab_size, hidden_dim)
  encoder = Encoder(word_emb, layer_num, hidden_dim)
  decoder = Decoder(word_emb, layer_num, hidden_dim, hidden_dim, vocab_size, input_feeding)
  AttentionalEncoderDecoder(encoder, decoder)
end

@compat function (encdec::AttentionalEncoderDecoder)(xs::Vector{Var}, ts::Vector{Var})
  last_hs, hs = encdec.encoder(xs)
  ys, as = encdec.decoder(ts, last_hs, hs)
  ys, as
end

function generate(encdec::AttentionalEncoderDecoder, xs::Vector{Var}, max_len::Int=50)
  last_hs, hs = encdec.encoder(xs)
  ys, as = generate(encdec.decoder, last_hs, hs, max_len)
  ys, as
end
