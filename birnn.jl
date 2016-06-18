using Merlin
using Compat

include("rnn.jl")

type BiRnn
  forward_rnn::Rnn
  backward_rnn::Rnn
end

function BiRnn(hidden_dim::Int)
  forward_rnn = Rnn(hidden_dim)
  backward_rnn = Rnn(hidden_dim)
  BiRnn(forward_rnn, backward_rnn)
end

@compat function (bi_rnn::BiRnn)(xs::Vector{Var})
  fs = bi_rnn.forward_rnn(xs)
  bs = bi_rnn.backward_rnn(reverse(xs))
  hs = fs + reverse(bs)  # TODO: concat or add?
  hs
end
