using Merlin
using Compat

type Rnn
  gru::Graph
  hidden_dim::Int
end

function Rnn(hidden_dim::Int)
  T = Float32
  gru = GRU(T, hidden_dim)
  Rnn(gru, hidden_dim)
end

@compat function (rnn::Rnn)(xs::Vector{Var})
  # initialize hidden layer
  T = Float32
  batch_size = size(xs[1].value, 2)
  h = Var(zeros(T, rnn.hidden_dim, batch_size))  # TODO: is this initialization standard?

  # process each input
  hs = Var[]
  for x in xs
    h = rnn.gru(:x => x, :h => h)
    push!(hs, h)
  end
  hs
end
