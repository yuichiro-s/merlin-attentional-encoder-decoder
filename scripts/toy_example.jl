using Merlin
using Logging
# using ProfileView

# load data
include("../vocabulary.jl")
include("../attention.jl")
include("../train.jl")

# create model
vocab_size = 5
layer_num = 1
hidden_dim = 10
input_feeding = false
encdec = AttentionalEncoderDecoder(vocab_size, layer_num, hidden_dim, input_feeding)

model_dir = "model"
mkpath(model_dir)

# create batches
function create_batches(batch_size, batch_num, max_len, vocab_size)
  batches = []
  for i in 1:batch_num
    l_x = rand(6:max_len)
    # l_x = max_len
    xs = rand(2:vocab_size, batch_size, l_x)
    ts = xs[:, end:-1:1]
    push!(batches, (xs, ts))
  end
  batches
end
max_len = 10
train_batch_size = 10
train_batch_num = 1000
train_batches = create_batches(train_batch_size, train_batch_num, max_len, vocab_size)
valid_batch_size = 1
valid_batch_num = 100
valid_batches = create_batches(valid_batch_size, valid_batch_num, max_len, vocab_size)

# valid
max_epoch = 100
opt = Adam()
Logging.configure(level=INFO)
#Profile.init(n=int(1e8))
# @profile train(encdec, train_batches, opt, max_epoch, valid_batches)
train(encdec, train_batches, opt, model_dir, max_epoch, valid_batches)
#li, lidict = Profile.retrieve()
#@save ARGS[1] li lidict

# ProfileView.view()
