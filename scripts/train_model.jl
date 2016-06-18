doc = """Create a vocabulary from a tokenized file.

Usage:
  create_vocabulary.jl <src-train> <trg-train> <src-valid> <trg-valid> <src-vocab> <trg-vocab> <vocab-size> <model-dir>
      [--layer <n>] [--hidden <n>]
      [--batch <n>] [--epoch <n>] [--max-length <n>]

Options:
  --layer <n>        number of layers [default: 1]
  --hidden <n>       hidden layer size [default: 128]
  --batch <n>        batch size [default: 32]
  --epoch <n>        maximum number of epochs [default: 100]
  --max-length <n>   filter samples with longer than this length [default: 1000]
"""

using Merlin
using DocOpt
using Logging

include("../vocabulary.jl")
include("../corpus.jl")
include("../attention.jl")
include("../train.jl")

function filter_batches!(batches, max_length)
  filter!(batch -> size(batch[1], 2) <= max_length && size(batch[2], 2) <= max_length, batches)
end

function main(args)
  # create model
  vocab_size = parse(Int, args["<vocab-size>"])
  layer_num = parse(Int, args["--layer"])
  hidden_dim = parse(Int, args["--hidden"])
  input_feeding = false
  encdec = AttentionalEncoderDecoder(vocab_size, layer_num, hidden_dim, input_feeding)

  # load data
  bucket_step = 5
  model_dir = args["<model-dir>"]
  vocab_src_path = args["<src-vocab>"]
  vocab_trg_path = args["<trg-vocab>"]
  train_src_path = args["<src-train>"]
  train_trg_path = args["<trg-train>"]
  valid_src_path = args["<src-valid>"]
  valid_trg_path = args["<trg-valid>"]
  batch_size = parse(Int, args["--batch"])
  max_epoch = parse(Int, args["--epoch"])
  max_length = parse(Int, args["--max-length"])
  println("Loading $vocab_src_path")
  vocab_src = load_vocabulary(vocab_src_path, vocab_size)
  println("Loading $vocab_trg_path")
  vocab_trg = load_vocabulary(vocab_trg_path, vocab_size)
  println("Loading $train_src_path and $train_trg_path")
  corpus_train = load_bitext(train_src_path, train_trg_path, vocab_src, vocab_trg)
  println("Loading $valid_src_path and $valid_trg_path")
  corpus_valid = load_bitext(valid_src_path, valid_trg_path, vocab_src, vocab_trg)
  println("Creating training batches")
  batches_train = create_batches(corpus_train, batch_size, bucket_step, EOS_ID)  # TODO: support ignoring ID
  println("Creating validation batches")
  batches_valid = create_batches(corpus_valid, batch_size, bucket_step, EOS_ID)
  filter_batches!(batches_train, max_length)
  filter_batches!(batches_valid, max_length)

  mkpath(model_dir)
  Logging.configure(level=INFO, filename=joinpath(model_dir, "log"))

  # valid
  opt = Adam()
  train(encdec, batches_train, opt, model_dir, max_epoch, batches_valid)
end

args = docopt(doc)
main(args)
