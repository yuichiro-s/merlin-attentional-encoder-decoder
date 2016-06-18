using DataStructures: DefaultDict

function convert_to_ids(line, vocab)
  toks = split(lowercase(chomp(line)))
  map(tok -> lookup_id(vocab, tok), toks)
end

function load_bitext(src_path, trg_path, src_vocab::Vocabulary, trg_vocab::Vocabulary)
  fp_src = open(src_path, "r")
  fp_trg = open(trg_path, "r")
  corpus = []
  for (line_src, line_trg) in zip(eachline(fp_src), eachline(fp_trg))
    ids_src = convert_to_ids(line_src, src_vocab)
    ids_trg = convert_to_ids(line_trg, trg_vocab)
    push!(corpus, (ids_src, ids_trg))
  end
  corpus
end

function create_batches(corpus, batch_size::Int, bucket_step::Int, pad_id::Int, reverse_input::Bool=false)
  T = Tuple{Vector{Int}, Vector{Int}}
  buckets = DefaultDict(Tuple{Int, Int}, Vector{T}, () -> T[])
  for (ids_src, ids_trg) in corpus
    len_src = length(ids_src)
    len_trg = length(ids_trg)
    # bucket_src = len_src  # no rounding for source sequence
    bucket_src = (div(len_src - 1, bucket_step) + 1) * bucket_step
    bucket_trg = (div(len_trg - 1, bucket_step) + 1) * bucket_step
    if reverse_input
      ids_src = reverse(ids_src)
    end
    push!(buckets[(bucket_src, bucket_trg)], (ids_src, ids_trg))
  end

  batches = []
  for ((len_src, len_trg), samples) in zip(keys(buckets), values(buckets))
    for i in 1:batch_size:length(samples)
      samples_batch = samples[i:min(end,i+batch_size-1)]
      src_val = fill(pad_id, length(samples_batch), len_src)
      trg_val = fill(pad_id, length(samples_batch), len_trg)
      for (j, (ids_src, ids_trg)) in enumerate(samples_batch)
        src_val[j, 1+end-length(ids_src):end] = ids_src
        trg_val[j, 1:length(ids_trg)] = ids_trg
      end
      push!(batches, (src_val, trg_val))
    end
  end
  batches
end
