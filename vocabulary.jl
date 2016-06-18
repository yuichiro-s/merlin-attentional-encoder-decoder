type Token
  word::UTF8String
  wordid::Int
  charids::Vector{Int}
  tagid::Int
end

type Vocabulary{T}
  word_to_id::Dict{T, Int}
  id_to_word::Vector{T}
  size::Int
end

function add_word{T}(word_to_id::Dict{T, Int}, id_to_word::Vector{T}, word::T)
  next_id = length(id_to_word) + 1
  word_to_id[word] = next_id
  push!(id_to_word, word)
end

const EOS_WORD = "<EOS>"
const UNK_WORD = "<UNK>"
const EOS_ID = 1
const UNK_ID = 2

function load_vocabulary(path, max_size::Int=nothing)
  word_to_id = Dict{AbstractString, Int}()
  id_to_word = Vector{AbstractString}()

  add_word(word_to_id, id_to_word, EOS_WORD)
  add_word(word_to_id, id_to_word, UNK_WORD)
  open(path, "r") do fp
    for line in eachline(fp)
      es = split(chomp(line), "\t")
      if max_size != nothing && length(id_to_word) >= max_size
        break
      end
      add_word(word_to_id, id_to_word, es[1])
    end
  end

  Vocabulary(word_to_id, id_to_word, length(id_to_word))
end

function lookup_id{T}(vocab::Vocabulary{T}, word::T)
  get(vocab.word_to_id, word, UNK_ID)
end

function lookup_word{T}(vocab::Vocabulary{T}, id::Int)
  if id <= vocab.size
    vocab.id_to_word[id]
  else
    UNK_WORD
  end
end
