doc = """Create a vocabulary from a tokenized file.

Usage:
  create_vocabulary.jl <path> [--lower]

Options:
  --lower   Lower-case tokens.
"""

using DocOpt
using DataStructures: DefaultDict

function main(args)
  path = args["<path>"]
  lower = args["--lower"]

  freqs = DefaultDict(AbstractString, Int, 0)

  open(path, "r") do fp
    for line in eachline(fp)
      toks = split(chomp(line))
      for tok in toks
        if lower
          tok = lowercase(tok)
        end
        freqs[tok] += 1
      end
    end
  end

  for (tok, freq) in sort(collect(zip(keys(freqs), values(freqs))), by=x->-x[2])
    @printf "%s\t%d\n" tok freq
  end
end

args = docopt(doc)
main(args)
