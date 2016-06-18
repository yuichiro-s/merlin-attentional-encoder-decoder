using Merlin
using JLD
using Logging

function create_vars(batch)
  xs_val, ts_val = batch
  batch_size, x_len = size(xs_val)
  t_len = size(ts_val, 2)
  xs = map(i -> Var(reshape(xs_val[:, i], 1, batch_size)), 1:x_len)
  ts = map(i -> Var(reshape(ts_val[:, i], 1, batch_size)), 1:t_len)
  xs, ts
end

function print_log(es)
  msg = join(map(e -> @sprintf("%s: %s", e[1], e[2]), es), "\t")
  info(msg)
end

function train(model, train_batches, opt, path::AbstractString, max_epoch::Int, valid_batches=nothing)
  info("Total number of batches: $(length(train_batches))")
  for epoch = 1:max_epoch
    for (batch_idx, i) in enumerate(randperm(length(train_batches)))
      xs, ts = create_vars(train_batches[i])
      batch_size = length(xs[1].value)
      eos = Var(fill(EOS_ID, batch_size))
      ts_val = train_batches[i][2]
      t_len = size(ts_val, 2)

      time1 = time()
      ys, as = model(xs, ts)  # [(vocab_size x batch_size)]

      # calculate loss
      loss = 0
      for (t, y) in zip(map(i -> ts_val[:, i], 1:t_len), ys[1:end-1])
        # y: (vocab_size x batch_size)
        # t: (batch_size,)
        loss += crossentropy(Var(t), y) # TODO: should be named softmax_crossentropy
      end
      loss += crossentropy(eos, ys[end])

      # average in batch direction
      #loss_avg = sum(loss, 2) * (1 / batch_size)  # TODO: should be able to write like this
      loss_avg = sum(loss, 2) .* Var(fill(Float32(1. / batch_size), 1, 1))
      time2 = time()

      # update parameters
      vars = gradient!(loss_avg)
      for v in vars
        isempty(v.args) && hasgrad(v) && update!(opt, v.value, v.grad)
      end
      time3 = time()

      print_log([
        ("epoch", string(epoch-1)),
        ("batch", string(batch_idx-1)),
        ("loss", string(loss.value[1])),
        ("loss_avg", string(loss_avg.value[1] / t_len)),
        ("batch_size", string(batch_size)),
        ("len_x", string(length(xs))),
        ("len_t", string(t_len)),
        ("time", string(round(Int, (time3 - time1) * 1000))),
        ("time_f", string(round(Int, (time2 - time1) * 1000))),
        ("time_b", string(round(Int, (time3 - time2) * 1000))),
      ])
    end

    # save model
    model_path = joinpath(path, @sprintf("epoch%d", epoch-1))
    @save model_path model

    # evaluate on validation set
    ok_train, tot_train = evaluate(model, train_batches[rand(1:length(train_batches), 3)])
    info(@sprintf("epoch: %d\ttrain_acc: %.2f%% [%d/%d]", epoch, (ok_train / tot_train * 100), ok_train, tot_train))
    if valid_batches != nothing
      ok_valid, tot_valid = evaluate(model, valid_batches)
      info(@sprintf("epoch: %d\tvalid_acc: %.2f%% [%d/%d]", epoch, (ok_valid / tot_valid * 100), ok_valid, tot_valid))
    end
  end
end

function evaluate(model, batches)
  ok = 0
  tot = 0
  for (i, batch) in enumerate(batches)
    xs, ts = create_vars(batch)
    ys, as = model(xs, ts)
    ts_val = batch[2]
    batch_size, t_len = size(ts_val)
    eos = fill(EOS_ID, batch_size)
    for (t, y, a) in zip(map(i -> ts_val[:, i], 1:t_len), ys[1:end-1], as[1:end-1])
      pred = argmax(y.value, 1)   # (batch_size,)
      a = argmax(a.value, 1)
      ok += sum(pred .== t)
      tot += length(t)
    end
    t = eos
    pred = argmax(ys[end].value, 1)
    a = argmax(as[end].value, 1)
    ok += sum(pred .== t)
    tot += length(t)
  end
  ok, tot
end
