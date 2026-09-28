"""
A reusable PyTorch training loop.

Implement:
    train(model, train_loader, val_loader, optimizer, epochs, device, grad_clip=None,
          scheduler=None, log_every=100, ckpt_path=None) -> history dict
        model.train(); per batch: forward, loss, optimizer.zero_grad(), loss.backward(),
        optional clip_grad_norm_, optimizer.step(), scheduler.step()
        save the best checkpoint (model and optimizer state_dict, plus the epoch)
    evaluate(model, loader) -> metrics, run under torch.no_grad() with model.eval()
    overfit_single_batch(model, batch, steps=200) -> final loss     # a debugging tool
    set_seed(seed)

Notes:
    The classic bugs this loop must avoid: forgetting zero_grad; forgetting eval() and
    no_grad() during validation; applying softmax before CrossEntropyLoss; and accumulating
    loss tensors instead of loss.item(), which keeps the computation graph alive.
"""