# mypy: ignore-errors

import copy
import os

import numpy as np
import numpy.random
import torch
import torch.optim as optim


def get_batch_standard(batch_size, *args):
    array_list = args
    n_total = args[0].shape[1]

    # batching on dim=1
    batch_ind = numpy.random.choice(np.arange(n_total), batch_size)
    batch_ind = torch.tensor(batch_ind, dtype=torch.long).to(args[0].device)

    mini_batch = []

    for a in array_list:
        mini_batch.append(a[:, batch_ind, ...])
    return mini_batch


def get_folds(start, split, *args):
    a = args
    ret = []

    for x in a:
        if x.dim() == 3:
            y = x[:, start::split, :]
        else:
            y = x[start::split]
        ret.append(y)
    return ret


def create_paths(*args):
    for base_path in args:
        if not os.path.exists(base_path):
            os.makedirs(base_path)


def pre_train_reconstruction_prognostic_loss(
    nsc,
    x_full,
    t_full,
    mask_full,
    y_full,
    y_mask_full,
    x_full_val=None,
    t_full_val=None,
    mask_full_val=None,
    y_full_val=None,
    y_mask_full_val=None,
    niters=5000,
    test_freq=1000,
    batch_size=None,
):
    if x_full_val is None:
        x_full_val = x_full
        t_full_val = t_full
        mask_full_val = mask_full
        y_full_val = y_full
        y_mask_full_val = y_mask_full

    enc = nsc.encoder
    dec = nsc.decoder

    assert nsc.decoder_Y is not None
    dec_Y = nsc.decoder_Y

    optimizer = optim.Adam(list(dec.parameters()) + list(enc.parameters()) + list(dec_Y.parameters()))

    y_mask_full = torch.stack([y_mask_full] * dec_Y.max_seq_len, dim=0).unsqueeze(-1)

    test_freq = test_freq if test_freq > 0 else 1

    best_loss = 1e9

    enc_sd = None
    dec_sd = None
    dec_Y_sd = None
    for itr in range(1, niters + 1):
        optimizer.zero_grad()

        if batch_size is not None:
            x, t, mask, y, y_mask = get_batch_standard(  # pylint: disable=unbalanced-tuple-unpacking
                batch_size, x_full, t_full, mask_full, y_full, y_mask_full
            )
        else:
            x, t, mask, y, y_mask = x_full, t_full, mask_full, y_full, y_mask_full

        C = nsc.get_representation(x, t, mask)
        x_hat = nsc.get_reconstruction(C, t, mask)
        loss_X = nsc.reconstruction_loss(x, x_hat, mask)

        y_hat = nsc.get_prognostics(C, t, mask)
        loss_Y = nsc.prognostic_loss2(y, y_hat, y_mask)

        loss = loss_X + loss_Y
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                if x_full_val.shape[1] < 5000:
                    C = nsc.get_representation(x_full_val, t_full_val, mask_full_val)
                    x_hat = nsc.get_reconstruction(C, t_full_val, mask_full_val)
                    loss_X = nsc.reconstruction_loss(x_full_val, x_hat, mask_full_val)

                    y_hat = nsc.get_prognostics(C, t_full_val, mask_full_val)
                    loss_Y = nsc.prognostic_loss2(y_full_val, y_hat, y_mask_full_val)

                    loss = loss_X + loss_Y
                else:
                    loss = 0
                    n_fold = x_full_val.shape[1] // 500

                    for fold in range(n_fold):
                        (  # pylint: disable=unbalanced-tuple-unpacking
                            x_full_vb,
                            t_full_vb,
                            mask_full_vb,
                            y_full_vb,
                            y_mask_full_vb,
                        ) = get_folds(fold, n_fold, x_full_val, t_full_val, mask_full_val, y_full_val, y_mask_full_val)

                        C = nsc.get_representation(x_full_vb, t_full_vb, mask_full_vb)
                        x_hat = nsc.get_reconstruction(C, t_full_vb, mask_full_vb)
                        loss_X = nsc.reconstruction_loss(x_full_vb, x_hat, mask_full_vb)

                        y_hat = nsc.get_prognostics(C, t_full_vb, mask_full_vb)
                        loss_Y = nsc.prognostic_loss2(y_full_vb, y_hat, y_mask_full_vb)

                        loss = loss_X + loss_Y + loss

                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))  # type: ignore
                if loss < best_loss:
                    best_loss = loss

                    enc_sd = copy.deepcopy(enc.state_dict())
                    dec_sd = copy.deepcopy(dec.state_dict())
                    dec_Y_sd = copy.deepcopy(dec_Y.state_dict())

    enc.load_state_dict(enc_sd)
    dec.load_state_dict(dec_sd)
    dec_Y.load_state_dict(dec_Y_sd)

    return best_loss


def update_representations(nsc, x_full, t_full, mask_full, batch_ind_full):
    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)
        nsc.update_C0(C, batch_ind_full)


def load_pre_train_and_init(
    nsc, x_full, t_full, mask_full, batch_ind_full, model_path="models/sync/{}.pth", init_decoder_Y=False
):
    enc = nsc.encoder
    dec = nsc.decoder

    enc.load_state_dict(torch.load(model_path.format("encoder.pth")))
    dec.load_state_dict(torch.load(model_path.format("decoder.pth")))

    if init_decoder_Y:
        dec_Y = nsc.decoder_Y
        dec_Y.load_state_dict(torch.load(model_path.format("decoder_Y.pth")))

    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)
        nsc.update_C0(C, batch_ind_full)


def train_B_self_expressive(
    nsc,
    x_full,
    t_full,
    mask_full,
    batch_ind_full,
    niters=20000,
    batch_size=None,
    lr=1e-3,
    test_freq=1000,
):
    # mini-batch training not implemented
    assert batch_size is None

    optimizer = optim.Adam([nsc.B], lr=lr)

    test_freq = test_freq if test_freq > 0 else 1

    best_loss = 1e9

    with torch.no_grad():
        C = nsc.get_representation(x_full, t_full, mask_full)

    nsc_sd = None
    for itr in range(1, niters + 1):
        optimizer.zero_grad()

        B_reduced = nsc.get_B_reduced(batch_ind_full)

        loss = nsc.self_expressive_loss(C, B_reduced)

        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                print("Iter {:04d} | Total Loss {:.6f}".format(itr, loss.item()))
                if np.isnan(loss.item()):
                    raise RuntimeError("NaN loss encountered")
                if loss < best_loss:
                    best_loss = loss
                    nsc_sd = copy.deepcopy(nsc.state_dict())

    nsc.load_state_dict(nsc_sd)

    return 0


def get_prediction(nsc, batch_ind_full, y_control, itr=500):
    batch_ind_full = batch_ind_full.to(nsc.B.device)
    y_control = y_control.to(nsc.B.device)
    y_hat_list = list()
    for _ in range(itr):
        with torch.no_grad():
            B_reduced = nsc.get_B_reduced(batch_ind_full)
            y_hat = torch.matmul(B_reduced, y_control)
            if torch.sum(torch.isinf(y_hat)).item() == 0:
                y_hat_list.append(y_hat)

    y_hat_mat = torch.stack(y_hat_list, dim=-1)
    y_hat_mat[torch.isinf(y_hat_mat)] = 0.0
    y_hat2 = torch.mean(y_hat_mat, dim=-1)
    return y_hat2


def get_treatment_effect(nsc, batch_ind_full, y_full, y_control, itr=500):
    batch_ind_full = batch_ind_full.to(nsc.B.device)
    y_control = y_control.to(nsc.B.device)
    y_full = y_full.to(nsc.B.device)
    y_hat_list = list()
    for _ in range(itr):
        with torch.no_grad():
            B_reduced = nsc.get_B_reduced(batch_ind_full)
            y_hat = torch.matmul(B_reduced, y_control)
            if torch.sum(torch.isinf(y_hat)).item() == 0:
                y_hat_list.append(y_hat)

    y_hat_mat = torch.stack(y_hat_list, dim=-1)
    y_hat_mat[torch.isinf(y_hat_mat)] = 0.0
    y_hat2 = torch.mean(y_hat_mat, dim=-1)
    return (y_full - y_hat2)[:, nsc.n_unit :, :], y_hat2
