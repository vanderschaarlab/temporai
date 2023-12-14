"""
Adapted from:
https://github.com/ZhaozhiQIAN/SyncTwin-NeurIPS-2021

Citation:
@inproceedings{synctwin2021,
  title={SyncTwin: Treatment Effect Estimation with Longitudinal Outcomes},
  author={Qian, Zhaozhi and Zhang, Yao and Bica, Ioana and Wood, Angela and van der Schaar, Mihaela},
  booktitle={Advances in neural information processing systems},
  year={2021}
}
"""
# mypy: ignore-errors


import numpy as np
import numpy.random
import scipy.integrate
import torch


def _f(t, y, Kin, K, O, H, D50):  # noqa: E741
    P = y[0]
    R = y[1]
    D = y[2]

    dP = Kin[int(t)] - K * P
    dR = K * P - (D / (D + D50)) * K * R
    dD = O[int(t)] - H * D

    return [dP, dR, dD]


def _solve(init, Kin, K, Os, H, D50, step=30):
    ode = scipy.integrate.ode(_f).set_integrator("dopri5")

    Ot = np.zeros(step + 1)
    if Os >= 0:
        Ot[Os:] = 1.0

    try:
        len(Kin)
    except Exception:  # pylint: disable=broad-except
        Kin = np.ones(step + 1) * Kin

    ode.set_initial_value(init, 0).set_f_params(Kin, K, Ot, H, D50)
    t1 = step
    dt = 1

    res_list = []

    while ode.successful() and ode.t < t1:
        res = ode.integrate(ode.t + dt, ode.t + dt)  # type: ignore
        res_list.append(res)

    res = np.stack(res_list, axis=-1)
    return res


def _get_Kin(step=30, n_basis=12):
    # define Kin
    Kin_b_list = list()
    Kin_b_list.append(np.ones(step + 1))
    x = np.arange(step + 1) / step
    Kin_b_list.append(x)

    for i in range(n_basis - 2):
        bn = 2 * x * Kin_b_list[-1] - Kin_b_list[-2]
        Kin_b_list.append(bn)

    Kin_b = np.stack(Kin_b_list, axis=-1)

    Kin_list = [Kin_b[:, i] for i in range(n_basis)]
    return Kin_list, Kin_b


def _get_clustered_Kin(Kin_b, n_cluster, n_sample_total):
    n_basis = Kin_b.shape[1]

    n_sample_cluster = n_sample_total // n_cluster
    if n_sample_total % n_cluster != 0:
        raise ValueError(f"Warning: sample size ({n_sample_total}) not divisible by number of clusters ({n_cluster})")

    # generate cluster masks
    mask_list = []
    for i in range(n_cluster):
        mask = np.zeros(n_basis)
        mask[i:-1:4] = 1.0
        mask_list.append(mask)

    Kin_list = []
    for mask in mask_list:
        for _ in range(n_sample_cluster):
            Kin = np.matmul(Kin_b, numpy.random.randn(n_basis) * mask)
            Kin_list.append(Kin)

    Kin_b = np.stack(Kin_list, axis=-1)
    return Kin_list, Kin_b


def _generate_data(Kin_list, K_list, P0_list, R0_list, train_step, H=0.1, D50=3, step=30):
    control_res_list = []

    for Kin in Kin_list:
        for K in K_list:
            for P0 in P0_list:
                for R0 in R0_list:
                    control_res = _solve([P0, R0, 0.0], Kin, K, train_step, H, D50, step)
                    control_res_list.append(control_res)

    control_res_arr = np.stack(control_res_list, axis=-1)
    # Dim, T, N
    # Dim = 3: precursor, Cholesterol, Statins concentration
    # slice on dim=1 to get the outcome of interest
    return control_res_arr


def _get_covariate(
    control_Kin_b,
    treat_Kin_b,
    control_res_arr,
    treat_res_arr,
    step=30,
    train_step=25,
    noise=None,
    double_up=False,
    hidden_confounder=0,
):
    n_units = control_res_arr.shape[-1] * 2 if double_up else control_res_arr.shape[-1]
    n_treated = treat_res_arr.shape[-1]

    covariates_control = np.concatenate([control_Kin_b[:step, :][None, :, :], control_res_arr], axis=0)
    covariates_treated = np.concatenate([treat_Kin_b[:step, :][None, :, :], treat_res_arr], axis=0)
    covariates = np.concatenate([covariates_control, covariates_treated], axis=2)

    covariates = torch.tensor(covariates, dtype=torch.float32)

    covariates = covariates.permute((1, 2, 0))
    # remove the last covariate
    covariates = covariates[:, :, :3]

    # standardize
    m = covariates.mean(dim=(0, 1))
    sd = covariates.std(dim=(0, 1))
    covariates = (covariates - m) / sd

    if double_up:
        covariates_control = covariates[:, : (covariates.shape[1] // 2), :]
        covariates_twin = covariates_control + torch.randn_like(covariates_control) * 0.1
        covariates = torch.cat([covariates_twin, covariates], dim=1)

    if noise is not None:
        covariates = covariates + torch.randn_like(covariates) * noise

    n_units_total = n_units + n_treated

    pretreatment_time = train_step

    x_full = covariates[:pretreatment_time, :, :]
    if hidden_confounder == 1:
        x_full[:, :, 0] = 0
    if hidden_confounder == 2:
        x_full[:, :, 0] = 0
        x_full[:, :, 1] = 0
    y_full = covariates[pretreatment_time:, :, 2:3].detach().clone()
    y_full_all = covariates[pretreatment_time:, :, :]
    y_control = covariates[pretreatment_time:, :n_units, 2:3]

    t_full = torch.ones_like(x_full)
    mask_full = torch.ones_like(x_full)
    batch_ind_full = torch.arange(n_units_total)
    y_mask_full = (batch_ind_full < n_units) * 1.0
    return (
        (n_units, n_treated, n_units_total),
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,
        y_full_all,
        m,
        sd,
    )


def _get_treatment_effect(treat_res_arr, treat_counterfactual_arr, train_step, m, sd):
    m = m[2:3].item()
    sd = sd[2:3].item()

    treat_res_arr = (treat_res_arr - m) / sd
    treat_counterfactual_arr = (treat_counterfactual_arr - m) / sd

    # The below 1:2 looks odd, but it does in fact correctly correspond to outcomes.
    return torch.tensor(treat_res_arr - treat_counterfactual_arr).permute((1, 2, 0))[train_step:, :, 1:2]


# ======================================================================================================================
# Main functions:


# TODO: Missing / irregular data scenario.
def generate(
    seed: int = 100,
    train_step: int = 25,
    step: int = 30,
    control_sample: int = 200,
    treatment_sample: int = 200,
    hidden_confounder: int = 0,  # ["0", "1", "2"]
):
    assert 0 < treatment_sample <= control_sample
    assert step > train_step > 0
    assert 0 <= hidden_confounder <= 2

    noise: float = 0.1
    n_basis: int = 6
    n_cluster: int = 2

    K_list = [0.18]
    P0_list = [0.5]
    R0_list = [0.5]

    # print("Data generation with seed {}".format(seed))
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    # print("Generating data")
    _, Kin_b = _get_Kin(step=step, n_basis=n_basis)
    control_Kin_list, control_Kin_b = _get_clustered_Kin(Kin_b, n_cluster=n_cluster, n_sample_total=control_sample)
    treat_Kin_list, treat_Kin_b = _get_clustered_Kin(Kin_b, n_cluster=n_cluster, n_sample_total=control_sample * 2)
    treat_Kin_list = treat_Kin_list[:treatment_sample]
    treat_Kin_b = treat_Kin_b[:, :treatment_sample]

    control_res_arr = _generate_data(
        control_Kin_list, K_list, P0_list, R0_list, train_step=-1, H=0.1, D50=0.1, step=step  # type: ignore
    )
    treat_res_arr = _generate_data(
        treat_Kin_list, K_list, P0_list, R0_list, train_step=train_step, H=0.1, D50=0.1, step=step  # type: ignore
    )
    # treat_counterfactual_arr = pkpd.generate_data(
    #     treat_Kin_list, K_list, P0_list, R0_list, train_step=-1, H=0.1, D50=0.1, step=step
    # )

    (
        _,  # n_tuple
        x_full,
        t_full,
        mask_full,
        batch_ind_full,
        y_full,
        y_control,
        y_mask_full,  # y_mask_full
        y_full_all,  # y_full_all
        _,  # m
        _,  # sd
    ) = _get_covariate(
        control_Kin_b,
        treat_Kin_b,
        control_res_arr,
        treat_res_arr,
        step=step,
        train_step=train_step,
        noise=noise,
        double_up=False,
        hidden_confounder=hidden_confounder,
    )

    # treatment_effect = pkpd.get_treatment_effect(treat_res_arr, treat_counterfactual_arr, train_step, m, sd)
    # n_units, n_treated, n_units_total = n_tuple

    return (
        x_full.numpy().astype(float),
        t_full.numpy().astype(float),
        mask_full.numpy().astype(float),
        batch_ind_full.numpy().astype(int),
        y_full.numpy().astype(float),
        y_control.numpy().astype(float),
        y_mask_full.numpy().astype(float),
        y_full_all.numpy().astype(float),
    )
