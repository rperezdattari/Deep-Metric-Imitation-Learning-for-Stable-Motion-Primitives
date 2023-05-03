import numpy as np
import similaritymeasures as sm


def get_RMSE(sim_trajectories, demos, eval_indexes=None, verbose=True):
    RMSE = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating RMSE; trajectory:', k + 1)
        if eval_indexes is not None:
            RMSE.append(np.sqrt(np.mean((sim_trajectories[eval_indexes[k], k, :] - demos[eval_indexes[k], k, :]) ** 2)))
        else:
            RMSE.append(np.sqrt(np.mean((sim_trajectories[:, k, :] - demos[:, k, :]) ** 2)))
    return RMSE


def get_DTWD(sim_trajectories, demos, eval_indexes=None, verbose=True):
    DTWD = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating dynamic time warping distance; trajectory:', k + 1)
        if eval_indexes is not None:
            dtw, d = sm.dtw(sim_trajectories[eval_indexes[k], k, :], demos[eval_indexes[k], k, :])
            DTWD.append(dtw / len(eval_indexes[k]))
        else:
            dtw, d = sm.dtw(sim_trajectories[:, k, :], demos[:, k, :])
            DTWD.append(dtw / sim_trajectories.shape[0])

    return DTWD


def get_FD(sim_trajectories, demos, eval_indexes=None, verbose=True):
    FD = []
    for k in range(sim_trajectories.shape[1]):
        if verbose:
            print('Calculating Frechet distance; trajectory:', k + 1)
        if eval_indexes is not None:
            FD.append(sm.frechet_dist(sim_trajectories[eval_indexes[k], k, :], demos[eval_indexes[k], k, :]))
        else:
            FD.append(sm.frechet_dist(sim_trajectories[:, k, :], demos[:, k, :]))

    return FD
