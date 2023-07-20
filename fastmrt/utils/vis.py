from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from typing import Any

@contextmanager
def draw_bland_altman_fig(ba_mean: Tensor, ba_error: Tensor, ba_error_mean: Tensor, ba_error_std: Tensor):

    # calculate datas
    mean_error = ba_error_mean.cpu().numpy()
    loa_upper_limit = ba_error_mean.cpu().numpy() + 1.96 * ba_error_std.cpu().numpy()
    loa_lower_limit = ba_error_mean.cpu().numpy() - 1.96 * ba_error_std.cpu().numpy()

    # start plot
    text_start = ba_mean.cpu().numpy().min()
    fig = plt.figure()
    plt.scatter(ba_mean.cpu().numpy(), ba_error.cpu().numpy())
    plt.axhline(mean_error, color='gray', linestyle='-')
    plt.axhline(loa_upper_limit, color='red', linestyle='--')
    plt.axhline(loa_lower_limit, color='red', linestyle='--')
    plt.text(text_start, mean_error, "mean: {:.3f}".format(mean_error))
    plt.text(text_start, loa_upper_limit, "upper limit: {:.3f}".format(loa_upper_limit))
    plt.text(text_start, loa_lower_limit, "lower limit: {:.3f}".format(loa_lower_limit))
    plt.xlabel("Mean of Recon and Full Tmap Patches (℃)")
    plt.ylabel("Difference (℃)")
    plt.title("Bland-Altman Analysis")
    yield plt

    plt.close(fig)

@contextmanager
def draw_linear_regression_fig(patch_recon_tmap: Tensor, patch_full_tmap: Tensor):

    # calculate datas
    data_x = patch_recon_tmap.flatten().cpu().numpy()
    data_y = patch_full_tmap.flatten().cpu().numpy()
    [k, b] = np.polyfit(data_x, data_y, deg=1)
    ref_x = np.linspace(np.min(data_x), np.max(data_x), 10)
    ref_y = ref_x
    fit_x = ref_x
    fit_y = k * fit_x + b

    # start plot
    fig = plt.figure()
    plt.plot(data_x, data_y, '.')
    plt.plot(ref_x, ref_y, color="red", linestyle="--")
    plt.plot(fit_x, fit_y, color="blue", linestyle="-")
    plt.text(ref_x[7], ref_y[4], "y={:.3f}x+{:.3f}".format(k, b))
    plt.xlabel("Temperature of Recon Tmap (℃)")
    plt.ylabel("Temperature of Full Tmap (℃)")
    plt.title("Linear Regression Analysis")

    yield plt
    plt.close(fig)

@contextmanager
def draw_tmap(tmap: Tensor, vmin: Any=0.0, vmax: Any=70.0) -> None:
    fig = plt.figure()
    plt.imshow(tmap.cpu(), cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    
    yield plt
    plt.close(fig)