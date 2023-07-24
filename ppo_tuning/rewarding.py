import re

colors = ['blue', 'green', 'red', 'orange', 'purple', 'yellow']


def str2grid(s):
    s = re.sub(',', '', s)
    vals = s.split(' ')
    grid = np.zeros((9, 20, 20))
    for i in range(0, len(vals), 4):
        if i + 4 > len(vals):
            break
        try:
            z = int(vals[i])
            x = int(vals[i + 1])
            y = int(vals[i + 2])
            if vals[i + 3] == 'remove':
                color = -1
                grid[z, x, y] = 0
            else:
                color = colors.index(vals[i + 3])
                grid[z, x, y] = color + 1
                print("add block")
        except:
            break
    return bbox(grid, bounds=(11, 11))


from gridworld.tasks import Task
import numpy as np


def get_metrics(target_grid, grid, start_grid=np.zeros((9, 11, 11))):
    task = Task('', target_grid=target_grid - start_grid)

    argmax = task.argmax_intersection(grid)
    builded = grid - start_grid
    maximal_intersection = task.get_intersection(builded, *argmax)

    target_size = task.target_size
    precision = maximal_intersection / (target_size + 1e-10)
    recall = maximal_intersection / (len(builded.nonzero()[0]) + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return maximal_intersection, f1


def count_metrics(res):
    metrics = []
    grids = []

    for i in range(len(res)):
        x, y = res[i]
        grid1 = str2grid(x)
        grid2 = str2grid(y)
        grids.append([grid1, grid2])
        m = get_metrics(grid2, grid1)[1]
        metrics.append(m)
    return metrics


def reinforce_loss(logits, labels):
    # dist = torch.distributions.categorical.Categorical(logits=logits)
    preds = logits  # dist.sample()
    # log_prob = dist.log_prob(preds)

    res_log_prob = []

    res = []
    for i in range(len(logits)):
        # id = (preds[i] == 0).nonzero()[0].item()
        # res_log_prob.append(log_prob[i][:id].sum())
        pred = preds[i]  # gpt2_tokenizer.decode(preds[i], skip_special_tokens=True)
        res.append([pred, labels[i]])

    # from utils import count_metrics
    metrics = count_metrics(res)
    return metrics


def bbox(grid, bounds):
    bgrid = np.zeros((9, *bounds))
    try:
        xmin = grid.nonzero()[1][0]
        ymin = grid.nonzero()[2][0]
    except IndexError:
        xmin = ymin = 0

    cut_grid = grid[:, xmin: xmin + bounds[0], ymin: ymin + bounds[1]]
    bgrid[:, :cut_grid.shape[1], :cut_grid.shape[2]] = cut_grid
    return bgrid