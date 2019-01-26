#!/usr/bin/env python

from pathlib import Path

from matplotlib.colors import cnames
from scipy import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


DATA = Path('./dataset/')
IMG = Path('./img/')

DATA.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def download_dataset(url: str):
    fn = url.split('/')[-1]
    fp = DATA / fn
    if fp.exists():
        print(f'{fp} was already downloaded')
        return

    resp = requests.get(url)
    resp.raise_for_status()

    with open(fp, 'wb') as f:
        f.write(resp.content)
    print(f'{fp} saved')


def load_data(fn: str) -> np.ndarray:
    mat = io.loadmat(fn)
    keys = [i for i in mat.keys() if not i.startswith('__')]
    if len(keys) != 1:
        raise Exception(f'to many keys: {keys}')
    key = keys[0]
    return mat[key]


def scale2int(x: np.ndarray) -> np.ndarray:
    a = x - x.min()
    b = a / a.max() * 255
    c = b.astype(int)
    return c


def get_data(fp: Path, gt: np.ndarray, ipc: np.ndarray) -> pd.DataFrame:
    if fp.exists():
        return pd.read_csv(fp)

    cols, rows, bands = ipc.shape

    gt2 = gt.reshape((-1, 1))
    ipc2 = ipc.reshape((cols * rows, bands))
    c = np.hstack([ipc2, gt2])

    columns = [f'band_{i}' for i in range(bands)] + ['target']

    df = pd.DataFrame(c, columns=columns)
    df.to_csv(fp, index=False)

    return df


def main():
    np.random.seed(42)

    urls = [
        'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
        'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
    ]
    for url in urls:
        download_dataset(url)

    gt = load_data(DATA / 'Indian_pines_gt.mat')
    plt.imsave(IMG / 'gt.png', gt)

    ipc = load_data(DATA / 'Indian_pines_corrected.mat')

    plt.imsave(IMG / '111.png', scale2int(ipc[..., 111]))

    data = get_data(DATA / 'indian_pines.csv', gt, ipc)

    X = data.copy().astype(np.float64)
    y = X.pop('target').astype(int)
    unique_y = len(y.unique())

    # print('X.describe')
    # print(X.describe())

    sc = StandardScaler().fit(X)
    X2 = sc.transform(X)

    # print('X2.describe')
    # print(pd.DataFrame(X2).describe())

    pca = PCA(n_components=2).fit(X2, y)

    # X_r2 = pca.fit_transform(X2)
    # print(X_r2.shape)

    c1, c2 = pca.transform(X2).reshape((2, -1))

    colorlist = np.random.choice(
        list(cnames.keys()), unique_y, replace=False).tolist()

    colors = y.map(lambda x: colorlist[x])

    print(colors)

    plt.scatter(c1, c2, color=colors)
    plt.show()

    # print('X3.describe')
    # print(pd.DataFrame(X3).describe())


if __name__ == '__main__':
    main()
