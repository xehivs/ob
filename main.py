#!/usr/bin/env python

from pathlib import Path

from matplotlib.colors import cnames
from scipy import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler as scaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from skimage.feature import canny
import cv2


DATA = Path('./data/')
IMG = Path('./img/')

DATA.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def download_dataset(url: str):
    fn = url.split('/')[-1]
    fp = DATA / fn
    if fp.exists():
        print(f'{fp} was already downloaded')
        return

    print(f'downloading {fp}')

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

    X2 = scaler().fit(X).transform(X)

    n_components = 4

    pca = PCA(n_components=n_components).fit(X2, y)
    X_pca = pca.fit_transform(X2)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('Principal Components')
    ax.set_ylabel('Variance Ratio')
    ax.set_title('Variance ratio for PCA on Indian Pines dataset')
    ax.grid()
    ax.set_xticks(range(1, n_components + 1))
    ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_)
    fig.savefig(IMG / 'pca_components.png')

    colorlist = np.random.choice(
        list(cnames.keys()), unique_y, replace=False).tolist()

    colors = y.map(lambda x: colorlist[x])

    df = pd.DataFrame(X_pca[:, :2])
    df = pd.concat([df, y, colors], axis=1)
    df.columns = ['PC1', 'PC2', 'target', 'color']

    df_0 = df[df['target'] != 0]

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('PC-1')
    ax.set_ylabel('PC-2')
    ax.set_title('PCA on Indian Pines dataset')
    ax.grid()
    ax.scatter(df_0['PC1'], df_0['PC2'], color=df_0['color'], s=3)
    fig.savefig(IMG / 'pc1_pc2.png')

    img = (df['PC1'] + df['PC2']).values.reshape((145, 145))
    plt.imsave(IMG / 'pc12.png', img)

    c = canny(img, sigma=2., low_threshold=.15,
              high_threshold=.6, use_quantiles=True)
    plt.imsave(IMG / 'pc12_canny.png', c)

    gt2 = cv2.imread((IMG / 'gt.png').as_posix(), 0)
    plt.imsave(IMG / 'gt_canny.png', canny(gt2))

if __name__ == '__main__':
    main()
