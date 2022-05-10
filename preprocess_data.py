import os
import argparse

import numpy as np
import pandas as pd


def load_and_save_data(seq_len=12, output_dir='data'):
    df = pd.read_csv('data/envitus_fimi_overlapped.csv', header=0)

    devices = ['e', '1', '14', '20', '27', '30']
    attrs = ['PM2_5', 'PM10', 'temp', 'humidity', 'CO', 'NO2', 'SO2']

    X, y = [], []
    
    for idx, id in enumerate(devices):
        ls_att = ['_'.join([a, id]) for a in attrs]
        
        for i in range(seq_len, df.shape[0], seq_len):
                ts = df[ls_att][i - seq_len:i]
                
                if id == 'e':
                    y.append(ts)
                else:
                    X.append(ts)
    
    X = np.array(X, dtype=np.float)
    y = np.array(y, dtype=np.float)
    y = np.repeat(y, len(devices) - 1, axis=0)
    
    print(f"X shape: {X.shape} y shape: {y.shape}")

    num_samples = X.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    X_train, y_train = X[:num_train], y[:num_train]
    X_val, y_val = (
        X[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    X_test, y_test = X[-num_test:], y[-num_test:]
    print(np.linalg.norm(X_test[:, :4] - y_test[:, :4]) / num_test)
    # print(np.linalg.norm(X_test[:, :4] - y_test[:, :4], ord=1))
    exit()

    for cat in ['train', 'val', 'test']:
        _x, _y = locals()["X_" + cat], locals()["y_" + cat]
        print(cat, "X: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=12, help="Sequence Length",)
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    load_and_save_data(args.seq_len, args.output_dir)
