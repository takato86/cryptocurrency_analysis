import glob
import os
import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import logging
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(funcName)s:%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


START_DATE_IN_FILE = '2013-01-01 00:00:00'
END_DATE_IN_FILE = '2020-01-03 00:00:00'
INPUT_DIR = os.path.join("..", "data", "top10")


def fill(df):
    # df.interpolate(inplace=True)
    # df.fillna(0, inplace=True)
    rdf = df.dropna(how='all')
    return rdf


# 1日ごとの差分を求める
def get_diff(df):
    df_diff = df.diff()
    df_diff.dropna(how='all', inplace=True)
    return df_diff


# 期間の指定
def designate_dur(df, start, end):
    rdf = df.reset_index()
    rdf = rdf[(rdf["index"] >= start) & (rdf["index"] <= end)]
    rdf.set_index("index", inplace=True, drop=True)
    return rdf


def from_adj_to_edge(df):
    li = df.values.tolist()
    label = df.index.tolist()
    from_l = []
    to_l = []
    w_l = []
    for i in range(len(li)):
        for j in range(i + 1, len(li)):
            from_l.append(label[i])
            to_l.append(label[j])
            w_l.append(li[i][j])
    rdf = pd.DataFrame()
    rdf["from"] = from_l
    rdf["to"] = to_l
    rdf["weight"] = w_l
    return rdf


def transform_variable(x):
    return np.sqrt(2*(1-x))


def perform_clustering(tran_corr_diff_price, file_path):
    logger.info("Start perform clustering...")
    li = tran_corr_diff_price.values.tolist()
    pdist = squareform(li)
    label = tran_corr_diff_price.index.tolist()

    res = linkage(pdist, method='ward')
    fig = plt.figure(figsize=(12, 8))
    fig.add_subplot(1, 1, 1)
    dendrogram(res, labels=label) 
    plt.title("Dendrogram")
    locs, labels = plt.xticks()
    # print(0.7 * max(res[:, 2]))
    plt.savefig(file_path)
    plt.close()


def perform_mst(tran_corr_diff_price, file_path):
    logger.info("Start perform minimum spanning tree...")
    G = nx.from_pandas_adjacency(tran_corr_diff_price)
    T = nx.minimum_spanning_tree(G)
    plt.figure(figsize=(10, 10))
    edge_labels = {(i, j): round(w["weight"], 3) for i, j, w in T.edges(data=True)}
    pos = nx.spring_layout(T, k=0.5)
    nx.draw_networkx_nodes(T, pos, node_size=1000, node_color="blue")
    nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(T, pos, font_size=15, font_color="white")
    nx.draw_networkx_edges(T, pos)
    plt.savefig(file_path)
    plt.close()
    

def plot_time_series(price_df, file_path):
    logger.info("Plotting time series graph...")
    price_df.fillna(method="ffill", inplace=True)
    scaler = MinMaxScaler()
    scaled_price_df \
        = pd.DataFrame(scaler.fit_transform(price_df),
                       columns=price_df.columns, index=price_df.index)
    scaled_price_df.index.name = "date"
    scaled_price_df.plot(figsize=(14, 10))
    plt.savefig(file_path)
    plt.close()


def plot_corr_heatmap(diff_price, file_path):
    logger.info("Plotting correlation graph...")
    autocorrs = [diff_price["btc"].autocorr(lag=i) for i in range(60)]
    print("max: {}, index: {}".format(np.max(autocorrs[1:]),
                                      np.argmax(autocorrs[1:])))
    plt.figure(figsize=(9, 7))
    sns.heatmap(diff_price.corr(), vmax=1, vmin=-1, center=0, annot=True)
    plt.savefig(file_path)
    plt.close()


def read_files(pattern):
    logger.info("Loading files...")
    files = glob.glob(pattern)
    cur_dic = {}
    for file in files:
        key = os.path.basename(file).split("-")[0]
        cur_dic[key] = pd.read_csv(file, index_col=["snapped_at"],
                                   parse_dates=["snapped_at"])
    print(cur_dic.keys())

    date_key = pd.date_range(start=START_DATE_IN_FILE, end=END_DATE_IN_FILE,
                             freq='D')
    print(date_key)
    price_df = pd.DataFrame(index=date_key)
    market_cap = pd.DataFrame(index=date_key)
    total_volume = pd.DataFrame(index=date_key)

    for key, val in cur_dic.items():
        price_df[key] = val["price"]
        market_cap[key] = val["market_cap"]
        total_volume[key] = val["total_volume"]

    price_df = fill(price_df)
    market_cap = fill(market_cap)
    total_volume = fill(total_volume)
    return [price_df, market_cap, total_volume]


def analyze_period(price_df, start, end):
    price_dur_df = designate_dur(price_df, start, end)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    corr_output_path = os.path.join(output_dir,
                                    f"{start_str}_{end_str}_corr.png")
    time_series_path = os.path.join(output_dir,
                                    f"{start_str}_{end_str}_time_series.png")
    dendrogram_path = os.path.join(output_dir,
                                   f"{start_str}_{end_str}_dendrogram.png")
    mst_path = os.path.join(output_dir, f"{start_str}_{end_str}_mst.png")
    plot_time_series(price_dur_df, time_series_path)

    diff_price = get_diff(price_dur_df)
    diff_price = diff_price.dropna(how='all', axis=1)

    plot_corr_heatmap(diff_price, corr_output_path)

    corr_diff_price = diff_price.corr()
    tran_corr_diff_price = corr_diff_price.apply(transform_variable)
    perform_mst(tran_corr_diff_price, mst_path)

    perform_clustering(tran_corr_diff_price, dendrogram_path)


def main():
    pattern = os.path.join(INPUT_DIR, "*")
    price_df, market_cap, total_volume = read_files(pattern)
    start = pd.to_datetime(args.start)
    period_candidates = pd.date_range(start=args.start,
                                      end=args.end,
                                      freq=args.freq).tolist()
    for end in period_candidates:
        logger.info(f"Analyze {start} to {end}")
        analyze_period(price_df, start, end)
        start = end + dt.timedelta(days=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01",
                        help="YYYY-MM-DD形式で入力してください．")
    parser.add_argument("--end", default="2019-12-31",
                        help="YYYY-MM-DD形式で入力してください．")
    parser.add_argument("--freq", default="Q",
                        help="startからendをどの感覚で分析するか？pandasのdate_rangeを参照してください．")
    args = parser.parse_args()
    output_dir = os.path.join("..", "output", f"{args.freq}_{args.start}_{args.end}")
    if not os.path.exists(output_dir):
        logger.info("Making dir @ {}".format(output_dir))
        os.makedirs(output_dir)
    main()
