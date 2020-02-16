# 仮想通貨時系列分析
## 準備
```
pip install -r requirements.txt
```

## 実行

```
cd src
python main.py --start=2019-01-01 --end=2019-12-25 --freq=Q
```

|オプション|説明|
|---|---|
|start|対象にする期間の開始日時，YYYY-MM-DD形式で指定する．2013-04-28~2019-12-30の間で指定可能|
|end|対象にする期間の終了日時，YYYY-MM-DD形式で指定する．2013-04-29~2019-12-31の間で指定可能|
|freq|対象期間を何分割にするかを指定できる．四半期ごとであれば"Q"．ここの指定は[pandasのoffset alianses](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases)にfreqに遵守．|


