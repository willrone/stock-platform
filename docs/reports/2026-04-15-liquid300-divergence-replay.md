# liquid300 alpha158 vs alpha360 divergence replay

Compared formal tasks:
- alpha158: `54a3681f-b995-470d-b5a8-e67f70d3b972`
- alpha360: `0e2e03b4-a830-452c-822a-f02154a9ac48`

Selected divergence stocks:
- 002611.SZ, 002371.SZ, 300085.SZ, 600809.SH, 600760.SH, 002517.SZ, 000066.SZ, 300383.SZ, 300458.SZ, 002050.SZ, 002466.SZ, 600988.SH, 002896.SZ, 600584.SH, 002465.SZ, 300207.SZ

## Return delta summary
```
stock_code  return_alpha158  trades_alpha158  return_alpha360  trades_alpha360  delta_158_minus_360
 002611.SZ     1.029023e+06               20    -1.200477e+06               24         2.229500e+06
 002371.SZ     1.543211e+06               19    -4.758126e+05                2         2.019024e+06
 300085.SZ     9.863652e+05               18    -9.954338e+05               13         1.981799e+06
 600809.SH     1.861483e+06               13    -2.171567e+04                1         1.883199e+06
 600760.SH     1.383530e+06               19    -3.188250e+05                8         1.702355e+06
 002517.SZ     3.026458e+05               11    -1.338991e+06               24         1.641637e+06
 000066.SZ     9.480807e+04               20    -1.487707e+06               22         1.582515e+06
 300383.SZ     5.119490e+05               13    -9.911032e+05               12         1.503052e+06
 300207.SZ    -8.650763e+05               20     7.895234e+05               25        -1.654600e+06
 002465.SZ    -7.516859e+05               12     9.269847e+05               25        -1.678671e+06
 600584.SH    -2.167023e+05               20     1.497824e+06               15        -1.714527e+06
 002896.SZ    -1.065917e+06               18     7.415383e+05                5        -1.807455e+06
 600988.SH    -1.953733e+06               12    -1.400583e+05               26        -1.813674e+06
 002466.SZ    -1.255275e+06               18     5.743126e+05                2        -1.829587e+06
 002050.SZ    -4.105246e+05               13     1.506506e+06               18        -1.917031e+06
 300458.SZ    -2.593625e+06               18     2.338252e+05                4        -2.827451e+06
```

## 002611.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": 1029023.03,
    "trade_count": 20,
    "win_rate": 0.8,
    "avg_hold_days": 14.0,
    "signal_total": 40,
    "executed_signals": 40,
    "executed_buy_signals": 20,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-05",
      -46543.842536354205
    ],
    "best_month": [
      "2020-02",
      257017.4812046052
    ],
    "worst_trade_pnl": -150800.89,
    "best_trade_pnl": 257017.48,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": -1200477.22,
    "trade_count": 24,
    "win_rate": 0.375,
    "avg_hold_days": 14.0,
    "signal_total": 48,
    "executed_signals": 48,
    "executed_buy_signals": 24,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2018-08",
      -835112.1479274749
    ],
    "best_month": [
      "2019-04",
      186094.47106308956
    ],
    "worst_trade_pnl": -835112.15,
    "best_trade_pnl": 186094.47,
    "top_unexecuted_reasons": {}
  }
]
```

## 002371.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": 1543211.16,
    "trade_count": 19,
    "win_rate": 0.6316,
    "avg_hold_days": 16.0,
    "signal_total": 39,
    "executed_signals": 39,
    "executed_buy_signals": 20,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2020-07",
      -205402.35646209726
    ],
    "best_month": [
      "2020-05",
      581191.2794219973
    ],
    "worst_trade_pnl": -205402.36,
    "best_trade_pnl": 581191.28,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": -475812.65,
    "trade_count": 2,
    "win_rate": 0.5,
    "avg_hold_days": 26.0,
    "signal_total": 5,
    "executed_signals": 4,
    "executed_buy_signals": 2,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-04",
      -570118.6595460894
    ],
    "best_month": [
      "2017-09",
      94306.01342582703
    ],
    "worst_trade_pnl": -570118.66,
    "best_trade_pnl": 94306.01,
    "top_unexecuted_reasons": {
      "可买数量不足: 无法买入100股": 1
    }
  }
]
```

## 300085.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": 986365.24,
    "trade_count": 18,
    "win_rate": 0.6667,
    "avg_hold_days": 14.0,
    "signal_total": 36,
    "executed_signals": 36,
    "executed_buy_signals": 18,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2018-08",
      -271850.3401743891
    ],
    "best_month": [
      "2019-03",
      756227.8251291276
    ],
    "worst_trade_pnl": -251489.34,
    "best_trade_pnl": 756227.83,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": -995433.84,
    "trade_count": 13,
    "win_rate": 0.3077,
    "avg_hold_days": 43.0,
    "signal_total": 26,
    "executed_signals": 26,
    "executed_buy_signals": 13,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-08",
      -397015.6179687022
    ],
    "best_month": [
      "2019-02",
      297910.99263877864
    ],
    "worst_trade_pnl": -397015.62,
    "best_trade_pnl": 297910.99,
    "top_unexecuted_reasons": {}
  }
]
```

## 600809.SH
```json
[
  {
    "model": "alpha158",
    "total_return": 1861482.9,
    "trade_count": 13,
    "win_rate": 0.6154,
    "avg_hold_days": 24.0,
    "signal_total": 26,
    "executed_signals": 26,
    "executed_buy_signals": 13,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2020-02",
      -141325.33289794927
    ],
    "best_month": [
      "2017-04",
      995170.5646820068
    ],
    "worst_trade_pnl": -141325.33,
    "best_trade_pnl": 995170.56,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": -21715.67,
    "trade_count": 1,
    "win_rate": 0.0,
    "avg_hold_days": 2.0,
    "signal_total": 4,
    "executed_signals": 2,
    "executed_buy_signals": 1,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-04",
      -21715.670219421387
    ],
    "best_month": [
      "2017-04",
      -21715.670219421387
    ],
    "worst_trade_pnl": -21715.67,
    "best_trade_pnl": -21715.67,
    "top_unexecuted_reasons": {
      "可买数量不足: 无法买入100股": 2
    }
  }
]
```

## 600760.SH
```json
[
  {
    "model": "alpha158",
    "total_return": 1383529.77,
    "trade_count": 19,
    "win_rate": 0.5789,
    "avg_hold_days": 12.0,
    "signal_total": 38,
    "executed_signals": 38,
    "executed_buy_signals": 19,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2018-02",
      -86643.20983972552
    ],
    "best_month": [
      "2020-07",
      683113.1737014772
    ],
    "worst_trade_pnl": -86643.21,
    "best_trade_pnl": 683113.17,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": -318825.01,
    "trade_count": 8,
    "win_rate": 0.5,
    "avg_hold_days": 15.0,
    "signal_total": 16,
    "executed_signals": 16,
    "executed_buy_signals": 8,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2020-03",
      -329826.7687654495
    ],
    "best_month": [
      "2019-12",
      79755.46133508673
    ],
    "worst_trade_pnl": -329826.77,
    "best_trade_pnl": 79755.46,
    "top_unexecuted_reasons": {}
  }
]
```

## 002517.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": 302645.84,
    "trade_count": 11,
    "win_rate": 0.5455,
    "avg_hold_days": 24.0,
    "signal_total": 23,
    "executed_signals": 22,
    "executed_buy_signals": 11,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2019-05",
      -368939.6580437063
    ],
    "best_month": [
      "2017-11",
      761216.0298371315
    ],
    "worst_trade_pnl": -368939.66,
    "best_trade_pnl": 761216.03,
    "top_unexecuted_reasons": {
      "可买数量不足: 无法买入100股": 1
    }
  },
  {
    "model": "alpha360",
    "total_return": -1338990.79,
    "trade_count": 24,
    "win_rate": 0.4583,
    "avg_hold_days": 15.0,
    "signal_total": 48,
    "executed_signals": 48,
    "executed_buy_signals": 24,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-10",
      -752203.8015977859
    ],
    "best_month": [
      "2020-07",
      252946.6881075143
    ],
    "worst_trade_pnl": -752203.8,
    "best_trade_pnl": 221922.88,
    "top_unexecuted_reasons": {}
  }
]
```

## 000066.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": 94808.07,
    "trade_count": 20,
    "win_rate": 0.55,
    "avg_hold_days": 13.0,
    "signal_total": 40,
    "executed_signals": 40,
    "executed_buy_signals": 20,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2019-04",
      -415083.15671138745
    ],
    "best_month": [
      "2020-07",
      416657.1190505028
    ],
    "worst_trade_pnl": -341277.96,
    "best_trade_pnl": 416657.12,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": -1487706.84,
    "trade_count": 22,
    "win_rate": 0.3182,
    "avg_hold_days": 17.0,
    "signal_total": 45,
    "executed_signals": 44,
    "executed_buy_signals": 22,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-12",
      -583428.7242157459
    ],
    "best_month": [
      "2019-05",
      376795.26043729763
    ],
    "worst_trade_pnl": -583428.72,
    "best_trade_pnl": 376795.26,
    "top_unexecuted_reasons": {
      "可买数量不足: 无法买入100股": 1
    }
  }
]
```

## 300383.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": 511948.97,
    "trade_count": 13,
    "win_rate": 0.6923,
    "avg_hold_days": 15.0,
    "signal_total": 26,
    "executed_signals": 26,
    "executed_buy_signals": 13,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2018-10",
      -102668.6351202965
    ],
    "best_month": [
      "2019-02",
      203503.5124282837
    ],
    "worst_trade_pnl": -102668.64,
    "best_trade_pnl": 203503.51,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": -991103.2,
    "trade_count": 12,
    "win_rate": 0.6667,
    "avg_hold_days": 20.0,
    "signal_total": 25,
    "executed_signals": 24,
    "executed_buy_signals": 12,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-05",
      -1563192.4617084502
    ],
    "best_month": [
      "2019-07",
      286676.66111297626
    ],
    "worst_trade_pnl": -1563192.46,
    "best_trade_pnl": 286676.66,
    "top_unexecuted_reasons": {
      "可买数量不足: 无法买入100股": 1
    }
  }
]
```

## 300458.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": -2593625.43,
    "trade_count": 18,
    "win_rate": 0.5556,
    "avg_hold_days": 14.0,
    "signal_total": 36,
    "executed_signals": 36,
    "executed_buy_signals": 18,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-05",
      -2305974.725715637
    ],
    "best_month": [
      "2017-07",
      200599.12008152017
    ],
    "worst_trade_pnl": -2305974.73,
    "best_trade_pnl": 200599.12,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": 233825.23,
    "trade_count": 4,
    "win_rate": 0.75,
    "avg_hold_days": 12.0,
    "signal_total": 8,
    "executed_signals": 8,
    "executed_buy_signals": 4,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-04",
      -113875.38586997986
    ],
    "best_month": [
      "2019-03",
      203051.47057733545
    ],
    "worst_trade_pnl": -113875.39,
    "best_trade_pnl": 203051.47,
    "top_unexecuted_reasons": {}
  }
]
```

## 002050.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": -410524.57,
    "trade_count": 13,
    "win_rate": 0.5385,
    "avg_hold_days": 15.0,
    "signal_total": 26,
    "executed_signals": 26,
    "executed_buy_signals": 13,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2019-06",
      -429532.9106609344
    ],
    "best_month": [
      "2017-10",
      179867.1561946869
    ],
    "worst_trade_pnl": -429532.91,
    "best_trade_pnl": 179867.16,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": 1506506.39,
    "trade_count": 18,
    "win_rate": 0.8889,
    "avg_hold_days": 16.0,
    "signal_total": 36,
    "executed_signals": 36,
    "executed_buy_signals": 18,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2018-08",
      -47977.61406311998
    ],
    "best_month": [
      "2017-12",
      317362.4567157745
    ],
    "worst_trade_pnl": -47977.61,
    "best_trade_pnl": 240981.59,
    "top_unexecuted_reasons": {}
  }
]
```

## 002466.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": -1255274.79,
    "trade_count": 18,
    "win_rate": 0.4444,
    "avg_hold_days": 15.0,
    "signal_total": 36,
    "executed_signals": 36,
    "executed_buy_signals": 18,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2018-01",
      -418289.58327770233
    ],
    "best_month": [
      "2018-11",
      174850.55722379684
    ],
    "worst_trade_pnl": -418289.58,
    "best_trade_pnl": 174850.56,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": 574312.63,
    "trade_count": 2,
    "win_rate": 1.0,
    "avg_hold_days": 12.0,
    "signal_total": 4,
    "executed_signals": 4,
    "executed_buy_signals": 2,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2019-09",
      247061.5463371277
    ],
    "best_month": [
      "2017-04",
      327251.0834142687
    ],
    "worst_trade_pnl": 247061.55,
    "best_trade_pnl": 327251.08,
    "top_unexecuted_reasons": {}
  }
]
```

## 600988.SH
```json
[
  {
    "model": "alpha158",
    "total_return": -1953732.63,
    "trade_count": 12,
    "win_rate": 0.3333,
    "avg_hold_days": 13.0,
    "signal_total": 25,
    "executed_signals": 25,
    "executed_buy_signals": 13,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-10",
      -1585185.7377163412
    ],
    "best_month": [
      "2019-09",
      48002.170729780104
    ],
    "worst_trade_pnl": -1638984.98,
    "best_trade_pnl": 53799.25,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": -140058.27,
    "trade_count": 26,
    "win_rate": 0.5385,
    "avg_hold_days": 17.0,
    "signal_total": 52,
    "executed_signals": 52,
    "executed_buy_signals": 26,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2019-10",
      -346326.9314230918
    ],
    "best_month": [
      "2018-03",
      323523.8378198147
    ],
    "worst_trade_pnl": -338063.51,
    "best_trade_pnl": 323523.84,
    "top_unexecuted_reasons": {}
  }
]
```

## 002896.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": -1065916.8,
    "trade_count": 18,
    "win_rate": 0.3333,
    "avg_hold_days": 16.0,
    "signal_total": 36,
    "executed_signals": 36,
    "executed_buy_signals": 18,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2019-05",
      -515210.2363283157
    ],
    "best_month": [
      "2020-07",
      262101.4694360732
    ],
    "worst_trade_pnl": -515210.24,
    "best_trade_pnl": 262101.47,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": 741538.33,
    "trade_count": 5,
    "win_rate": 0.8,
    "avg_hold_days": 16.0,
    "signal_total": 10,
    "executed_signals": 10,
    "executed_buy_signals": 5,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2019-10",
      -33811.1276584625
    ],
    "best_month": [
      "2020-07",
      617407.0473489761
    ],
    "worst_trade_pnl": -33811.13,
    "best_trade_pnl": 617407.05,
    "top_unexecuted_reasons": {}
  }
]
```

## 600584.SH
```json
[
  {
    "model": "alpha158",
    "total_return": -216702.31,
    "trade_count": 20,
    "win_rate": 0.6,
    "avg_hold_days": 13.0,
    "signal_total": 40,
    "executed_signals": 40,
    "executed_buy_signals": 20,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2018-11",
      -512791.0766327381
    ],
    "best_month": [
      "2019-09",
      178129.79550056462
    ],
    "worst_trade_pnl": -512791.08,
    "best_trade_pnl": 178129.8,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": 1497824.43,
    "trade_count": 15,
    "win_rate": 0.6,
    "avg_hold_days": 23.0,
    "signal_total": 30,
    "executed_signals": 30,
    "executed_buy_signals": 15,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-05",
      -297488.9760994911
    ],
    "best_month": [
      "2017-03",
      757271.6845899578
    ],
    "worst_trade_pnl": -458312.62,
    "best_trade_pnl": 757271.68,
    "top_unexecuted_reasons": {}
  }
]
```

## 002465.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": -751685.92,
    "trade_count": 12,
    "win_rate": 0.4167,
    "avg_hold_days": 13.0,
    "signal_total": 24,
    "executed_signals": 24,
    "executed_buy_signals": 12,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2018-06",
      -492653.2035140991
    ],
    "best_month": [
      "2017-09",
      44148.77447037678
    ],
    "worst_trade_pnl": -492653.2,
    "best_trade_pnl": 70877.69,
    "top_unexecuted_reasons": {}
  },
  {
    "model": "alpha360",
    "total_return": 926984.72,
    "trade_count": 25,
    "win_rate": 0.6,
    "avg_hold_days": 14.0,
    "signal_total": 50,
    "executed_signals": 50,
    "executed_buy_signals": 25,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2018-06",
      -177552.9325930595
    ],
    "best_month": [
      "2018-09",
      174907.002570963
    ],
    "worst_trade_pnl": -177552.93,
    "best_trade_pnl": 174907.0,
    "top_unexecuted_reasons": {}
  }
]
```

## 300207.SZ
```json
[
  {
    "model": "alpha158",
    "total_return": -865076.34,
    "trade_count": 20,
    "win_rate": 0.4,
    "avg_hold_days": 14.0,
    "signal_total": 42,
    "executed_signals": 41,
    "executed_buy_signals": 21,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2020-04",
      -754000.9078551293
    ],
    "best_month": [
      "2019-02",
      151176.4476168633
    ],
    "worst_trade_pnl": -754000.91,
    "best_trade_pnl": 151176.45,
    "top_unexecuted_reasons": {
      "可买数量不足: 无法买入100股": 1
    }
  },
  {
    "model": "alpha360",
    "total_return": 789523.39,
    "trade_count": 25,
    "win_rate": 0.6,
    "avg_hold_days": 13.0,
    "signal_total": 50,
    "executed_signals": 50,
    "executed_buy_signals": 25,
    "negative_executed_buy_signals": 0,
    "avg_executed_buy_predicted_return": null,
    "worst_month": [
      "2017-08",
      -199198.29869656544
    ],
    "best_month": [
      "2020-06",
      340144.45398521423
    ],
    "worst_trade_pnl": -199198.3,
    "best_trade_pnl": 294577.06,
    "top_unexecuted_reasons": {}
  }
]
```

## High-level findings

- negative_executed_buy_signals > 0 means the strategy still bought names whose ranking score / predicted return was negative, i.e. they were only relatively top-ranked inside a weak cross-section.
- worst_month pinpoints where each stock did the most damage; if one model has the same stock but much worse worst_month, that is usually where the net delta came from.
- executed_signals is generally high, so this replay focuses on ranking quality and stock selection, not rejection bottlenecks.