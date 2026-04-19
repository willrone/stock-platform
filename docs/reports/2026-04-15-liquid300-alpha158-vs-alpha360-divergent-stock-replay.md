# Liquid300 alpha158 vs alpha360 divergent stock replay

Tasks:
- alpha158: `54a3681f-b995-470d-b5a8-e67f70d3b972`
- alpha360: `0e2e03b4-a830-452c-822a-f02154a9ac48`

## Top 12 divergent stocks by absolute PnL delta

```
stock_code  total_return_158  total_return_360  delta_158_minus_360  trade_count_158  trade_count_360
 300458.SZ     -2.593625e+06      2.338252e+05        -2.827451e+06               18              4.0
 002611.SZ      1.029023e+06     -1.200477e+06         2.229500e+06               20             24.0
 002371.SZ      1.543211e+06     -4.758126e+05         2.019024e+06               19              2.0
 300085.SZ      9.863652e+05     -9.954338e+05         1.981799e+06               18             13.0
 002050.SZ     -4.105246e+05      1.506506e+06        -1.917031e+06               13             18.0
 600809.SH      1.861483e+06     -2.171567e+04         1.883199e+06               13              1.0
 002466.SZ     -1.255275e+06      5.743126e+05        -1.829587e+06               18              2.0
 600988.SH     -1.953733e+06     -1.400583e+05        -1.813674e+06               12             26.0
 002896.SZ     -1.065917e+06      7.415383e+05        -1.807455e+06               18              5.0
 600584.SH     -2.167023e+05      1.497824e+06        -1.714527e+06               20             15.0
 600760.SH      1.383530e+06     -3.188250e+05         1.702355e+06               19              8.0
 002465.SZ     -7.516859e+05      9.269847e+05        -1.678671e+06               12             25.0
```

## Stocks where alpha158 clearly beats alpha360

### 002611.SZ
- PnL delta (158 - 360): 2,229,500.25
- alpha158: return=1,029,023.03, trades=20, signals=40, executed=40, avg_score=None, hold_days=14.0
- alpha360: return=-1,200,477.22, trades=24, signals=48, executed=48, avg_score=None, hold_days=14.0
- alpha158 trade window: 2017-02-07 -> 2020-06-08; alpha360 trade window: 2017-04-24 -> 2020-06-22
- alpha158 best scores: []
- alpha360 worst scores: []
- alpha158 trade preview: [{'date': '2017-02-07', 'action': 'BUY', 'qty': 100, 'price': 15.0, 'pnl': 0.0}, {'date': '2017-04-10', 'action': 'SELL', 'qty': 100, 'price': 17.5, 'pnl': 245.0}, {'date': '2020-05-19', 'action': 'BUY', 'qty': 432800, 'price': 4.2, 'pnl': 0.0}, {'date': '2020-06-08', 'action': 'SELL', 'qty': 432800, 'price': 4.17, 'pnl': -15691.05}]
- alpha360 trade preview: [{'date': '2017-04-24', 'action': 'BUY', 'qty': 66700, 'price': 15.3, 'pnl': 0.0}, {'date': '2017-06-21', 'action': 'SELL', 'qty': 66700, 'price': 13.98, 'pnl': -89442.74}, {'date': '2020-06-19', 'action': 'BUY', 'qty': 282100, 'price': 4.51, 'pnl': 0.0}, {'date': '2020-06-22', 'action': 'SELL', 'qty': 282100, 'price': 4.58, 'pnl': 17808.89}]

### 002371.SZ
- PnL delta (158 - 360): 2,019,023.81
- alpha158: return=1,543,211.16, trades=19, signals=39, executed=39, avg_score=None, hold_days=16.0
- alpha360: return=-475,812.65, trades=2, signals=5, executed=4, avg_score=None, hold_days=26.0
- alpha158 trade window: 2017-07-05 -> 2020-07-29; alpha360 trade window: 2017-03-28 -> 2017-09-01
- alpha158 best scores: []
- alpha360 worst scores: []
- alpha158 trade preview: [{'date': '2017-07-05', 'action': 'BUY', 'qty': 108400, 'price': 24.48, 'pnl': 0.0}, {'date': '2017-07-10', 'action': 'SELL', 'qty': 108400, 'price': 24.84, 'pnl': 34985.08}, {'date': '2020-07-20', 'action': 'SELL', 'qty': 12700, 'price': 182.28, 'pnl': -205402.36}, {'date': '2020-07-29', 'action': 'BUY', 'qty': 8000, 'price': 189.2, 'pnl': 0.0}]
- alpha360 trade preview: [{'date': '2017-03-28', 'action': 'BUY', 'qty': 116600, 'price': 31.2, 'pnl': 0.0}, {'date': '2017-04-19', 'action': 'SELL', 'qty': 116600, 'price': 26.35, 'pnl': -570118.66}, {'date': '2017-08-01', 'action': 'BUY', 'qty': 70800, 'price': 23.96, 'pnl': 0.0}, {'date': '2017-09-01', 'action': 'SELL', 'qty': 70800, 'price': 25.33, 'pnl': 94306.01}]

### 300085.SZ
- PnL delta (158 - 360): 1,981,799.08
- alpha158: return=986,365.24, trades=18, signals=36, executed=36, avg_score=None, hold_days=14.0
- alpha360: return=-995,433.84, trades=13, signals=26, executed=26, avg_score=None, hold_days=43.0
- alpha158 trade window: 2017-02-07 -> 2020-07-20; alpha360 trade window: 2017-04-25 -> 2020-06-30
- alpha158 best scores: []
- alpha360 worst scores: []
- alpha158 trade preview: [{'date': '2017-02-07', 'action': 'BUY', 'qty': 1600, 'price': 19.1, 'pnl': 0.0}, {'date': '2017-04-20', 'action': 'SELL', 'qty': 1600, 'price': 18.4, 'pnl': -1164.16}, {'date': '2020-07-09', 'action': 'BUY', 'qty': 51000, 'price': 24.78, 'pnl': 0.0}, {'date': '2020-07-20', 'action': 'SELL', 'qty': 51000, 'price': 25.93, 'pnl': 56666.34}]
- alpha360 trade preview: [{'date': '2017-04-25', 'action': 'BUY', 'qty': 66700, 'price': 18.55, 'pnl': 0.0}, {'date': '2017-06-12', 'action': 'SELL', 'qty': 66700, 'price': 16.86, 'pnl': -114409.75}, {'date': '2020-06-11', 'action': 'BUY', 'qty': 151300, 'price': 14.36, 'pnl': 0.0}, {'date': '2020-06-30', 'action': 'SELL', 'qty': 151300, 'price': 16.02, 'pnl': 247522.38}]

### 600809.SH
- PnL delta (158 - 360): 1,883,198.57
- alpha158: return=1,861,482.90, trades=13, signals=26, executed=26, avg_score=None, hold_days=24.0
- alpha360: return=-21,715.67, trades=1, signals=4, executed=2, avg_score=None, hold_days=2.0
- alpha158 trade window: 2017-02-07 -> 2020-06-10; alpha360 trade window: 2017-04-25 -> 2017-04-27
- alpha158 best scores: []
- alpha360 worst scores: []
- alpha158 trade preview: [{'date': '2017-02-07', 'action': 'BUY', 'qty': 168000, 'price': 24.94, 'pnl': 0.0}, {'date': '2017-04-13', 'action': 'SELL', 'qty': 168000, 'price': 30.91, 'pnl': 995170.56}, {'date': '2020-04-09', 'action': 'BUY', 'qty': 9200, 'price': 93.8, 'pnl': 0.0}, {'date': '2020-06-10', 'action': 'SELL', 'qty': 9200, 'price': 135.14, 'pnl': 378463.03}]
- alpha360 trade preview: [{'date': '2017-04-25', 'action': 'BUY', 'qty': 75500, 'price': 31.99, 'pnl': 0.0}, {'date': '2017-04-27', 'action': 'SELL', 'qty': 75500, 'price': 31.75, 'pnl': -21715.67}]

### 600760.SH
- PnL delta (158 - 360): 1,702,354.78
- alpha158: return=1,383,529.77, trades=19, signals=38, executed=38, avg_score=None, hold_days=12.0
- alpha360: return=-318,825.01, trades=8, signals=16, executed=16, avg_score=None, hold_days=15.0
- alpha158 trade window: 2017-06-19 -> 2020-07-27; alpha360 trade window: 2017-07-10 -> 2020-03-10
- alpha158 best scores: []
- alpha360 worst scores: []
- alpha158 trade preview: [{'date': '2017-06-19', 'action': 'BUY', 'qty': 22600, 'price': 36.23, 'pnl': 0.0}, {'date': '2017-06-20', 'action': 'SELL', 'qty': 22600, 'price': 33.93, 'pnl': -53130.21}, {'date': '2020-07-09', 'action': 'BUY', 'qty': 47800, 'price': 51.56, 'pnl': 0.0}, {'date': '2020-07-27', 'action': 'SELL', 'qty': 47800, 'price': 65.95, 'pnl': 683113.17}]
- alpha360 trade preview: [{'date': '2017-07-10', 'action': 'BUY', 'qty': 34700, 'price': 33.83, 'pnl': 0.0}, {'date': '2017-07-13', 'action': 'SELL', 'qty': 34700, 'price': 32.01, 'pnl': -64820.24}, {'date': '2020-01-09', 'action': 'BUY', 'qty': 78300, 'price': 32.4, 'pnl': 0.0}, {'date': '2020-03-10', 'action': 'SELL', 'qty': 78300, 'price': 28.23, 'pnl': -329826.77}]

## Stocks where alpha360 clearly beats alpha158

### 300458.SZ
- PnL delta (158 - 360): -2,827,450.66
- alpha158: return=-2,593,625.43, trades=18, signals=36, executed=36, avg_score=None, hold_days=14.0
- alpha360: return=233,825.23, trades=4, signals=8, executed=8, avg_score=None, hold_days=12.0
- alpha158 trade window: 2017-02-07 -> 2020-05-11; alpha360 trade window: 2017-03-30 -> 2019-08-27
- alpha158 worst scores: []
- alpha360 best scores: []
- alpha158 trade preview: [{'date': '2017-02-07', 'action': 'BUY', 'qty': 143200, 'price': 71.47, 'pnl': 0.0}, {'date': '2017-05-16', 'action': 'SELL', 'qty': 143200, 'price': 55.45, 'pnl': -2305974.73}, {'date': '2020-04-28', 'action': 'BUY', 'qty': 39300, 'price': 26.77, 'pnl': 0.0}, {'date': '2020-05-11', 'action': 'SELL', 'qty': 39300, 'price': 29.44, 'pnl': 103195.51}]
- alpha360 trade preview: [{'date': '2017-03-30', 'action': 'BUY', 'qty': 21500, 'price': 69.55, 'pnl': 0.0}, {'date': '2017-04-11', 'action': 'SELL', 'qty': 21500, 'price': 64.35, 'pnl': -113875.39}, {'date': '2019-08-20', 'action': 'BUY', 'qty': 75000, 'price': 22.33, 'pnl': 0.0}, {'date': '2019-08-27', 'action': 'SELL', 'qty': 75000, 'price': 23.7, 'pnl': 100083.81}]

### 002050.SZ
- PnL delta (158 - 360): -1,917,030.96
- alpha158: return=-410,524.57, trades=13, signals=26, executed=26, avg_score=None, hold_days=15.0
- alpha360: return=1,506,506.39, trades=18, signals=36, executed=36, avg_score=None, hold_days=16.0
- alpha158 trade window: 2017-08-18 -> 2020-07-17; alpha360 trade window: 2017-02-07 -> 2019-11-14
- alpha158 worst scores: []
- alpha360 best scores: []
- alpha158 trade preview: [{'date': '2017-08-18', 'action': 'BUY', 'qty': 114600, 'price': 16.0, 'pnl': 0.0}, {'date': '2017-08-30', 'action': 'SELL', 'qty': 114600, 'price': 16.47, 'pnl': 51030.73}, {'date': '2020-07-09', 'action': 'BUY', 'qty': 63600, 'price': 24.81, 'pnl': 0.0}, {'date': '2020-07-17', 'action': 'SELL', 'qty': 63600, 'price': 22.95, 'pnl': -120485.35}]
- alpha360 trade preview: [{'date': '2017-02-07', 'action': 'BUY', 'qty': 7200, 'price': 10.51, 'pnl': 0.0}, {'date': '2017-04-21', 'action': 'SELL', 'qty': 7200, 'price': 11.89, 'pnl': 9807.59}, {'date': '2019-11-12', 'action': 'BUY', 'qty': 129600, 'price': 14.24, 'pnl': 0.0}, {'date': '2019-11-14', 'action': 'SELL', 'qty': 129600, 'price': 14.79, 'pnl': 68404.85}]

### 002466.SZ
- PnL delta (158 - 360): -1,829,587.42
- alpha158: return=-1,255,274.79, trades=18, signals=36, executed=36, avg_score=None, hold_days=15.0
- alpha360: return=574,312.63, trades=2, signals=4, executed=4, avg_score=None, hold_days=12.0
- alpha158 trade window: 2017-09-11 -> 2020-07-21; alpha360 trade window: 2017-03-28 -> 2019-09-25
- alpha158 worst scores: []
- alpha360 best scores: []
- alpha158 trade preview: [{'date': '2017-09-11', 'action': 'BUY', 'qty': 21300, 'price': 81.98, 'pnl': 0.0}, {'date': '2017-09-12', 'action': 'SELL', 'qty': 21300, 'price': 76.38, 'pnl': -121720.47}, {'date': '2020-07-08', 'action': 'BUY', 'qty': 69000, 'price': 26.66, 'pnl': 0.0}, {'date': '2020-07-21', 'action': 'SELL', 'qty': 69000, 'price': 26.74, 'pnl': 2752.4}]
- alpha360 trade preview: [{'date': '2017-03-28', 'action': 'BUY', 'qty': 46100, 'price': 40.35, 'pnl': 0.0}, {'date': '2017-04-05', 'action': 'SELL', 'qty': 46100, 'price': 47.52, 'pnl': 327251.08}, {'date': '2019-09-09', 'action': 'BUY', 'qty': 54700, 'price': 24.33, 'pnl': 0.0}, {'date': '2019-09-25', 'action': 'SELL', 'qty': 54700, 'price': 28.89, 'pnl': 247061.55}]

### 600988.SH
- PnL delta (158 - 360): -1,813,674.35
- alpha158: return=-1,953,732.63, trades=12, signals=25, executed=25, avg_score=None, hold_days=13.0
- alpha360: return=-140,058.27, trades=26, signals=52, executed=52, avg_score=None, hold_days=17.0
- alpha158 trade window: 2017-05-02 -> 2020-07-27; alpha360 trade window: 2017-03-15 -> 2020-06-11
- alpha158 worst scores: []
- alpha360 best scores: []
- alpha158 trade preview: [{'date': '2017-05-02', 'action': 'BUY', 'qty': 99300, 'price': 13.36, 'pnl': 0.0}, {'date': '2017-05-26', 'action': 'SELL', 'qty': 99300, 'price': 11.72, 'pnl': -164597.63}, {'date': '2019-11-13', 'action': 'SELL', 'qty': 202500, 'price': 4.54, 'pnl': -70229.06}, {'date': '2020-07-27', 'action': 'BUY', 'qty': 93300, 'price': 17.84, 'pnl': 0.0}]
- alpha360 trade preview: [{'date': '2017-03-15', 'action': 'BUY', 'qty': 140900, 'price': 15.13, 'pnl': 0.0}, {'date': '2017-03-29', 'action': 'SELL', 'qty': 140900, 'price': 14.88, 'pnl': -38369.89}, {'date': '2020-06-03', 'action': 'BUY', 'qty': 146600, 'price': 9.84, 'pnl': 0.0}, {'date': '2020-06-11', 'action': 'SELL', 'qty': 146600, 'price': 9.98, 'pnl': 18329.31}]

### 002896.SZ
- PnL delta (158 - 360): -1,807,455.13
- alpha158: return=-1,065,916.80, trades=18, signals=36, executed=36, avg_score=None, hold_days=16.0
- alpha360: return=741,538.33, trades=5, signals=10, executed=10, avg_score=None, hold_days=16.0
- alpha158 trade window: 2017-10-12 -> 2020-07-27; alpha360 trade window: 2019-09-27 -> 2020-07-15
- alpha158 worst scores: []
- alpha360 best scores: []
- alpha158 trade preview: [{'date': '2017-10-12', 'action': 'BUY', 'qty': 55300, 'price': 49.83, 'pnl': 0.0}, {'date': '2017-10-18', 'action': 'SELL', 'qty': 55300, 'price': 45.81, 'pnl': -226105.96}, {'date': '2020-07-07', 'action': 'BUY', 'qty': 74700, 'price': 23.97, 'pnl': 0.0}, {'date': '2020-07-27', 'action': 'SELL', 'qty': 74700, 'price': 27.52, 'pnl': 262101.47}]
- alpha360 trade preview: [{'date': '2019-09-27', 'action': 'BUY', 'qty': 32400, 'price': 23.38, 'pnl': 0.0}, {'date': '2019-10-18', 'action': 'SELL', 'qty': 32400, 'price': 22.37, 'pnl': -33811.13}, {'date': '2020-06-09', 'action': 'BUY', 'qty': 103800, 'price': 21.98, 'pnl': 0.0}, {'date': '2020-07-15', 'action': 'SELL', 'qty': 103800, 'price': 27.97, 'pnl': 617407.05}]

### 600584.SH
- PnL delta (158 - 360): -1,714,526.74
- alpha158: return=-216,702.31, trades=20, signals=40, executed=40, avg_score=None, hold_days=13.0
- alpha360: return=1,497,824.43, trades=15, signals=30, executed=30, avg_score=None, hold_days=23.0
- alpha158 trade window: 2017-04-17 -> 2020-07-17; alpha360 trade window: 2017-02-07 -> 2020-01-16
- alpha158 worst scores: []
- alpha360 best scores: []
- alpha158 trade preview: [{'date': '2017-04-17', 'action': 'BUY', 'qty': 191700, 'price': 17.7, 'pnl': 0.0}, {'date': '2017-04-19', 'action': 'SELL', 'qty': 191700, 'price': 17.86, 'pnl': 25536.33}, {'date': '2020-07-15', 'action': 'BUY', 'qty': 32600, 'price': 46.05, 'pnl': 0.0}, {'date': '2020-07-17', 'action': 'SELL', 'qty': 32600, 'price': 39.17, 'pnl': -226203.45}]
- alpha360 trade preview: [{'date': '2017-02-07', 'action': 'BUY', 'qty': 586200, 'price': 17.46, 'pnl': 0.0}, {'date': '2017-03-10', 'action': 'SELL', 'qty': 586200, 'price': 18.78, 'pnl': 757271.68}, {'date': '2020-01-10', 'action': 'BUY', 'qty': 42100, 'price': 22.65, 'pnl': 0.0}, {'date': '2020-01-16', 'action': 'SELL', 'qty': 42100, 'price': 24.1, 'pnl': 59523.12}]

### 002465.SZ
- PnL delta (158 - 360): -1,678,670.63
- alpha158: return=-751,685.92, trades=12, signals=24, executed=24, avg_score=None, hold_days=13.0
- alpha360: return=926,984.72, trades=25, signals=50, executed=50, avg_score=None, hold_days=14.0
- alpha158 trade window: 2017-05-05 -> 2020-07-10; alpha360 trade window: 2017-03-31 -> 2020-07-29
- alpha158 worst scores: []
- alpha360 best scores: []
- alpha158 trade preview: [{'date': '2017-05-05', 'action': 'BUY', 'qty': 272200, 'price': 10.96, 'pnl': 0.0}, {'date': '2017-05-19', 'action': 'SELL', 'qty': 272200, 'price': 10.76, 'pnl': -58833.26}, {'date': '2020-07-08', 'action': 'BUY', 'qty': 65900, 'price': 14.3, 'pnl': 0.0}, {'date': '2020-07-10', 'action': 'SELL', 'qty': 65900, 'price': 13.93, 'pnl': -25759.97}]
- alpha360 trade preview: [{'date': '2017-03-31', 'action': 'BUY', 'qty': 290500, 'price': 11.74, 'pnl': 0.0}, {'date': '2017-04-06', 'action': 'SELL', 'qty': 290500, 'price': 12.0, 'pnl': 70301.07}, {'date': '2020-06-23', 'action': 'BUY', 'qty': 81700, 'price': 12.76, 'pnl': 0.0}, {'date': '2020-07-29', 'action': 'SELL', 'qty': 81700, 'price': 14.32, 'pnl': 125697.04}]

## Synthesis

- alpha158 wins on several names because it either trades them much more effectively (e.g. 002371.SZ, 600809.SH) or avoids turning them into deep losers (e.g. 002611.SZ, 300085.SZ).
- alpha360 wins on another cluster because alpha158 turns them into repeated large losers while alpha360 either captures them positively or at least loses much less (e.g. 300458.SZ, 002050.SZ, 600584.SH, 000625.SZ).
- In the divergent names, the main difference is not signal rejection. Both tasks execute nearly all signals. The difference is the ranking path and resulting trade sequence.
- The replay strongly suggests the next model iteration should focus on ranking/label quality in the cross-section rather than more execution tweaks.