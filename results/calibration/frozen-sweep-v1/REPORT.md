# Threshold feasibility on frozen synthetic scores

Under the empirical 5% limit, no deterministic threshold recovers native 0 dB detection at nominal gain for this archived GAP checkpoint. At most 15 false alarms were allowed among 300 noise frames at each gain.

| Gain drift | Native detections at −10 / 0 / +10 dB | INT8 detections at −10 / 0 / +10 dB |
|---:|---:|---:|
| −1 dB | 14 / 98 / 100 | 2 / 89 / 100 |
| 0 dB | 14 / 98 / 100 | 0 / 69 / 100 |
| +1 dB | 14 / 98 / 100 | 10 / 98 / 100 |

Each count is out of 100 signal frames. At nominal gain, the archived native threshold transferred to INT8 gives 60/300 false alarms. The independently INT8-calibrated threshold gives 0/300. Its 69/100 detections at 0 dB already equal the best attainable count at that gain. Native FP32 can reach 98/100 in the sweep; its originally deployed threshold gave 97/100.

The sweep includes every distinct score in the combined sample for each gain, an explicit reject-all endpoint and an accept-all endpoint. Equal scores enter together. Among equal best detection counts, select fewer false alarms and then the higher threshold. SNR-specific thresholds in the table are illustrative hindsight choices. The envelope diagnoses what these test scores allow; selection and evaluation on the same scores cannot establish deployment performance. [Counts and selected thresholds](best_points.csv) · [All operating points](all_thresholds.csv) · [Figure](feasibility.png)
