# Human Activity Recognition with a Hidden Markov Model

**Author:** Amaliza Shalom · Formative 2 – Hidden Markov Models

A Gaussian Hidden Markov Model, written from scratch in NumPy (Baum-Welch training, Viterbi decoding), that recognises four activities from phone accelerometer and gyroscope data: standing, walking, jumping and still.

| | |
|---|---|
| Device | iPhone 13, Sensor Logger 1.68 |
| Sampling rate | Set to 10 ms, measured 99.45 Hz, no resampling needed |
| Data | 60 recordings, 15 per activity |
| Training data | 48 recordings |
| Unseen test data | 12 recordings |
| Test accuracy | 100 % (61/61 windows, 12/12 recordings) |

## Repository structure

```
Data_clean/
  train/<activity>/        cleaned training recordings
  test/<activity>/         cleaned unseen test recordings
  manifest.csv             details of each file
  excluded.csv             files left out (none)
  trim_rule_impact.csv     effect of the trimming rule
data/<activity>/           raw Sensor Logger recordings
figures/                   all graphs
report/                    final report (PDF and Word)
tests/test_hmm.py          tests for Viterbi and Baum-Welch
videos/                    the four activity videos
HMM_Activity_Recognition.ipynb   full notebook with outputs
prepare_data.py            raw data to cleaned CSVs
requirements.txt
```

CSV columns: `timestamp`, `ax`, `ay`, `az` (m/s², gravity removed), `gx`, `gy`, `gz` (rad/s).

## Videos

| Activity | Video |
|---|---|
| Standing | https://drive.google.com/file/d/1Tq2zAthdj9Cif8v5GVrvESlW4DmJisJH/view |
| Walking | https://drive.google.com/file/d/1KrsBVEBJWtxzm3FCOFKxx6KnelZwWZ2k/view |
| Jumping | https://drive.google.com/file/d/1xfj1qU6cLenPN_gOmMwR0pfqNTGwYzM5/view |
| Still | https://drive.google.com/file/d/1uLwOWfO70HMY1DmjN6t-fPd8Pg7eEMzD/view |

## How to run

```bash
pip install -r requirements.txt
python prepare_data.py                  # rebuilds Data_clean/
python -m unittest discover tests       # runs the tests
jupyter notebook HMM_Activity_Recognition.ipynb
```

Run the notebook from the repository root.

## Limitations

- One person, one phone, one room.
- Still is tested minutes after its training files. The other three are tested on a different day.
- Each recording holds one activity, so the model never saw a real change between activities.
- Clean data for standing, walking and jumping is under the 90 s target. Raw data is over it.
