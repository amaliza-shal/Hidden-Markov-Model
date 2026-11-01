# HMM Activity Recognition

Formative 2 · Individual assignment · Amaliza Shalom

I recorded four activities on my phone (standing, walking, jumping, still) and recognise them with a Gaussian Hidden Markov Model written from scratch in NumPy.

**Result:** 61 of 61 test windows and 12 of 12 unseen recordings correct. See the limitations below.

- [Report (PDF)](report/Formative2_HMM_Report.pdf)
- [Notebook](notebook/HMM_Activity_Recognition.ipynb)
- Videos: [Standing](https://drive.google.com/file/d/1Tq2zAthdj9Cif8v5GVrvESlW4DmJisJH/view) · [Walking](https://drive.google.com/file/d/1KrsBVEBJWtxzm3FCOFKxx6KnelZwWZ2k/view) · [Jumping](https://drive.google.com/file/d/1xfj1qU6cLenPN_gOmMwR0pfqNTGwYzM5/view) · [Still](https://drive.google.com/file/d/1uLwOWfO70HMY1DmjN6t-fPd8Pg7eEMzD/view)

## Structure

```
data/           raw Sensor Logger recordings, one folder per activity
Data_clean/     cleaned CSVs, split into train/ and test/
notebook/       HMM code, plots and tests, with outputs saved
scripts/        data cleaning and report build
figures/        plots
artifacts/      saved model and metrics
report/         final report
videos/         the four activity videos
```

## How to run

```bash
pip install -r requirements.txt
python scripts/prepare_data.py
jupyter nbconvert --to notebook --execute --inplace notebook/HMM_Activity_Recognition.ipynb
python scripts/build_report.py
```

## Data

iPhone 13, Sensor Logger 1.68, measured at 99.45 Hz. 60 recordings, none dropped.

| Activity | Train files | Test files | Clean seconds |
|---|---|---|---|
| Standing | 12 | 3 | 80.2 |
| Walking | 12 | 3 | 86.6 |
| Jumping | 12 | 3 | 87.4 |
| Still | 12 | 3 | 128.7 |

## Limitations

- One person, one phone, one room.
- Still is tested minutes after its training files. The other three are tested on a different day.
- Each recording holds one activity, so the model never saw a real change between activities.
- Clean data for standing, walking and jumping is under the 90 s target. Raw data is over it.
