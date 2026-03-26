# Benchmark Analysis

`benchmark/analysis/` contains the released analysis scripts used to compute difficulty statistics, timestamp distributions, file-size summaries, and difficulty-vs-performance plots.

The public GitHub release does not ship the required analysis inputs. Before running any script here, download the following fullset files from Hugging Face into [`data/`](./data/):

- `Adam/Fullset/Adam.json`
- `Bei/Fullset/Bei.json`
- `Victoria/Fullset/Victoria.json`
- `Adam/Fullset/Adam_files.xlsx`
- `Bei/Fullset/Bei_files.xlsx`
- `Victoria/Fullset/Victoria_files.xlsx`

See [`data/README.md`](./data/README.md) for exact copy commands.

## Scripts

- `difficulty/generate_difficulty_reports.py`: computes the released scalar difficulty score and writes figures to `benchmark/analysis/outputs/figs/`
- `file_time/file_combined_boxplot.py`: plots timestamp distributions from the metadata spreadsheets
- `difficulty_vs_performance/difficulty_performance_plot.py`: aligns per-question difficulty with your local evaluation outputs
- `file_size/file_size_stats.py`: computes per-profile file-size breakdowns from raw source files downloaded from Hugging Face

## Typical commands

Difficulty figures:

```bash
python3 benchmark/analysis/difficulty/generate_difficulty_reports.py
```

Timestamp figure:

```bash
python3 benchmark/analysis/file_time/file_combined_boxplot.py
```

Difficulty vs performance with your own local results:

```bash
python3 benchmark/analysis/difficulty_vs_performance/difficulty_performance_plot.py \
  --results-root /path/to/your/local_eval_results
```

File-size summaries from the raw source folders downloaded from Hugging Face:

```bash
python3 benchmark/analysis/file_size/file_size_stats.py \
  --data-dir /path/to/HippoCamp \
  --output-dir benchmark/analysis/outputs/file_size
```

All generated figures are written under `benchmark/analysis/outputs/`.
