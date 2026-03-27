# Analysis Data Placement

The analysis scripts expect six fullset files from the Hugging Face release in this directory:

- `Adam.json`
- `Bei.json`
- `Victoria.json`
- `Adam_files.xlsx`
- `Bei_files.xlsx`
- `Victoria_files.xlsx`

Copy them into place as follows:

```bash
mkdir -p benchmark/analysis/data

cp /path/to/HippoCamp/Adam/Fullset/Adam.json benchmark/analysis/data/Adam.json
cp /path/to/HippoCamp/Bei/Fullset/Bei.json benchmark/analysis/data/Bei.json
cp /path/to/HippoCamp/Victoria/Fullset/Victoria.json benchmark/analysis/data/Victoria.json

cp /path/to/HippoCamp/Adam/Fullset/Adam_files.xlsx benchmark/analysis/data/Adam_files.xlsx
cp /path/to/HippoCamp/Bei/Fullset/Bei_files.xlsx benchmark/analysis/data/Bei_files.xlsx
cp /path/to/HippoCamp/Victoria/Fullset/Victoria_files.xlsx benchmark/analysis/data/Victoria_files.xlsx
```

After these files are in place, the scripts described in the [`analysis README`](../README.md) can be run directly.
