# Data

The experiments use the public KuaiRec dataset:

- Project: https://kuairec.com/
- Source repository: https://github.com/chongminggao/KuaiRec
- Paper: https://doi.org/10.1145/3511808.3557220

Place the downloaded and extracted data at:

```text
data/kuairec/extracted/KuaiRec 2.0/data/
```

The experiment scripts discover the required CSV files recursively below `data/kuairec/extracted`.

The local dataset directory is ignored by Git because it contains approximately 2 GB of third-party data. A local copy is preserved with this working repository.
