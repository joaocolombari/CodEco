# Codeco collaboration rules

The authoritative corpus definition is `data/freeze-spec-v1.json`, together
with `data/manifest-lock-v1.json` and the versioned manifests. Do not silently
change corpus revisions, selected files, exclusions, splits, or hashes.

For a fresh installation, run:

```bash
./scripts/install_frozen_corpora.sh all
```

Large and reproducible material belongs under the ignored directories
`external/`, `data/downloads/`, `data/audio/`, and `data/derived/`. Never add
those directories to Git. Generated audits may be committed only when a new
freeze is intentionally created and documented.

Any intentional corpus change must:

1. receive a new manifest/freeze version;
2. regenerate `data/freeze-spec-v1.json` and the manifests;
3. document the reason and resulting counts in `README.md` and
   `docs/corpus-audit.md`;
4. pass `python3 scripts/verify_frozen_install.py --stage metadata` before a
   commit.

Keep the 4 s, 48 kHz study protocol and the parameters in
`configs/acoustic-indices-v1.yaml` unchanged unless the experimental design is
explicitly revised.
