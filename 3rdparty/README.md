# Third-Party Dependencies

BFQ expects the following external repositories for multimodal model loading and evaluation:

- `3rdparty/LLaVA-NeXT`
- `3rdparty/lmms-eval`

You can fetch them with:

```bash
bash scripts/setup_thirdparty.sh
```

The script pins upstream revisions to avoid version drift. Avoid mixing in newer upstream versions when reproducing results.

If you keep those repositories elsewhere, set the environment variables `LLAVA_SRC_PATH` and `LMMS_EVAL_SRC_PATH` before running BFQ.
