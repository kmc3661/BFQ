# Third-Party Dependencies

BFQ expects the following external repositories for multimodal model loading and evaluation:

- `3rdparty/LLaVA-NeXT`
- `3rdparty/lmms-eval`

You can fetch them with:

```bash
bash scripts/setup_thirdparty.sh
```

Or clone them manually:

```bash
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git 3rdparty/LLaVA-NeXT
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git 3rdparty/lmms-eval
```

If you keep those repositories elsewhere, set the environment variables `LLAVA_SRC_PATH` and `LMMS_EVAL_SRC_PATH` before running BFQ.
