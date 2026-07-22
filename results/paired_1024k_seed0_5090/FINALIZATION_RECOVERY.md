# Finalization recovery audit

1. The first formal metric launch stopped before producing a metric because `torchrun` was absent from the non-interactive PATH. The failed launch log is preserved as `kid5k/sigmoid/nfe1/launch_failure_127.log`. Evaluation resumed with a runtime-only wrapper that invokes the frozen venv as `python -m torch.distributed.run`; no evaluation code or protocol parameter changed.
2. All eight KID/FID cells then completed successfully. The first finalization attempt found that the evaluation venv lacks matplotlib. Its traceback is preserved in `finalize.log`. Pure-CPU plotting and packaging were rerun with `/root/miniconda3/bin/python`, which has matplotlib 3.10.3 and the same project data; no metric was rerun or altered.
