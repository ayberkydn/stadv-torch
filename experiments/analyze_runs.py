# %%
import torch, torchvision
import wandb
import src

api = wandb.Api()
sweep = api.sweep("paper/mtk6kkd4")
benign_imgs = src.utils.NIPS2017TargetedDataset("./data", head=10)
for run in sweep.runs:
    runhist = run.history()

    run.summary["mean_success_rate"] = runhist["success"].mean()
    run.summary["mean_final_loss"] = runhist["final_loss"].mean()
    artifacts = run.logged_artifacts()
    for art in artifacts:
        path = art.download()
    break

    # run.update()


# %%

