#%%
import wandb

api = wandb.Api()
runs = api.runs("ayberkydn/paper_dryrun")
print("---")
for run in runs:
    print(
        f'Mode: {run.config["mode"]}, Kappa: {run.config["kappa"]}, Restricted: {run.config["restricted"]}'
    )
    print(f'Avg LPIPS: {run.summary["avg_lpips"]}')
    print(f'Success rate: {run.summary["success_rate"]}')
    print("---")
# %%
