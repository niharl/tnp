import os

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf
from plot import plot

import wandb
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback, DetailedTimingCallback

# Import the new loader
from gp_loading import ChunkedGPDataset

def main():
    experiment = initialize_experiment()

    model = experiment.model
    
    train_params = experiment.train_params
    train_batches_per_epoch = train_params.samples_per_epoch // train_params.batch_size
    val_params = experiment.val_params
    val_batches_per_epoch = val_params.samples_per_epoch // val_params.batch_size

    print(f"Loading TRAINING data from: {experiment.misc.gp_folder}")
    gen_train = ChunkedGPDataset(
        experiment.misc.gp_folder, 
        shuffle_files=True,
        nc = experiment.params.nc,
        nt = experiment.params.nt
    )

    print(f"Loading VALIDATION data from: {experiment.misc.gp_folder}")
    gen_val = ChunkedGPDataset(
        experiment.misc.gp_folder, 
        shuffle_files=False,
        nc = experiment.params.nc,
        nt = experiment.params.nt
    )
    # --- DATASET LOADING LOGIC END ---

    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    train_loader = torch.utils.data.DataLoader(
        gen_train,
        batch_size=None,
        num_workers=experiment.misc.num_workers,
        # We don't use worker_init_fn because ChunkedGPDataset handles splitting internally
        worker_init_fn=None,
        persistent_workers=True if experiment.misc.num_workers > 0 else False,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        gen_val,
        batch_size=None,
        num_workers=experiment.misc.num_val_workers,
        worker_init_fn=None,
        persistent_workers=True if experiment.misc.num_val_workers > 0 else False,
        pin_memory=False,
    )

    def plot_fn(model, batches, name):
        plot(
            model=model,
            batches=batches,
            num_fig=min(5, len(batches)),
            name=name,
            pred_fn=experiment.misc.pred_fn,
            savefig=experiment.misc.savefig,
            logging=experiment.misc.logging
        )

    if experiment.misc.resume_from_checkpoint is not None:
        api = wandb.Api()
        artifact = api.artifact(experiment.misc.resume_from_checkpoint)
        artifact_dir = artifact.download()
        ckpt_file = os.path.join(artifact_dir, "model.ckpt")

        lit_model = (
            LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                ckpt_file,
            )
        )
    else:
        ckpt_file = None
        lit_model = LitWrapper(
            model=model,
            optimiser=optimiser,
            loss_fn=experiment.misc.loss_fn,
            pred_fn=experiment.misc.pred_fn,
            plot_fn=plot_fn,
            plot_interval=experiment.misc.plot_interval,
        )

    # Initialise the combined timing/memory callback
    metrics_callback = DetailedTimingCallback()

    if experiment.misc.logging:
        logger = pl.loggers.WandbLogger(
            project=experiment.misc.project,
            name=experiment.misc.name,
            config=OmegaConf.to_container(experiment.config),
            log_model="all",
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=experiment.misc.checkpoint_interval,
            save_last=True,
        )
        performance_callback = LogPerformanceCallback()
        callbacks = [checkpoint_callback, performance_callback, metrics_callback]
    else:
        logger = False
        callbacks = [metrics_callback]
        
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=train_batches_per_epoch,
        limit_val_batches=val_batches_per_epoch,
        log_every_n_steps=(
            experiment.misc.log_interval if not experiment.misc.logging else None
        ),
        devices="auto",
        accelerator="auto",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=(experiment.misc.check_val_every_n_epoch),
        gradient_clip_val=experiment.misc.gradient_clip_val,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_file,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()