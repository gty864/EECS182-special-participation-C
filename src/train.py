import hydra
from omegaconf import DictConfig
from src.trainers.base import SGDTrainer

@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    trainer = SGDTrainer(cfg)
    losses_gd, losses_mom = trainer.train()

    # Save artifacts (optional)
    import joblib, pathlib
    out = pathlib.Path("outputs")
    out.mkdir(exist_ok=True)
    joblib.dump({"gd": losses_gd, "mom": losses_mom}, out / "losses.pkl")

if __name__ == "__main__":
    main()