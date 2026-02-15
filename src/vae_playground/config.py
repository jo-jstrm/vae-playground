from pathlib import Path
from typing import List, Self

from pydantic import BaseModel 
from pydantic_settings import SettingsConfigDict, YamlConfigSettingsSource, PydanticBaseSettingsSource, BaseSettings



class ModelConfig(BaseModel):
    img_size: int = 32
    input_dim: int = 3
    hidden_dims: List[int] = [16, 32, 64]
    latent_dim: int = 64


class TrainConfig(BaseModel):
    num_epochs: int = 1000
    batch_size: int = 512
    num_workers: int = 2
    learn_rate: float = 1e-3
    val_freq: int = 100
    checkpoint_freq: int = 1000


class DataConfig(BaseModel):
    train_split: float = 0.8


class TestConfig(BaseModel):
    batch_size: int = 512
    num_workers: int = 2


class Config(BaseSettings):
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    data: DataConfig = DataConfig()
    test: TestConfig = TestConfig()
    
    # @classmethod
    # def settings_customise_sources(
    #     cls,
    #     settings_cls: type[BaseSettings],
    #     init_settings: PydanticBaseSettingsSource,
    #     env_settings: PydanticBaseSettingsSource,
    #     dotenv_settings: PydanticBaseSettingsSource,
    #     file_secret_settings: PydanticBaseSettingsSource,
    # ) -> tuple[PydanticBaseSettingsSource, ...]:
    #     return (YamlConfigSettingsSource(settings_cls),)

    @classmethod
    def from_yaml(cls, path: Path) -> Self:
        """Needed to set yaml path during runtime. See https://github.com/pydantic/pydantic-settings/issues/259."""
        return cls(**YamlConfigSettingsSource(cls, path)())
