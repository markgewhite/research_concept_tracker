"""Configuration settings for ArXiv Concept Tracker"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with Kalman filter parameters"""

    # Embedding configuration
    embedding_model: str = "Qwen/Qwen2.5-Embedding"
    cache_dir: str = "./cache"

    # ArXiv API settings
    arxiv_rate_limit: float = 3.0
    arxiv_max_retries: int = 3

    # Kalman Filter - Physics constraints
    max_velocity: float = 0.05
    """Maximum concept drift per time step (higher = allows faster concept evolution)"""

    max_acceleration: float = 0.02
    """Maximum change in velocity (higher = allows more sudden direction changes)"""

    process_noise: float = 0.01
    """Natural drift/uncertainty in concept position"""

    measurement_noise: float = 0.1
    """Uncertainty in embedding measurements"""

    # Similarity thresholds
    threshold_auto_include: float = 0.85
    """Papers above this similarity are auto-accepted with high confidence"""

    threshold_strong: float = 0.75
    """Papers above this are accepted if they pass velocity checks"""

    threshold_moderate: float = 0.65
    """Minimum similarity for consideration (below this = reject)"""

    threshold_reject: float = 0.55
    """Hard cutoff - papers below this are always rejected"""

    # Tracking defaults
    default_window_months: int = 6
    max_papers_per_window: int = 500

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
