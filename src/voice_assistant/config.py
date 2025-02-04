"""Configuration settings for the voice assistant"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    # API Keys with aliases to match environment variable names
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    google_api_key: str = Field(alias="GOOGLE_API_KEY")
    anthropic_api_key: str = Field(alias="ANTHROPIC_API_KEY")
    hf_token: str = Field(alias="HF_TOKEN")
    gemini_apikey: str = Field(alias="GEMINI_APIKEY")
    deepseek_api_key: str = Field(alias="DEEPSEEK_API_KEY")
    
    # Audio settings
    sample_rate: int = Field(default=16000)
    channels: int = Field(default=1)
    dtype: str = Field(default="float32")
    
    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=True,
        extra='allow',
        env_file_encoding='utf-8'
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

settings = get_settings()