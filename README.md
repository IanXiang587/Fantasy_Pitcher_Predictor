## Stage 1 – Data Pipeline

Built a reproducible data pipeline using pybaseball to download 2024 Statcast pitch-level data. Aggregated the raw pitch-level dataset into a pitcher-game table, calculating metrics such as strikeouts, average velocity, average spin rate, and CSW%. Organizing the data at the pitcher-game level establishes the foundation for leakage-safe feature engineering in later stages while preserving the underlying pitch-level information needed to compute advanced metrics.
