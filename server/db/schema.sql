-- idempotent. runs at server boot.
CREATE TABLE IF NOT EXISTS runs (
  run_id      TEXT PRIMARY KEY,
  dataset     TEXT, split TEXT, model TEXT, feature_set TEXT, seed INT,
  source      TEXT NOT NULL DEFAULT 'hybrid',
  is_mock     BOOLEAN NOT NULL DEFAULT FALSE,
  metrics     JSONB,
  synced_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Numbers copied from the OceanVerse paper (Lu et al., KDD'26, Table 2). Label: PUBLISHED. Never mix with ours.
CREATE TABLE IF NOT EXISTS published_results (
  dataset TEXT NOT NULL, split TEXT NOT NULL, model TEXT NOT NULL,
  rmse DOUBLE PRECISION, r2 DOUBLE PRECISION,
  reference TEXT NOT NULL DEFAULT 'Lu et al., KDD 2026, Table 2',
  PRIMARY KEY (dataset, split, model)
);

INSERT INTO published_results (dataset, split, model, rmse, r2) VALUES
 ('CESM2_omip1','temporal','XGBoost',0.074,0.356), ('CESM2_omip1','temporal','LSTM',0.056,0.628),
 ('CESM2_omip1','temporal','MLP',0.050,0.696),     ('CESM2_omip1','temporal','Transformer',0.069,0.433),
 ('CESM2_omip1','temporal','OxyGenerator',0.044,0.764),
 ('CESM2_omip2','temporal','XGBoost',0.076,0.344), ('CESM2_omip2','temporal','LSTM',0.075,0.319),
 ('CESM2_omip2','temporal','MLP',0.054,0.661),     ('CESM2_omip2','temporal','Transformer',0.074,0.371),
 ('CESM2_omip2','temporal','OxyGenerator',0.049,0.716),
 ('GFDL_ESM4','temporal','XGBoost',0.072,0.201),   ('GFDL_ESM4','temporal','LSTM',0.072,0.225),
 ('GFDL_ESM4','temporal','MLP',0.055,0.529),       ('GFDL_ESM4','temporal','Transformer',0.072,0.216),
 ('GFDL_ESM4','temporal','OxyGenerator',0.053,0.579)
ON CONFLICT DO NOTHING;
