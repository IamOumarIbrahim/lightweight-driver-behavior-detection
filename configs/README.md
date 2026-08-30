# Benchmark Configuration

## Purpose

Machine-readable, frozen benchmark definitions.

## Contents

[`RGB`](RGB/) and [`NIR`](NIR/) contain track-specific dataset and model
settings. `backends.yaml` pins external code revisions, package versions, and
pretrained weights. [`patches`](patches/) contains the five audited backend
patches for accumulation and native Windows execution.

> [!IMPORTANT]
> Do not change a frozen protocol after test access.
