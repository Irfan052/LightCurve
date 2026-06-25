# Light Curve CSV Upload Format

The AstroExo AI Pipeline accepts custom light curve data via CSV uploads. To ensure the pipeline can accurately process your data, please adhere to the following formatting requirements.

## Required Columns

The CSV file **must** include the following column headers (case-insensitive):
- `time`: The timestamp or phase of the observation (typically in days).
- `flux`: The relative or absolute flux measurement.

### Optional Columns
- `flux_err`: The uncertainty/error associated with the flux measurement. If omitted, a default error of 0.001 is applied to all points.

## Formatting Rules

1. **Numeric Data Only**: All rows below the header must contain numeric values. Non-numeric rows or unparseable values will be converted to `NaN` and filtered out.
2. **Missing Values**: Rows containing missing (`NaN` / empty) values in `time` or `flux` will be automatically dropped.
3. **Empty Files**: The pipeline will reject empty files.
4. **Delimiters**: Use a standard comma (`,`) delimiter.

## Example valid CSV format

```csv
time,flux,flux_err
1325.2345,1.0001,0.0005
1325.2483,0.9998,0.0005
1325.2621,0.9850,0.0006
1325.2760,0.9852,0.0006
1325.2899,1.0000,0.0005
```

## Creating a Sample File
You can create a template by copying the block above into a plain text editor and saving it as `sample_lightcurve.csv`.
