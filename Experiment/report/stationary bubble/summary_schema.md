# Stationary Bubble / Ellipse Summary CSV Schema

- CSV: `Experiment/report/stationary bubble/summary.csv`
- Rows: 40 = 4 circle rows + 36 ellipse rows
- Existing circle rows are retained; ellipse rows are regenerated from raw summaries/traces.
- Use `geometry_family`, `tier`, and `evidence_status` to filter plot-ready records.
- Complete ellipse process rows have `evidence_status=complete` and `report_ready=true`.
- Partial traces and missing rows are retained only for progress/accounting.

