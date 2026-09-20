import os

from src.classes import BioprocessMonitor

bm_infos = [
    dict(mode="A", ph_lims=(4.8, 5.6), temperature_lims=(34.0, 36.0)),
    dict(mode="B", ph_lims=(5.1, 5.5), temperature_lims=(34.5, 35.5))
]

filepath_dataset = os.path.join("datasets", "dataset_fermentation.csv")

for bm_info in bm_infos:
    bm = BioprocessMonitor(
        filepath=filepath_dataset,
        ph_lims=bm_info["ph_lims"],
        temperature_lims=bm_info["temperature_lims"]
    )

    for batch_id in range(1, bm.get_n_batches() + 1):
        filepath_figure = os.path.join("figures", f"Batch_{batch_id:03d}_Mode_{bm_info['mode']}.png")
        bm.export_dashboard(batch_id=batch_id, filepath=filepath_figure)

    bm.export_summary(filepath=os.path.join("tables", f"Summary_Mode_{bm_info['mode']}.csv"))
