# Maternal XAI Research App

A multipage, browser-based showcase for the final-year project **Explainable AI for Maternal and Child Health Outcomes** by Adesanmi Anuoluwapo Gideon (CSC/2022/81014), Department of Computer Science, Federal University Oye-Ekiti.

The app includes:

- an interactive maternal-risk studio using the fitted five-model ensemble;
- a results dashboard with class-sensitive evaluation and confusion analysis;
- a methodology page covering the dataset, leakage controls, feature engineering, models, and explainability; and
- a project profile with the aim, objectives, contribution, limitations, and recommendations.

## Research result

The leakage-resistant champion reached **74.19% accuracy** on the 93-row profile-isolated holdout, with macro F1 of **0.5999**. Low-risk recall was **96.43%**, high-risk recall was **66.67%**, and mid-risk recall was **15.79%**. The app presents the separate 90.15% legacy row-split benchmark only as a leakage audit.

## Privacy and hosting

Model inference runs in the browser through the open-source ONNX Runtime Web library. The pinned runtime and five exported model files are bundled with the site, so assessment does not depend on a third-party API or CDN. Entered measurements remain on the visitor's device. The static site is configured for free GitHub Pages hosting and does not require a paid server.

## Local preview

Install the pinned browser dependency and refresh the bundled runtime when needed:

```text
npm install
npm run bundle:runtime
```

Serve the `dist` directory with any static HTTP server, for example `npm run serve`. Opening `index.html` directly from the filesystem will not load the model because browsers block local fetch requests.

Run `python scripts/validate_static.py` and `npm run check:js` before deployment. The included GitHub Actions workflow publishes `dist` to GitHub Pages whenever the `main` branch changes.

## Model export

The source deployment pipeline is converted into five ONNX model files with `scripts/export_onnx.py`. The exporter compares every member and the averaged ensemble against the original Python pipeline on 40 sampled records. The checked-in export achieved 100% class agreement with a maximum ensemble probability difference of `7.54e-7`.

## Dataset

[Maternal Health Risk Data](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data), originally collected from hospitals, community clinics, and maternal-care facilities through an IoT-based risk-monitoring system.

## Scope

This is a research prototype, not a medical device. The available dataset supports maternal-risk classification only; it contains no child or neonatal outcome variables.
