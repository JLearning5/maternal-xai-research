from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT
    / "Adesanmi Anuoluwapo Gideon [code]"
    / "maternal_child_health_artifacts v1"
    / "deployment_inference_pipeline.joblib"
)

wrapper = joblib.load(ARTIFACT)
print("wrapper", type(wrapper), vars(wrapper).keys())
print("model", type(wrapper.model), vars(wrapper.model).keys())
for name, pipeline in wrapper.model.estimators:
    print("\n", name, type(pipeline))
    print("steps", [(step_name, type(step)) for step_name, step in pipeline.steps])
    model = pipeline.named_steps["model"]
    print("model_type", type(model))
    print("classes", getattr(model, "classes_", None))
    print("features", getattr(model, "n_features_in_", None))
