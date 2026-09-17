from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from onnxmltools import convert_lightgbm, convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as ONNXMLFloatTensorType
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "Adesanmi Anuoluwapo Gideon [code]" / "maternal_child_health_artifacts v1"
PIPELINE_PATH = ARTIFACTS / "deployment_inference_pipeline.joblib"
METADATA_PATH = ARTIFACTS / "preprocessing" / "preprocessing_metadata.json"
DATA_PATH = ARTIFACTS / "split_assignment.csv"
OUTPUT_DIR = APP_ROOT / "dist" / "assets" / "models"
RAW_FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
EXPECTED_MEMBERS = ["svc", "extra_trees", "xgboost", "catboost", "lightgbm"]
# onnxmltools 1.16 supports tree-ensemble exports through ONNX opset 15.
# ONNX Runtime Web supports this opset, so use the same portable target for
# every ensemble member.
TARGET_OPSET = 15


def save_model(model: onnx.ModelProto, path: Path) -> None:
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def convert_member(name: str, estimator: Any, path: Path) -> None:
    feature_count = int(estimator.n_features_in_)
    if name in {"svc", "extra_trees"}:
        model = convert_sklearn(
            estimator,
            initial_types=[("input", FloatTensorType([None, feature_count]))],
            options={id(estimator): {"zipmap": False}},
            target_opset=TARGET_OPSET,
        )
        save_model(model, path)
        return

    if name == "xgboost":
        model = convert_xgboost(
            estimator,
            initial_types=[("input", ONNXMLFloatTensorType([None, feature_count]))],
            target_opset=TARGET_OPSET,
        )
        save_model(model, path)
        return

    if name == "lightgbm":
        model = convert_lightgbm(
            estimator,
            initial_types=[("input", ONNXMLFloatTensorType([None, feature_count]))],
            target_opset=TARGET_OPSET,
            zipmap=False,
        )
        save_model(model, path)
        return

    if name == "catboost":
        estimator.save_model(
            str(path),
            format="onnx",
            export_parameters={
                "onnx_domain": "maternal.xai",
                "onnx_model_version": 1,
                "onnx_doc_string": "CatBoost member of the maternal-risk ensemble",
                "onnx_graph_name": "maternal_xai_catboost",
            },
        )
        model = onnx.load(path)

        # CatBoost adds an ai.onnx.ml ZipMap node, which turns the probability
        # tensor into a sequence of maps. ONNX Runtime Web works most reliably
        # with the underlying [batch, class] tensor, so expose that tensor as a
        # graph output and remove only the presentation-layer ZipMap node.
        zipmaps = [node for node in model.graph.node if node.op_type == "ZipMap"]
        if len(zipmaps) != 1:
            raise RuntimeError(f"Expected one CatBoost ZipMap node, found {len(zipmaps)}")
        zipmap = zipmaps[0]
        probability_tensor = zipmap.input[0]
        retained_nodes = [node for node in model.graph.node if node is not zipmap]
        del model.graph.node[:]
        model.graph.node.extend(retained_nodes)
        retained_outputs = [
            output for output in model.graph.output if output.name != zipmap.output[0]
        ]
        del model.graph.output[:]
        model.graph.output.extend(retained_outputs)
        model.graph.output.extend(
            [
                onnx.helper.make_tensor_value_info(
                    probability_tensor,
                    onnx.TensorProto.FLOAT,
                    ["N", 3],
                )
            ]
        )
        save_model(model, path)
        return

    raise ValueError(f"Unsupported ensemble member: {name}")


def probability_output(session: ort.InferenceSession) -> str:
    for output in session.get_outputs():
        shape = output.shape
        if len(shape) >= 2 and (shape[-1] == 3 or str(shape[-1]) == "3"):
            return output.name
    raise RuntimeError(
        "No three-class probability output found. Outputs: "
        + repr([(output.name, output.shape, output.type) for output in session.get_outputs()])
    )


def run_onnx(session: ort.InferenceSession, output_name: str, values: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output = session.run([output_name], {input_name: values.astype(np.float32)})[0]
    probabilities = np.asarray(output, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)
    return probabilities


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = joblib.load(PIPELINE_PATH)
    members = dict(wrapper.model.estimators)
    missing = [name for name in EXPECTED_MEMBERS if name not in members]
    if missing:
        raise RuntimeError(f"Deployment ensemble is missing members: {missing}")

    data = pd.read_csv(DATA_PATH)
    samples = data.loc[:, RAW_FEATURES].sample(n=40, random_state=20260917).reset_index(drop=True)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    config_members: list[dict[str, Any]] = []
    original_probabilities: list[np.ndarray] = []
    converted_probabilities: list[np.ndarray] = []
    fidelity: dict[str, Any] = {}

    for name in EXPECTED_MEMBERS:
        pipeline = members[name]
        estimator = pipeline.named_steps["model"]
        path = OUTPUT_DIR / f"{name}.onnx"
        print(f"Converting {name} -> {path.name}", flush=True)
        convert_member(name, estimator, path)

        preprocessor = pipeline[:-1]
        transformed = np.asarray(preprocessor.transform(samples), dtype=np.float32)
        expected = np.asarray(pipeline.predict_proba(samples), dtype=float)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        output_name = probability_output(session)
        actual = run_onnx(session, output_name, transformed)
        max_error = float(np.max(np.abs(expected - actual)))
        class_agreement = float(np.mean(expected.argmax(axis=1) == actual.argmax(axis=1)))
        if class_agreement != 1.0 or max_error > 2e-4:
            raise RuntimeError(
                f"{name} conversion fidelity failed: max_error={max_error}, "
                f"agreement={class_agreement}"
            )

        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps.get("scale")
        member_config: dict[str, Any] = {
            "name": name,
            "url": f"assets/models/{path.name}",
            "input": session.get_inputs()[0].name,
            "output": output_name,
            "imputer": np.asarray(imputer.statistics_, dtype=float).tolist(),
            "scaled": scaler is not None,
        }
        if scaler is not None:
            member_config["mean"] = np.asarray(scaler.mean_, dtype=float).tolist()
            member_config["scale"] = np.asarray(scaler.scale_, dtype=float).tolist()
        config_members.append(member_config)
        original_probabilities.append(expected)
        converted_probabilities.append(actual)
        fidelity[name] = {
            "max_probability_error": max_error,
            "class_agreement": class_agreement,
            "size_bytes": path.stat().st_size,
        }

    original_ensemble = np.mean(original_probabilities, axis=0)
    converted_ensemble = np.mean(converted_probabilities, axis=0)
    ensemble_error = float(np.max(np.abs(original_ensemble - converted_ensemble)))
    ensemble_agreement = float(
        np.mean(original_ensemble.argmax(axis=1) == converted_ensemble.argmax(axis=1))
    )
    if ensemble_agreement != 1.0 or ensemble_error > 2e-4:
        raise RuntimeError(
            f"Ensemble fidelity failed: max_error={ensemble_error}, agreement={ensemble_agreement}"
        )

    model_config = {
        "model_name": "Diverse hard-vote maternal-risk ensemble",
        "deployment_behavior": "mean member probabilities followed by argmax",
        "class_names": metadata["class_names"],
        "raw_features": metadata["raw_features"],
        "engineered_features": metadata["engineered_features"],
        "raw_medians": {
            key: value["median"] for key, value in metadata["feature_ranges"].items()
        },
        "members": config_members,
        "validation": {
            "sample_rows": int(len(samples)),
            "ensemble_max_probability_error": ensemble_error,
            "ensemble_class_agreement": ensemble_agreement,
        },
    }
    (OUTPUT_DIR / "model-config.json").write_text(
        json.dumps(model_config, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "verification.json").write_text(
        json.dumps(
            {
                "source_artifact": PIPELINE_PATH.name,
                "sample_rows": int(len(samples)),
                "members": fidelity,
                "ensemble_max_probability_error": ensemble_error,
                "ensemble_class_agreement": ensemble_agreement,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Export verified on {len(samples)} rows: max ensemble probability error "
        f"{ensemble_error:.8g}, class agreement {ensemble_agreement:.1%}",
        flush=True,
    )


if __name__ == "__main__":
    main()
