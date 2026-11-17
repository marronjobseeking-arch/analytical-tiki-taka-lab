"""NumPy code for the fixed-weight ensemble and the three-state Viterbi decoder."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

H, A, D, X = 0, 1, 2, 3
STATE_ORDER = ("H", "A", "D")
PUBLIC_MATCH_IDS = {
    "DFL-MAT-J03WMX", "DFL-MAT-J03WN1", "DFL-MAT-J03WOH", "DFL-MAT-J03WOY",
    "DFL-MAT-J03WPY", "DFL-MAT-J03WQQ", "DFL-MAT-J03WR9",
    "Sample_Game_1", "Sample_Game_2", "Sample_Game_3",
}
_SOURCE_KEYS = {
    "match_id", "frame_index", "frame", "period", "state_order", "metadata_json",
    "proba", "probabilities", "x_mask", "prediction",
}
_FEATURE_KEYS = {"match_id", "frame_index", "frame", "period", "FH", "FA", "metadata_json"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _scalar_text(archive, key: str) -> str:
    value = np.asarray(archive[key])
    if value.ndim != 0 or value.dtype.kind not in "US":
        raise ValueError(f"{key} must be a Unicode scalar")
    return str(value.item())


def _metadata_from_archive(archive) -> tuple[dict[str, Any], str]:
    if "metadata_json" not in archive.files:
        raise ValueError("probability archive requires metadata_json")
    payload = _scalar_text(archive, "metadata_json")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("metadata_json must contain an object")
    return value, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_match_id(value: str) -> str:
    raw = str(value)
    for prefix in ("idsse_", "metrica_", "atd_", "bar_ath_", "bar_"):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def _roster_ids(value: Any, field: str, *, allow_pff: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"lineage {field} must be a list of non-empty strings")
    normalized = [_canonical_match_id(item) for item in value]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"lineage {field} has duplicate IDs after namespace normalization")
    unknown = [raw for raw, normalized_id in zip(value, normalized)
               if normalized_id not in PUBLIC_MATCH_IDS
               and not (allow_pff and raw.startswith("pff_") and len(raw) > 4)]
    if unknown:
        raise ValueError(f"lineage {field} has unknown IDs: {unknown[:3]}")
    return normalized


def _groups_from_lineage(fitting: Mapping[str, Any], field: str, *, allow_pff: bool
                         ) -> list[tuple[str, list[str]]]:
    flat = fitting.get(field)
    by_member = fitting.get(f"{field}_by_member")
    groups: list[tuple[str, list[str]]] = []
    if flat is not None:
        groups.append(("flat", _roster_ids(flat, field, allow_pff=allow_pff)))
    if by_member is not None:
        if not isinstance(by_member, list) or not by_member:
            raise ValueError(f"lineage {field}_by_member must be a non-empty list")
        seen_members: set[str] = set()
        for row in by_member:
            if not isinstance(row, dict) or not isinstance(row.get("member_id"), str):
                raise ValueError(f"lineage {field}_by_member rows need member_id")
            member = row["member_id"]
            if not member or member in seen_members:
                raise ValueError(f"lineage {field}_by_member has an empty or duplicate member_id")
            seen_members.add(member)
            groups.append((member, _roster_ids(row.get(field), f"{field}_by_member.{member}", allow_pff=allow_pff)))
    if not groups:
        raise ValueError(f"lineage requires {field} or {field}_by_member")
    if flat is not None and by_member is not None:
        flat_set = set(groups[0][1])
        member_union = set().union(*(set(roster) for _, roster in groups[1:]))
        if flat_set != member_union:
            raise ValueError(f"flat and per-member {field} rosters disagree")
    return groups


def audit_lineage(fitting: Mapping[str, Any], target_id: str, *,
                  expected_outer_target: str | None = None,
                  allow_pff: bool = True) -> dict[str, Any]:
    """Require explicit fit/cal rosters and prove the prediction target is absent."""
    if not isinstance(fitting, Mapping):
        raise ValueError("fitting lineage must be an object")
    target = _canonical_match_id(target_id)
    fit_groups = _groups_from_lineage(fitting, "fit_match_ids", allow_pff=allow_pff)
    cal_groups = _groups_from_lineage(fitting, "calibration_match_ids", allow_pff=allow_pff)
    for label, groups in (("fit", fit_groups), ("calibration", cal_groups)):
        for name, roster in groups:
            if target in roster:
                raise ValueError(f"prediction target {target} occurs in {label} roster {name}")
    train_ids = fitting.get("meta_training_match_ids")
    normalized_train = None
    if train_ids is not None:
        normalized_train = _roster_ids(train_ids, "meta_training_match_ids", allow_pff=False)
        if target in normalized_train:
            raise ValueError(f"prediction target {target} occurs in meta_training_match_ids")
    if expected_outer_target is not None:
        outer = fitting.get("outer_target_match_id", fitting.get("outer_target"))
        if not isinstance(outer, str) or _canonical_match_id(outer) != _canonical_match_id(expected_outer_target):
            raise ValueError("meta-model outer target does not match the selected target route")
    return {
        "target_id": target,
        "fit_groups": [{"member_id": member, "match_ids": ids} for member, ids in fit_groups],
        "calibration_groups": [{"member_id": member, "match_ids": ids} for member, ids in cal_groups],
        "meta_training_match_ids": normalized_train,
        "target_excluded": True,
    }


def _validate_probabilities(probability: np.ndarray, n_frames: int, period: np.ndarray,
                            source_name: str) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.shape != (n_frames, 3):
        raise ValueError(f"{source_name}: proba must have shape (T, 3)")
    active = np.isin(period, (1, 2))
    rows = probability[active]
    if (not np.isfinite(rows).all() or (rows < 0).any() or (rows > 1).any()
            or not np.allclose(rows.sum(axis=1), 1.0, atol=1e-5, rtol=0)):
        raise ValueError(f"{source_name}: active rows are not normalized finite H/A/D probabilities")
    output = np.full((n_frames, 3), 1.0 / 3.0, dtype=np.float64)
    output[active] = rows
    return output


def read_probability_archive(path: Path, expected_match_id: str, source_name: str,
                             expected_metadata_sha256: str | None = None
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                        dict[str, Any], str, str]:
    """Read only probability/grid/metadata arrays; never access saved predictions."""
    with np.load(path, allow_pickle=False) as archive:
        unexpected = sorted(set(archive.files) - _SOURCE_KEYS)
        if unexpected:
            raise ValueError(f"{path}: unsupported archive arrays: {unexpected}")
        metadata, metadata_payload_sha = _metadata_from_archive(archive)
        if expected_metadata_sha256 and metadata_payload_sha != expected_metadata_sha256:
            raise ValueError(f"{path}: metadata payload SHA mismatch")
        scalar_id = _scalar_text(archive, "match_id") if "match_id" in archive.files else None
        match_meta = metadata.get("match") or {}
        metadata_id = match_meta.get("id") if isinstance(match_meta, dict) else None
        actual_id = scalar_id or metadata_id
        if actual_id != expected_match_id:
            raise ValueError(f"{source_name}: expected source target {expected_match_id}, found {actual_id!r}")
        if "frame" in archive.files and "frame_index" in archive.files:
            raise ValueError(f"{source_name}: archive must have one frame index key")
        frame_key = "frame_index" if "frame_index" in archive.files else "frame"
        if frame_key not in archive.files or "period" not in archive.files:
            raise ValueError(f"{source_name}: frame and period are required")
        frame = np.asarray(archive[frame_key])
        period = np.asarray(archive["period"])
        if (frame.ndim != 1 or frame.dtype.kind not in "iu" or len(frame) == 0
                or period.ndim != 1 or period.shape != frame.shape or period.dtype.kind not in "iu"):
            raise ValueError(f"{source_name}: frame and period are not aligned integer vectors")
        frame = frame.astype(np.int64, copy=True)
        period = period.astype(np.int8, copy=True)
        if not np.array_equal(frame, np.arange(len(frame), dtype=np.int64)):
            raise ValueError(f"{source_name}: only full native frame arange(T) is supported")
        if not np.isin(period, (0, 1, 2)).all() or not np.isin(period, (1, 2)).any():
            raise ValueError(f"{source_name}: invalid or inactive-only period mapping")
        if "proba" in archive.files and "probabilities" in archive.files:
            raise ValueError(f"{source_name}: use only one probability key")
        pkey = "proba" if "proba" in archive.files else "probabilities"
        if pkey not in archive.files:
            raise ValueError(f"{source_name}: probability array is missing")
        representation = metadata.get("representation") or {}
        declared_order = representation.get("channel_order")
        if declared_order != ["H", "A", "D"] and tuple(declared_order or ()) != STATE_ORDER:
            raise ValueError(f"{source_name}: metadata must declare H/A/D probability channels")
        if representation.get("kind") not in ("probability", "factorized_probability"):
            raise ValueError(f"{source_name}: input is not declared verified probability")
        if "state_order" in archive.files:
            order_array = np.asarray(archive["state_order"])
            if order_array.ndim != 1 or order_array.dtype.kind not in "US" or tuple(order_array.tolist()) != STATE_ORDER:
                raise ValueError(f"{source_name}: archive state_order must be H/A/D")
        probability = _validate_probabilities(archive[pkey], len(frame), period, source_name)
        x_mask = period == 0
        if "x_mask" in archive.files:
            extra_mask = np.asarray(archive["x_mask"])
            if extra_mask.shape != period.shape or extra_mask.dtype.kind not in "bu":
                raise ValueError(f"{source_name}: x_mask must be boolean and frame-aligned")
            x_mask |= extra_mask.astype(bool)
        # The optional `prediction` key is intentionally never accessed.
    return frame, period, x_mask, probability, metadata, metadata_payload_sha, sha256_array(probability)


def factorized_to_had(p_dead: np.ndarray, p_home_alive: np.ndarray) -> np.ndarray:
    dead = np.asarray(p_dead, dtype=np.float64)
    home = np.asarray(p_home_alive, dtype=np.float64)
    if dead.ndim != 1 or home.shape != dead.shape:
        raise ValueError("factorized probabilities must be equal-length vectors")
    return np.column_stack(((1.0 - dead) * home, (1.0 - dead) * (1.0 - home), dead))


def probability_tensor(source_probabilities: Mapping[str, np.ndarray], order: Sequence[str] | None = None) -> np.ndarray:
    names = list(order) if order is not None else sorted(source_probabilities)
    if not names or len(set(names)) != len(names) or set(names) != set(source_probabilities):
        raise ValueError("source order must name each probability source exactly once")
    arrays = [np.asarray(source_probabilities[name], dtype=np.float64) for name in names]
    if any(array.ndim != 2 or array.shape[1] != 3 or array.shape != arrays[0].shape for array in arrays):
        raise ValueError("source probability arrays are not aligned H/A/D matrices")
    result = np.stack(arrays, axis=1)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=2), 1.0, atol=1e-5, rtol=0):
        raise ValueError("source probabilities are not finite normalized H/A/D")
    return result


def combine_fixed_mean(probabilities: Mapping[str, np.ndarray], recipe: Mapping[str, Any],
                       *, output_dtype: str = "float32") -> np.ndarray:
    names = sorted(probabilities)
    mode = recipe.get("mode")
    def resolve_weights(value: Any, label: str) -> dict[str, float]:
        if value == "uniform":
            resolved = {name: 1.0 / len(names) for name in names}
        elif isinstance(value, dict) and set(value) == set(names):
            resolved = {name: float(value[name]) for name in names}
        else:
            raise ValueError(f"{label} must cover every source or be 'uniform'")
        if (not all(math.isfinite(weight) and weight >= 0 for weight in resolved.values())
                or not math.isclose(sum(resolved.values()), 1.0, rel_tol=0, abs_tol=1e-10)):
            raise ValueError(f"{label} must be finite, non-negative, and sum to one")
        return resolved

    weights = resolve_weights(recipe.get("weights", "uniform"), "fixed mean weights")
    if mode == "probability_mean":
        result = sum(weights[name] * np.asarray(probabilities[name], dtype=np.float64) for name in names)
    elif mode == "factorized_mean":
        dead_weights = resolve_weights(recipe.get("dead_weights", recipe.get("weights", "uniform")),
                                       "fixed mean dead_weights")
        ha_weights = resolve_weights(recipe.get("ha_weights", recipe.get("weights", "uniform")),
                                     "fixed mean ha_weights")
        dead = sum(dead_weights[name] * np.asarray(probabilities[name], dtype=np.float64)[:, D]
                   for name in names)
        home = []
        for name in names:
            p = np.asarray(probabilities[name], dtype=np.float64)
            alive = p[:, H] + p[:, A]
            home.append(np.divide(p[:, H], alive, out=np.full(len(p), 0.5), where=alive > 1e-12))
        p_home_alive = sum(ha_weights[name] * home[index] for index, name in enumerate(names))
        result = factorized_to_had(dead, p_home_alive)
    else:
        raise ValueError(f"unsupported fixed mean mode {mode!r}")
    if output_dtype not in ("float32", "float64"):
        raise ValueError("fixed mean output_dtype must be float32 or float64")
    return np.asarray(result, dtype=np.dtype(output_dtype))


def logistic_factorized(probabilities: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    tensor = np.asarray(probabilities, dtype=np.float64)
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        raise ValueError("logistic meta-model requires an (T, sources, 3) probability tensor")
    count = tensor.shape[1]
    if int(model.get("source_count", -1)) != count:
        raise ValueError("logistic source_count differs from input tensor")
    dead_w = np.asarray(model.get("dead_weights"), dtype=np.float64)
    ha_w = np.asarray(model.get("ha_weights"), dtype=np.float64)
    if dead_w.shape != (count + 1,) or ha_w.shape != (count,):
        raise ValueError("logistic model coefficient shapes are malformed")
    pdead = np.clip(tensor[:, :, D], 1e-7, 1 - 1e-7)
    dead_logit = np.log(pdead / (1 - pdead))
    ha = np.divide(tensor[:, :, H], tensor[:, :, H] + tensor[:, :, A],
                   out=np.full((len(tensor), count), 0.5),
                   where=(tensor[:, :, H] + tensor[:, :, A]) > 1e-12)
    ha = np.clip(ha, 1e-7, 1 - 1e-7)
    ha_logit = np.log(ha / (1 - ha))
    dead_probability = _sigmoid(dead_w[0] + dead_logit @ dead_w[1:])
    home_probability = _sigmoid(ha_logit @ ha_w)
    return factorized_to_had(dead_probability, home_probability).astype(np.float32)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    z = np.asarray(value, dtype=np.float64)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-np.clip(z[positive], -60, 60)))
    exp_value = np.exp(np.clip(z[~positive], -60, 60))
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def context_features(fh: np.ndarray, fa: np.ndarray) -> np.ndarray:
    left, right = np.asarray(fh, dtype=np.float32), np.asarray(fa, dtype=np.float32)
    if left.ndim != 2 or left.shape[1] != 77 or right.shape != left.shape:
        raise ValueError("gate context requires aligned FH/FA matrices with 77 columns each")
    pair = np.concatenate(((left + right) / 2.0, np.abs(left - right) / 2.0), axis=1)
    return _sanitize_nonfinite(pair)


def gate_features(mode: str, fh: np.ndarray, fa: np.ndarray, experts: np.ndarray) -> np.ndarray:
    context = context_features(fh, fa)
    if mode == "context":
        return context
    if mode != "context_probability":
        raise ValueError(f"unsupported gate feature mode {mode!r}")
    p = np.asarray(experts, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (7, 3) or len(p) != len(context):
        raise ValueError("context_probability gate needs seven aligned H/A/D experts")
    alive = p[:, :, H] + p[:, :, A]
    conditional_home = np.full(alive.shape, 0.5, dtype=np.float64)
    np.divide(p[:, :, H], alive, out=conditional_home, where=alive > 1e-6)
    extra = np.concatenate((p[:, :, D], np.abs(np.clip(conditional_home, 0, 1) - 0.5)), axis=1)
    return _sanitize_nonfinite(np.concatenate((context, extra), axis=1))


def _sanitize_nonfinite(value: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(value, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def _softmax(value: np.ndarray) -> np.ndarray:
    z = np.asarray(value, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    exp_value = np.exp(z)
    return exp_value / exp_value.sum(axis=1, keepdims=True)


def factorized_gate(probabilities: np.ndarray, features: np.ndarray,
                    model: Mapping[str, np.ndarray]) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (7, 3) or x.ndim != 2 or len(x) != len(p):
        raise ValueError("factorized gate requires aligned (T,7,3) experts and 2D features")
    mean = np.asarray(model["norm_mean"], dtype=np.float64)
    std = np.asarray(model["norm_std"], dtype=np.float64)
    dw = np.asarray(model["dead_weight"], dtype=np.float64)
    db = np.asarray(model["dead_bias"], dtype=np.float64)
    hw = np.asarray(model["ha_weight"], dtype=np.float64)
    hb = np.asarray(model["ha_bias"], dtype=np.float64)
    width = x.shape[1]
    if (mean.shape != (width,) or std.shape != (width,) or np.any(std <= 0)
            or dw.shape != (7, width) or hw.shape != (7, width)
            or db.shape != (7,) or hb.shape != (7,)):
        raise ValueError("factorized gate model shapes do not match feature width")
    normalized = np.clip((_sanitize_nonfinite(x) - mean) / std, -10.0, 10.0)
    wd = _softmax(normalized @ dw.T + db)
    wh = _softmax(normalized @ hw.T + hb)
    alive = p[:, :, H] + p[:, :, A]
    home = np.full(alive.shape, 0.5, dtype=np.float64)
    np.divide(p[:, :, H], alive, out=home, where=alive > 1e-6)
    dead_mix = np.sum(wd * p[:, :, D], axis=1)
    home_mix = np.sum(wh * np.clip(home, 0, 1), axis=1)
    result = factorized_to_had(dead_mix, home_mix)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError("factorized gate produced invalid H/A/D probabilities")
    return result.astype(np.float32)


def hgb_features(mode: str, experts: np.ndarray, fh: np.ndarray | None = None,
                 fa: np.ndarray | None = None) -> np.ndarray:
    p = np.asarray(experts, dtype=np.float32)
    if p.ndim != 3 or p.shape[1:] != (7, 3):
        raise ValueError("probability HGB requires seven H/A/D experts")
    prob = p.reshape(len(p), 21)
    if mode == "prob_only":
        return prob
    if fh is None or fa is None:
        raise ValueError(f"HGB mode {mode} requires pinned FH/FA context features")
    left, right = np.asarray(fh, dtype=np.float32), np.asarray(fa, dtype=np.float32)
    if left.shape != (len(p), 77) or right.shape != left.shape:
        raise ValueError("HGB context features must be aligned 77-column FH/FA arrays")
    context = np.concatenate((left, right), axis=1)
    if mode == "feature_only":
        return context
    if mode == "prob_feature":
        return np.concatenate((prob, context), axis=1)
    raise ValueError(f"unsupported probability HGB feature mode {mode!r}")


def _swap_hgb_features(x: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(x, dtype=np.float32)
    if mode == "prob_only":
        return values.reshape(len(values), 7, 3)[:, :, [A, H, D]].reshape(len(values), 21).copy()
    if mode == "feature_only":
        return np.concatenate((values[:, 77:], values[:, :77]), axis=1)
    if mode == "prob_feature":
        prob = values[:, :21].reshape(len(values), 7, 3)[:, :, [A, H, D]].reshape(len(values), 21)
        context = np.concatenate((values[:, 21 + 77:], values[:, 21:21 + 77]), axis=1)
        return np.concatenate((prob, context), axis=1)
    raise ValueError(f"unsupported HGB mode {mode!r}")


def hgb_probabilities(model: Any, x: np.ndarray, mode: str) -> np.ndarray:
    """Apply a pinned scikit-learn model symmetrically; import sklearn lazily."""
    original = _predict_hgb(model, x)
    swapped = _predict_hgb(model, _swap_hgb_features(x, mode))[:, [A, H, D]]
    result = 0.5 * (original + swapped)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("symmetric HGB probabilities are not normalized")
    return result.astype(np.float32)


def _predict_hgb(model: Any, x: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.predict_proba(np.asarray(x, dtype=np.float32)), dtype=np.float64)
    classes = np.asarray(getattr(model, "classes_", ()))
    if raw.ndim != 2 or raw.shape != (len(x), 3) or set(classes.astype(int).tolist()) != {H, A, D}:
        raise ValueError("pinned HGB model must return classes H/A/D")
    output = np.zeros_like(raw)
    for column, label in enumerate(classes):
        output[:, int(label)] = raw[:, column]
    return output


def viterbi_decode(probability: np.ndarray, period: np.ndarray, x_mask: np.ndarray,
                   switch_penalty: float) -> np.ndarray:
    """Symmetric 3-state decoder, with X and half resets."""
    p = np.asarray(probability, dtype=np.float64)
    period = np.asarray(period)
    xmask = np.asarray(x_mask, dtype=bool)
    if p.ndim != 2 or p.shape[1] != 3 or period.shape != (len(p),) or xmask.shape != (len(p),):
        raise ValueError("probabilities, period, and x mask are not aligned")
    if not math.isfinite(float(switch_penalty)) or switch_penalty < 0:
        raise ValueError("switch penalty must be finite and non-negative")
    active = np.isin(period, (1, 2))
    if (not np.isfinite(p[active]).all() or (p[active] < 0).any() or (p[active] > 1).any()
            or not np.allclose(p[active].sum(axis=1), 1.0, atol=1e-5, rtol=0)):
        raise ValueError("active decoder rows must be normalized H/A/D probabilities")
    output = np.full(len(period), X, dtype=np.int8)
    transition = np.full((3, 3), -float(switch_penalty), dtype=np.float64)
    np.fill_diagonal(transition, 0.0)
    emission = np.log(np.clip(p, 1e-12, 1.0))
    index = 0
    while index < len(period):
        if period[index] not in (1, 2) or xmask[index]:
            index += 1
            continue
        half = int(period[index])
        start = index
        while index < len(period) and period[index] == half and not xmask[index]:
            index += 1
        stop = index
        segment = emission[start:stop]
        score = np.empty((len(segment), 3), dtype=np.float64)
        back = np.zeros((len(segment), 3), dtype=np.int8)
        score[0] = segment[0]
        for row in range(1, len(segment)):
            candidates = score[row - 1][:, None] + transition
            back[row] = np.argmax(candidates, axis=0)
            score[row] = segment[row] + np.max(candidates, axis=0)
        state = int(np.argmax(score[-1]))
        output[stop - 1] = state
        for row in range(len(segment) - 1, 0, -1):
            state = int(back[row, state])
            output[start + row - 1] = state
    return output

