"""All Atlas entries remain visible; only real adapters are executable."""
from __future__ import annotations
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("image_models", "inverse_models", "research_models", "foundation_models", "extended_models")


def entries():
    return json.loads((ROOT/"catalog.json").read_text(encoding="utf-8"))["models"]


def resolve(query):
    text = str(query).strip().casefold()
    matches = [m for m in entries() if text in {str(m['num']),f"{m['num']:03d}",m['id'].casefold(),m['title'].casefold()}]
    if not matches:
        matches=[m for m in entries() if text == m['id'].split('_',3)[-1].casefold()]
    if len(matches)!=1:
        raise ValueError(f"Use an exact model name or number 001–137; query {query!r} did not identify one model")
    return matches[0]


def specs():
    result={}
    for name in MODULES:
        path=ROOT/"biomed_lab"/"models"/f"{name}.py"
        if not path.exists():
            continue
        module=importlib.import_module(f"biomed_lab.models.{name}")
        for num,spec in module.SPECS.items():
            if num in result:
                raise RuntimeError(f"Duplicate model adapter {num}")
            result[num]={**spec,"module":name}
    return result


def get_spec(query):
    m=resolve(query)
    spec=specs().get(m["num"])
    if spec is None:
        raise ValueError(f"{m['title']} has an original-source route and dataset, but no verified CPU adapter. Use `info {m['num']}`. No substitute model will be run.")
    return m,spec


def build(query, overrides=None):
    m,spec=get_spec(query)
    config=dict(spec.get("config",{}))
    for key,value in (overrides or {}).items():
        if key not in config:
            raise ValueError(f"Unsupported setting {key!r}. Supported: {', '.join(config) or '(none)'}")
        config[key]=value
    module=importlib.import_module(f"biomed_lab.models.{spec['module']}")
    return module.build_model(m["num"],config),m,spec,config


def source(query):
    m=resolve(query)
    rows=json.loads((ROOT/"sources.json").read_text(encoding="utf-8"))["models"]
    return next(x for x in rows if x['num']==m['num'])
