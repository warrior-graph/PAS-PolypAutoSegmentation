from .emcad import EMCADNet

MODELS = {"emcad": EMCADNet}


def build_model(cfg: dict, pretrained: bool | None = None):
    """Build a model from the `model` section of a config.

    `pretrained` overrides the config value (e.g. False when loading a checkpoint).
    """
    cfg = dict(cfg)
    name = cfg.pop("name")
    if pretrained is not None:
        cfg["pretrained"] = pretrained
    if "kernel_sizes" in cfg:
        cfg["kernel_sizes"] = tuple(cfg["kernel_sizes"])
    return MODELS[name](**cfg)
