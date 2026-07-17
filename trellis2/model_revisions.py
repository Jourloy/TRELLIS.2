"""Pinned Hugging Face inputs used by the supported inference CLIs."""

TRELLIS_REPO = "microsoft/TRELLIS.2-4B"
TRELLIS_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"

DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_REVISION = "ea8dc2863c51be0a264bab82070e3e8836b02d51"

RMBG_REPO = "briaai/RMBG-2.0"
RMBG_REVISION = "5df4c9c76d8170882c34f6986e848ee07fd0ba43"

TRELLIS_IMAGE_LARGE_REPO = "microsoft/TRELLIS-image-large"
TRELLIS_IMAGE_LARGE_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"

MODEL_REVISIONS = {
    TRELLIS_REPO: TRELLIS_REVISION,
    TRELLIS_IMAGE_LARGE_REPO: TRELLIS_IMAGE_LARGE_REVISION,
    DINOV3_REPO: DINOV3_REVISION,
    RMBG_REPO: RMBG_REVISION,
}

SOURCE_REVISIONS = {
    "pedronaugusto/mtlgemm": "867aec8234299a7fe1ede7f802c8debe5a939a82",
    "pedronaugusto/mtldiffrast": "4668cd91cb6d27f5e264731f94a06841fbf7aab8",
    "pedronaugusto/mtlbvh": "23f441c470ce1f537e1fd836f3ffb5b8245f7975",
    "pedronaugusto/mtlmesh": "212079e55772cff3d648a21372392c37e0643f3b",
    "EasternJournalist/utils3d": "9a4eb15e4021b67b12c460c7057d642626897ec8",
    "pedronaugusto/trellis2-apple": "6055b868734af6e12769d229d90580e775fae9f0",
    "shivampkumar/trellis-mac": "d58628f4f5b9c3de8274cb110074154f4b31cef2",
}


def revision_for_repo(repo_id, default=None):
    return MODEL_REVISIONS.get(repo_id, default)
