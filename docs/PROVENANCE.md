# Provenance

## Sources

| Component | Origin | Notes |
|-----------|--------|-------|
| CSFI-UNet / IGAB / Illum / Merge | `legacy_prototype/model/MainNetCSFIUNet.py` | Split into `csfi_unet.py` + `attention.py`; pad/unpad, soft inject init, `ref=None` |
| LTE / SearchTransfer / MeanShift | TTSR-lowlight family (`TTSR-lowlight-v1-data1SOTA/model/`) | Ported to `asrt/models/reference.py` |
| Warm-start converter | `legacy_prototype/scripts/warmstart_csfi_unet.py` | → `scripts/convert_legacy_weights.py`; skips old RGB tail / legacy illum heads |

## Not forked in-place

Old TTSR repos under `/root/projects/TTSR-*` are **not** patched. ASRT lives only at `/root/projects/ASRT`.

## Data paths (external; not in git)

- LOLv2 Real / Syn Train+Test under `/root/data/datasets/lol-v2-*`
- Nano refs: `nanobanana_ref(_v2/_v3)` beside Low/Normal
- Huawei/Nikon Nano: `/root/data/datasets/data1/Training data/{cam}/...`

Missing Nano → sample excluded from reference branch; never fill with GT.
