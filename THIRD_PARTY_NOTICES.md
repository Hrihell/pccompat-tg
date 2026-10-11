# Third-Party Notices

PCCompat-TG includes facts and derived values from the sources below. We matched each source's records to our own model IDs, aggregated them (medians, modes), and reformatted them. Each source below is used under its own license. None of the sources endorses PCCompat-TG.

All product names are trademarks of their respective owners. Their use here does not imply endorsement.

## Blender Open Data
Benchmark snapshot dated 2026-10-07, from https://opendata.blender.org/. Licensed under CC0 1.0 Universal. Attribution is given as a courtesy.

## GFXBench and CompuBench results (Kishonti Ltd)
Source: https://github.com/Kishonti-Opensource/GFXBench_and_CompuBench_results. Copyright (c) 2005-2025, Kishonti Ltd. The non-code assets are licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Changes: we kept desktop discrete-GPU rows only, matched them to PCCompat-TG model IDs, and took the median per model, test and API.
Attribution (as agreed with Kishonti Ltd by e-mail, 2026-10-08): "GFXBench/CompuBench results snapshot © Kishonti Ltd, https://github.com/Kishonti-Opensource/GFXBench_and_CompuBench_results, CC BY 4.0 (values aggregated and matched by the authors)."

## GPU Models dataset (GetDeploying)
Source: https://getdeploying.com/dataset/gpu-models. Licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The license covers the published dataset files only. Its use here implies no endorsement by GetDeploying.
Changes: we used consumer NVIDIA rows released 2020-10 or later, for cross-checking and for datasheet links.

## Linux Hardware Database (linuxhw)
Sources:
- https://github.com/linuxhw/DMI, commit 57b21bc6fdd863380e99c228c22aafcd1509528c
- https://github.com/linuxhw/LsPCI, commit 716d5be960ffcac591ebb3309361bdce17cf6079

Both are licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Changes: we publish only per-model aggregates (cache size modes and observed-combination counts). Individual reports are not redistributed.

## PCI ID Repository (pci.ids)
Source: https://pci-ids.ucw.cz/, version 2026.10.07. The copyright is held by Martin Mares and Albert Pool and covers the aggregation and formatting of the database. Used under the 3-clause BSD License, chosen from the file's GPL-2.0-or-later / BSD-3-Clause dual license:

> Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
> 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
> 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Wikipedia
Factual values were extracted from the English Wikipedia revisions listed in `field_provenance` (oldid links). No article text or table layout is reproduced.
