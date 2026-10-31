"""Bounded, checksum-recorded reads of public multiplexed BrainVision recordings.

Only declared binary formats are supported. HTTP range and ETag checks prevent
silently downloading a multi-gigabyte recording or combining different versions.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import requests


@dataclass
class Header:
    data_file: str
    sfreq: float
    channels: list[str]
    resolutions: np.ndarray
    units: list[str]
    dtype: str


def parse_header(text):
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    start = text.find("[Common Infos]")
    if start < 0:
        raise ValueError("Missing BrainVision Common Infos")
    parser.read_string(text[start:])
    common = parser["Common Infos"]
    if common.get("DataFormat") != "BINARY" or common.get("DataOrientation") != "MULTIPLEXED":
        raise ValueError("Only binary multiplexed BrainVision recordings are supported")
    formats = {"IEEE_FLOAT_32": "<f4", "INT_16": "<i2", "INT_32": "<i4"}
    fmt = parser["Binary Infos"].get("BinaryFormat")
    if fmt not in formats:
        raise ValueError("Unsupported BrainVision binary format")
    channels, resolutions, units = [], [], []
    for i in range(1, int(common["NumberOfChannels"]) + 1):
        pieces = parser["Channel Infos"][f"Ch{i}"].split(",")
        channels.append(pieces[0].replace("\\1", ","))
        resolutions.append(float(pieces[2]) if len(pieces) > 2 and pieces[2] else 1.)
        units.append(pieces[3] if len(pieces) > 3 else "")
    name = common["DataFile"]
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("DataFile must be a neighboring filename")
    return Header(name, 1e6 / float(common["SamplingInterval"]), channels,
                  np.asarray(resolutions), units, formats[fmt])


class RangeReader:
    def __init__(self, url, header, cache, session=None):
        self.url, self.header, self.cache = url, header, Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        head = self.session.head(url, timeout=30, allow_redirects=False)
        head.raise_for_status()
        self.etag = head.headers.get("ETag")
        self.size = int(head.headers["Content-Length"])
        self.frame_bytes = len(header.channels) * np.dtype(header.dtype).itemsize
        if not self.etag or self.size % self.frame_bytes:
            raise ValueError("Missing version identifier or incomplete binary frames")
        self.ranges = []

    def read_samples(self, start, count):
        if start < 0 or count < 2 or (start + count) * self.frame_bytes > self.size:
            raise ValueError("Requested samples are outside the recording")
        first, size = start * self.frame_bytes, count * self.frame_bytes
        if size > 32 * 1024 * 1024:
            raise ValueError("A single range is limited to 32 MiB")
        last = first + size - 1
        identity = f"{self.url}|{self.etag}|{first}:{last}"
        key = hashlib.sha256(identity.encode()).hexdigest()
        binary, manifest = self.cache / (key + ".bin"), self.cache / (key + ".json")
        if binary.exists() and manifest.exists():
            content = binary.read_bytes()
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            if hashlib.sha256(content).hexdigest() != metadata["sha256"] or len(content) != size:
                raise ValueError("Cached real-data segment failed its checksum")
        else:
            response = self.session.get(self.url, headers={"Range": f"bytes={first}-{last}", "If-Match": self.etag},
                                        timeout=(15, 60), stream=True, allow_redirects=False)
            expected_range = f"bytes {first}-{last}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected_range or response.headers.get("ETag") != self.etag:
                response.close()
                raise ValueError("Server did not honor the exact, version-checked byte range")
            content = response.raw.read(size + 1)
            response.close()
            if len(content) != size:
                raise ValueError("Incomplete or oversized real-data range")
            metadata = {"url": self.url, "etag": self.etag, "first_byte": first, "last_byte": last,
                        "sha256": hashlib.sha256(content).hexdigest(), "bytes": size,
                        "integrity_scope": "This downloaded byte range, not a full-recording checksum"}
            binary.write_bytes(content)
            manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.ranges.append(metadata)
        values = np.frombuffer(content, dtype=self.header.dtype).reshape(count, len(self.header.channels)).astype(float)
        values *= self.header.resolutions
        if not np.isfinite(values).all():
            raise ValueError("Non-finite samples in real recording")
        return values

    def read_seconds(self, start, stop):
        a, b = round(start * self.header.sfreq), round(stop * self.header.sfreq)
        return self.read_samples(a, b-a)
