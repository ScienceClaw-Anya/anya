import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from neurotarget.brainvision import RangeReader, parse_header
from neurotarget.real_data import annex_identity, contact_annotations, unique_stimulation_sites


HEADER = """Brain Vision Data Exchange Header File Version 1.0
[Common Infos]
DataFile=example.eeg
DataFormat=BINARY
DataOrientation=MULTIPLEXED
NumberOfChannels=2
SamplingInterval=1000
[Binary Infos]
BinaryFormat=IEEE_FLOAT_32
[Channel Infos]
Ch1=A1,,2,uV
Ch2=A2,,1,uV
"""


def test_brainvision_parser_obeys_orientation_format_and_resolution():
    header = parse_header(HEADER)
    assert header.sfreq == 1000 and header.channels == ["A1", "A2"]
    assert header.resolutions.tolist() == [2., 1.]
    with pytest.raises(ValueError): parse_header(HEADER.replace("MULTIPLEXED", "VECTORIZED"))
    with pytest.raises(ValueError): parse_header(HEADER.replace("example.eeg", "../secret.eeg"))


def test_clinical_contact_ranges_preserve_exact_bipolar_names():
    matched, missing = contact_annotations(" lat1-2; LAT6-7; LAH6; F7-8 ", ["LAT1", "LAT2", "LAT6", "LAT7", "LAH6", "F7-8"])
    assert matched == {"LAT1", "LAT2", "LAT6", "LAT7", "LAH6", "F7-8"} and not missing
    assert contact_annotations("A1-3", ["A1", "A2"])[1] == ["A3"]
    with pytest.raises(ValueError): contact_annotations("A9-1", ["A1"])


def test_both_public_annex_checksum_types_are_preserved():
    assert annex_identity(b"../annex/objects/MD5E-s3--900150983cd24fb0d6963f7d28e17f72.xlsx") == ("md5", 3, hashlib.md5(b"abc").hexdigest())
    assert annex_identity(b"../annex/objects/SHA256E-s3--" + hashlib.sha256(b"abc").hexdigest().encode() + b".vhdr")[0] == "sha256"
    with pytest.raises(ValueError): annex_identity(b"../annex/objects/unknown-key")


def test_polarity_reversal_is_not_a_distinct_anatomical_site():
    assert unique_stimulation_sites(["TL01-TL02", "TL02-TL01", "TL02-TL03"]) == ["TL01-TL02", "TL02-TL03"]
    with pytest.raises(ValueError): unique_stimulation_sites(["A1-A2", "A2-A1"])


class Response:
    def __init__(self, status, headers, content=b""):
        import io
        self.status_code, self.headers = status, headers
        self.raw = io.BytesIO(content)
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError("HTTP failure")
    def close(self): pass


class Session:
    def __init__(self, bad=False):
        self.content = np.arange(8, dtype="<f4").tobytes()
        self.gets = 0; self.bad = bad
    def head(self, *args, **kwargs):
        return Response(200, {"Content-Length": str(len(self.content)), "ETag": '"version-a"'})
    def get(self, *args, **kwargs):
        self.gets += 1
        assert kwargs["headers"]["If-Match"] == '"version-a"'
        return Response(200 if self.bad else 206,
                        {"Content-Range": "bytes 8-23/32", "ETag": '"version-a"'}, self.content[8:24])


def test_range_reader_checks_version_exact_bytes_cache_and_scaling(tmp_path):
    session = Session()
    reader = RangeReader("https://example.org/recording.eeg", parse_header(HEADER), tmp_path, session)
    assert reader.read_samples(1, 2).tolist() == [[4., 3.], [8., 5.]]
    reader.read_samples(1, 2)
    assert session.gets == 1 and len(reader.ranges) == 2
    assert reader.ranges[0]["sha256"] == hashlib.sha256(session.content[8:24]).hexdigest()
    binary = next(tmp_path.glob("*.bin")); binary.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"): reader.read_samples(1, 2)


def test_full_file_fallback_is_rejected_before_reading(tmp_path):
    reader = RangeReader("https://example.org/recording.eeg", parse_header(HEADER), tmp_path, Session(bad=True))
    with pytest.raises(ValueError, match="exact"): reader.read_samples(1, 2)
    with pytest.raises(ValueError): reader.read_samples(-1, 2)
