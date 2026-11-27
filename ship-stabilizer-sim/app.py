"""Akademik Tryoshnikov 2-axis stabilization simulator + automatic tuning (2nd independent app).

Modes
- Virtual: synthetic wave -> simplified ship motion -> simulated IMU -> complementary filter
           -> FF+PD -> servo model -> stabilized plate.
- Actual ACE voyage: automatically downloads the complete public daily INS archive and the
  reconstructed sea-state CSV from Zenodo, indexes all daily files, then replays every continuous
  segment in chronological order without asking the user to upload files one by one.
- Auto tuning (simulation-based random search + local search, stages 0-5: seeds and convergence,
  all candidates and Pareto front, stability_check and constrained objective, Kp x Kd grid,
  WMO sea-state binned tuning with transfer/regret matrices): searches FF, Kp, Kd and
  complementary-filter alpha using representative windows sampled across the daily INS files.
  Since stage 0 the tuner evaluates candidates with the SAME 50 Hz loop, the same linear
  interpolation and the same noise sequence as live playback (see simulate_batch and
  consistency_check). Servo parameters and assumed sensor-error settings are held fixed.

Important limitations
- ACE INS is uncorrected / not quality checked by the provider and was not synchronized with other
  onboard instrumentation as far as known.
- ACE INS is 1 Hz. Playback and tuning both interpolate between samples to run the 50 Hz model;
  interpolation does not recover unrecorded high-frequency ship motion.
- ACE Hs/Tp is reconstructed from five-minute ship-heave spectra using a ship-specific RAO; it is
  not a direct buoy measurement at every instant.
- Auto-tuned parameters are model recommendations, not certified settings for a physical servo.
  Verify them on the built hardware with real 50 Hz IMU and actuator measurements before use.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import sea_state as phys    # 물리 기반 선박 운동 모델 (가상 파도 모드의 선박 Roll·Pitch)
import verified_sim as vs   # 검증된 식만 담은 계산 엔진 (verify_engine.py 로 56개 시험 통과)

# -----------------------------------------------------------------------------
# Constants and paths
# -----------------------------------------------------------------------------
G = 9.81
CONTROL_DT = 0.02  # 50 Hz model integration
HISTORY_S = 30.0
SHIP = {
    "name": "R/V Akademik Tryoshnikov",
    "length": 133.53,
    "breadth": 23.25,
    "displacement": 16539,
    "draft": 8.5,
}
INS_URL = "https://zenodo.org/records/3733228/files/ace-inertial-navigation-csv.zip?download=1"
WAVE_URL = "https://zenodo.org/records/4541564/files/Reconstructed-Sea-State.csv?download=1"
INS_EXPECTED_SIZE = 646_925_085
INS_MD5 = "7778fe7e1a2472d0240022f47430e149"
WAVE_MD5 = "5fce25d202cf8588cc0963e2f0b2b835"
CACHE_DIR = Path(os.environ.get("AKADEMIK_CACHE_DIR", str(Path.home() / ".cache" / "akademik_2axis")))
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"                         # GitHub 저장소에 함께 올리는 경량 데이터 폴더
COMPACT_INDEX = DATA_DIR / "akademik_index.json"
REPO_WAVE_CSV = DATA_DIR / "Reconstructed-Sea-State.csv"
COMPACT_PART_BYTES = 18_000_000                     # 부분 파일 하나의 압축 전 크기 상한 18MB → 실제 파일은 이보다 작음 (GitHub 웹 업로드 한도 25MB)
COMPACT_ROW_BYTES = 12                              # 행당 시각 int32 4B + 롤 float32 4B + 피치 float32 4B
INS_ARCHIVE = CACHE_DIR / "ace-inertial-navigation-csv.zip"
WAVE_CSV = CACHE_DIR / "Reconstructed-Sea-State.csv"
MANIFEST_JSON = CACHE_DIR / "ace_manifest.json"

SHIP_COLOR = "#F28C38"
PLATE_COLOR = "#2EC4B6"
PLATE_OFF_COLOR = "#E4572E"

SEA_STATES = {
    "잔잔한 파도": dict(Hs=0.5, Tp=4.0, direction=0, components=3, shock=3.0, gain=0.6),
    "거친 파도": dict(Hs=2.0, Tp=7.0, direction=25, components=5, shock=7.0, gain=2.0),
    "폭풍 파도": dict(Hs=4.0, Tp=10.0, direction=45, components=7, shock=12.0, gain=3.5),
}

ROLL_ALIASES = ["platform_roll", "roll", "roll_deg"]
PITCH_ALIASES = ["platform_pitch", "pitch", "pitch_deg"]
DATE_ALIASES = ["pc_date_utc", "date_utc", "date"]
TIME_ALIASES = ["pc_time_hhmmssss", "pc_time_hhmmsssss", "pc_time_hhmmss", "pc_time", "time_utc", "time"]


# -----------------------------------------------------------------------------
# Download and cache the complete official data once per app runtime/cache folder
# -----------------------------------------------------------------------------
def md5_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


DOWNLOAD_LOCK = CACHE_DIR / "download.lock"
LOCK_STALE_S = 1800      # 30분 넘게 갱신되지 않은 잠금은 끊긴 다운로드로 보고 무시한다


def download_to_path(url: str, dest: Path, progress_cb: Callable[[int, int], None] | None = None,
                     expected_size: int | None = None, timeout: int = 240) -> None:
    """큰 Zenodo 파일을 이번 실행 전용 임시 파일에 받은 뒤 원자적으로 이름을 바꾼다.
    임시 파일 이름에 실행마다 다른 꼬리표를 붙여, 두 세션이 동시에 받아도 서로의 파일을 덮어쓰지 않는다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(f"{dest.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Akademik-2Axis-Stabilizer/2.0", "Accept": "*/*"},
    )
    downloaded = 0
    ok = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, part.open("wb") as out:
            total = int(response.headers.get("Content-Length") or expected_size or 0)
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                touch_lock()
                if progress_cb:
                    progress_cb(downloaded, total)
        if downloaded <= 0:
            raise IOError("서버에서 빈 파일을 받았습니다.")
        if expected_size and downloaded != expected_size:
            raise IOError(
                f"다운로드 크기가 예상과 다릅니다. 예상 {expected_size:,} bytes / 실제 {downloaded:,} bytes"
            )
        part.replace(dest)
        ok = True
    finally:
        # 화면 갱신 등으로 실행이 중단돼도(Streamlit 중단 신호는 Exception이 아님) 임시 파일을 지운다.
        if not ok:
            try:
                part.unlink(missing_ok=True)
            except Exception:
                pass


def acquire_lock() -> bool:
    """한 번에 하나의 다운로드만 돌도록 잠금 파일을 만든다. 이미 다른 세션이 받고 있으면 False."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(DOWNLOAD_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time():.0f}".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - DOWNLOAD_LOCK.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > LOCK_STALE_S:
                DOWNLOAD_LOCK.unlink(missing_ok=True)
                continue
            return False
    return False


def touch_lock() -> None:
    try:
        os.utime(DOWNLOAD_LOCK, None)
    except OSError:
        pass


def release_lock() -> None:
    try:
        DOWNLOAD_LOCK.unlink(missing_ok=True)
    except OSError:
        pass


def download_in_progress() -> dict | None:
    """다른 세션이 다운로드 중이면 진행 상황(받은 MB)을 돌려준다."""
    if not DOWNLOAD_LOCK.exists():
        return None
    try:
        age = time.time() - DOWNLOAD_LOCK.stat().st_mtime
    except FileNotFoundError:
        return None
    if age > LOCK_STALE_S:
        return None
    parts = list(CACHE_DIR.glob("*.part"))
    size = max((p.stat().st_size for p in parts if p.exists()), default=0)
    return {"mb": size / 1e6, "age_s": age}


def dataset_ready() -> bool:
    try:
        return (
            INS_ARCHIVE.is_file() and INS_ARCHIVE.stat().st_size >= 600_000_000
            and WAVE_CSV.is_file() and WAVE_CSV.stat().st_size >= 500_000
            and download_in_progress() is None
        )
    except OSError:
        return False


def fetch_verified(url: str, dest: Path, md5: str, label: str, status, expected_size: int | None = None,
                   min_size: int = 1, attempts: int = 2) -> None:
    """파일을 받고 MD5를 확인한다. 이미 있는 파일이 맞으면 다시 받지 않는다. 틀리면 자동으로 한 번 더 받는다."""
    stamp = dest.with_name(dest.name + ".md5ok")
    if dest.exists() and dest.stat().st_size >= min_size:
        sig = f"{dest.stat().st_size} {int(dest.stat().st_mtime)} {md5}"
        if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == sig:
            return                                  # 이전에 확인 완료 → 다시 계산하지 않음
        status.write(f"{label}: 저장된 파일의 체크섬을 확인합니다(처음 한 번만)…")
        got = md5_file(dest)
        if got == md5:
            stamp.write_text(sig, encoding="utf-8")
            return
        status.write(f"{label}: 저장된 파일의 체크섬이 다릅니다(받은 값 {got[:8]}…). 다시 받습니다.")
        dest.unlink(missing_ok=True)
    seen = []
    for attempt in range(1, attempts + 1):
        status.write(f"{label} 다운로드 중 (시도 {attempt}/{attempts})…")
        bar = status.progress(0.0)

        def cb(done, total, bar=bar):
            if total > 0:
                bar.progress(min(done / total, 1.0), text=f"{label}: {done/1e6:.0f} / {total/1e6:.0f} MB")
        download_to_path(url, dest, cb, expected_size)
        bar.progress(1.0, text=f"{label}: 다운로드 완료, 체크섬 확인 중…")
        got = md5_file(dest)
        if got == md5:
            status.write(f"{label}: 체크섬 일치 ✓")
            stamp.write_text(f"{dest.stat().st_size} {int(dest.stat().st_mtime)} {md5}", encoding="utf-8")
            return
        seen.append(got)
        dest.unlink(missing_ok=True)
        status.write(f"{label}: 체크섬 불일치(받은 값 {got}, 기대값 {md5}).")
    raise IOError(f"{label}의 MD5 검증에 {attempts}번 모두 실패했습니다. 받은 값 {', '.join(seen)} / 기대값 {md5}. "
                  "네트워크가 불안정하거나 Zenodo 파일이 바뀌었을 수 있습니다. 잠시 후 다시 시도해 주세요.")


def ensure_dataset(progress: st.delta_generator.DeltaGenerator | None = None) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    status = progress if progress is not None else st
    if not acquire_lock():
        info = download_in_progress()
        raise IOError("다른 화면(세션)에서 이미 다운로드 중입니다"
                      + (f" (현재 {info['mb']:.0f} MB)" if info else "")
                      + ". 동시에 받으면 파일이 섞이므로 기다렸다가 새로고침해 주세요.")
    try:
        fetch_verified(INS_URL, INS_ARCHIVE, INS_MD5, "① INS 전체 항해 ZIP(약 646.9MB)", status,
                       INS_EXPECTED_SIZE, min_size=600_000_000)
        fetch_verified(WAVE_URL, WAVE_CSV, WAVE_MD5, "② 파고·주기 재구성 CSV", status, None, min_size=500_000)
        if not zipfile.is_zipfile(INS_ARCHIVE):
            raise IOError("다운로드한 INS 파일이 정상적인 ZIP 압축파일이 아닙니다.")
    finally:
        release_lock()
    return INS_ARCHIVE, WAVE_CSV


# -----------------------------------------------------------------------------
# Dataset parsing / chronological manifest
# -----------------------------------------------------------------------------
def find_col(columns, aliases):
    mapping = {str(c).replace("\ufeff", "").strip().lower(): c for c in columns}
    for a in aliases:
        if a.lower() in mapping:
            return mapping[a.lower()]
    return None


def read_ins_csv(file_obj) -> pd.DataFrame:
    try:
        raw = pd.read_csv(file_obj, low_memory=False)
    except Exception as exc:
        raise ValueError(f"INS CSV 읽기 실패: {exc}") from exc
    raw.columns = [str(c).replace("\ufeff", "").strip() for c in raw.columns]
    roll_col = find_col(raw.columns, ROLL_ALIASES)
    pitch_col = find_col(raw.columns, PITCH_ALIASES)
    date_col = find_col(raw.columns, DATE_ALIASES)
    time_col = find_col(raw.columns, TIME_ALIASES)
    stamp_col = find_col(raw.columns, ["timestamp", "datetime", "datetime_utc"])
    if roll_col is None or pitch_col is None:
        raise ValueError("platform_roll / platform_pitch 열을 찾지 못했습니다.")
    if stamp_col is not None:
        timestamp = pd.to_datetime(raw[stamp_col], utc=True, errors="coerce")
    elif date_col is not None and time_col is not None:
        timestamp = pd.to_datetime(raw[date_col].astype(str).str.strip() + " " +
                                    raw[time_col].astype(str).str.strip(), utc=True, errors="coerce")
    else:
        raise ValueError("UTC 시간 열(pc_date_utc, pc_time_hhmmssss)을 찾지 못했습니다.")

    def numeric(aliases):
        c = find_col(raw.columns, aliases)
        if c is None:
            return np.full(len(raw), np.nan, dtype=float)
        return pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame({
        "timestamp": timestamp,
        "ship_roll": pd.to_numeric(raw[roll_col], errors="coerce"),
        "ship_pitch": pd.to_numeric(raw[pitch_col], errors="coerce"),
        "heading": numeric(["heading"]),
        "north_speed": numeric(["north_speed_ms-1"]),
        "east_speed": numeric(["east_speed_ms-1"]),
        "vertical_speed": numeric(["vertical_speed_ms-1"]),
    })
    out = out.dropna(subset=["timestamp", "ship_roll", "ship_pitch"])
    out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if len(out) < 10:
        raise ValueError(f"유효한 자세 기록이 {len(out)}개뿐입니다.")
    if (out[["ship_roll", "ship_pitch"]].abs() > 90).any().any():
        raise ValueError("Roll/Pitch가 ±90°를 초과합니다. 열 이름과 단위를 확인해 주세요.")
    return out


def read_wave_csv(path_or_file) -> pd.DataFrame:
    raw = pd.read_csv(path_or_file, low_memory=False)
    raw.columns = [str(c).replace("\ufeff", "").strip() for c in raw.columns]
    tcol = find_col(raw.columns, ["Date_Time", "datetime", "timestamp", "time"])
    hcol = find_col(raw.columns, ["Hs", "significant_wave_height"])
    pcol = find_col(raw.columns, ["Tp", "peak_period"])
    if not all([tcol, hcol, pcol]):
        raise ValueError("파랑 CSV에 Date_Time, Hs, Tp 열이 없습니다.")
    out = pd.DataFrame({
        "timestamp": pd.to_datetime(raw[tcol], utc=True, errors="coerce"),
        "Hs": pd.to_numeric(raw[hcol], errors="coerce"),
        "Tp": pd.to_numeric(raw[pcol], errors="coerce"),
    }).dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def epoch_seconds(series: pd.Series) -> np.ndarray:
    """UTC 시각 → epoch 초. pandas 버전·시간 해상도(ns/us)에 상관없이 같은 값을 준다.
    (astype("int64")/1e9는 pandas 3의 마이크로초 해상도에서 1000배 틀어진다.)"""
    base = pd.Timestamp("1970-01-01", tz="UTC")
    return ((series - base) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def split_continuous(df: pd.DataFrame):
    ts = epoch_seconds(df["timestamp"])
    dt = np.diff(ts)
    pos = dt[dt > 0]
    med = float(np.median(pos)) if len(pos) else 1.0
    gap_limit = max(2.5, med * 3.0)
    cuts = np.where(dt > gap_limit)[0] + 1
    bounds = np.r_[0, cuts, len(df)]
    result = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= 10:
            seg = df.iloc[a:b].copy().reset_index(drop=True)
            seg["elapsed_s"] = (seg["timestamp"] - seg["timestamp"].iloc[0]).dt.total_seconds()
            result.append(seg)
    return result, med, gap_limit


def list_ins_members(zf: zipfile.ZipFile) -> list[str]:
    pat = re.compile(r"ace_hydrins_\d{8}\.(?:csv|txt)$", re.IGNORECASE)
    return sorted([n for n in zf.namelist() if pat.search(Path(n).name)], key=lambda x: Path(x).name)


class ZipSource:
    """Zenodo 원본 ZIP(647MB)에서 일별 INS CSV를 읽는다."""
    kind = "zip"

    def __init__(self, path: Path):
        self.path = Path(path)

    def members(self) -> list[str]:
        with zipfile.ZipFile(self.path, "r") as zf:
            return list_ins_members(zf)

    def read_daily(self, member: str) -> pd.DataFrame:
        with zipfile.ZipFile(self.path, "r") as zf, zf.open(member, "r") as f:
            return read_ins_csv(f)

    def signature(self) -> str:
        st_ = self.path.stat()
        return f"zip|{st_.st_size}|{int(st_.st_mtime)}"


class CompactSource:
    """저장소 data/ 폴더의 경량 파일(롤·피치·시각만, 부분 파일 여러 개)에서 읽는다. 다운로드가 필요 없다."""
    kind = "compact"

    def __init__(self, index_path: Path = COMPACT_INDEX):
        self.dir = Path(index_path).parent
        self.index = json.loads(Path(index_path).read_text(encoding="utf-8"))

    def members(self) -> list[str]:
        return sorted(self.index["members"], key=lambda n: Path(n).name)

    def read_daily(self, member: str) -> pd.DataFrame:
        info = self.index["members"][member]
        chunks = info["chunks"] if "chunks" in info else [info]      # 하루치가 여러 부분 파일에 나뉘어 있을 수 있음
        ts, rs, ps = [], [], []
        for c in chunks:
            with np.load(self.dir / c["part"]) as z:
                ts.append(int(c["t0_cs"]) + z[c["key"] + "_t"].astype(np.int64))
                rs.append(z[c["key"] + "_r"].astype(float)); ps.append(z[c["key"] + "_p"].astype(float))
        off = np.concatenate(ts); roll = np.concatenate(rs); pitch = np.concatenate(ps)
        ns = off * 10_000_000
        nan = np.full(len(off), np.nan)
        return pd.DataFrame({"timestamp": pd.to_datetime(ns, utc=True), "ship_roll": roll, "ship_pitch": pitch,
                             "heading": nan, "north_speed": nan, "east_speed": nan, "vertical_speed": nan})

    def signature(self) -> str:
        return f"compact|{self.index.get('created', '')}|{self.index.get('rows', 0)}"


def compact_available() -> bool:
    try:
        idx = json.loads(COMPACT_INDEX.read_text(encoding="utf-8"))
        return all((DATA_DIR / part).is_file() for part in idx.get("parts", [])) and bool(idx.get("manifest"))
    except Exception:
        return False


def build_compact(src: ZipSource, manifest: list[dict], progress_slot=None) -> list[Path]:
    """원본 ZIP → data/akademik_partNN.npz + akademik_index.json (+ 파랑 CSV 복사).
    시각은 0.01초 단위 정수, 롤·피치는 float32로 저장한다. 같은 연속구간 분할(split_continuous)을 그대로 다시 쓴다.
    부분 파일은 압축 전 크기가 COMPACT_PART_BYTES 를 넘지 않게 나누며, 하루치가 커도 중간에서 잘라 여러 파일에 담는다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in DATA_DIR.glob("akademik_part*.npz"):
        old.unlink()
    members = src.members(); total = len(members)
    cap_rows = COMPACT_PART_BYTES // COMPACT_ROW_BYTES
    index = {"members": {}, "parts": [], "manifest": manifest, "rows": 0, "format": 2,
             "created": time.strftime("%Y-%m-%d %H:%M:%S"), "source_md5": INS_MD5,
             "note": "ACE INS (Thomas & Pina Estany 2020, Zenodo 10.5281/zenodo.3733228, CC BY 4.0)의 roll/pitch/시각만 추출"}
    buf, rows_in_part, part_no = {}, 0, 1

    def flush():
        nonlocal buf, rows_in_part, part_no
        if buf:
            name = f"akademik_part{part_no:02d}.npz"
            np.savez_compressed(DATA_DIR / name, **buf)
            index["parts"].append(name)
            buf, rows_in_part, part_no = {}, 0, part_no + 1
    for i, member in enumerate(members):
        try:
            d = src.read_daily(member)
        except ValueError:
            continue
        cs = np.round(epoch_seconds(d["timestamp"]) * 100).astype(np.int64)
        r = d["ship_roll"].to_numpy(np.float32); q = d["ship_pitch"].to_numpy(np.float32)
        chunks, a, j = [], 0, 0
        while a < len(cs):
            take = min(len(cs) - a, cap_rows - rows_in_part)
            key = f"m{i:04d}c{j:02d}"; b = a + take
            buf[key + "_t"] = (cs[a:b] - cs[a]).astype(np.int32)
            buf[key + "_r"] = r[a:b]; buf[key + "_p"] = q[a:b]
            chunks.append({"part": f"akademik_part{part_no:02d}.npz", "key": key, "t0_cs": int(cs[a]), "rows": int(take)})
            rows_in_part += take; a = b; j += 1
            if rows_in_part >= cap_rows:
                flush()
        index["members"][member] = {"chunks": chunks, "rows": int(len(cs))}
        index["rows"] += int(len(cs))
        if progress_slot is not None:
            progress_slot.progress((i + 1) / total, text=f"경량 파일 만드는 중: {i+1}/{total}일")
    flush()
    COMPACT_INDEX.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    if WAVE_CSV.exists():
        REPO_WAVE_CSV.write_bytes(WAVE_CSV.read_bytes())
    files = [COMPACT_INDEX, REPO_WAVE_CSV] + [DATA_DIR / p for p in index["parts"]]
    big = [f.name for f in files if f.exists() and f.stat().st_size > 24_000_000]
    if big:
        raise IOError(f"25MB 넘는 파일이 생겼습니다: {big}. COMPACT_PART_BYTES 를 줄여 주세요.")
    return files


def compact_zip_bytes() -> bytes:
    """data/ 폴더를 zip으로 묶어 내려받게 한다 (압축 풀어서 저장소의 data/ 에 그대로 올리면 된다)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for f in sorted(DATA_DIR.iterdir()):
            if f.is_file():
                zf.write(f, f"data/{f.name}")
    return buf.getvalue()


def build_manifest(archive_path, progress_slot=None) -> list[dict]:
    src = archive_path if hasattr(archive_path, "read_daily") else ZipSource(archive_path)
    if src.kind == "compact":
        return src.index["manifest"]
    cache_sig = {"archive_size": src.path.stat().st_size, "archive_mtime": int(src.path.stat().st_mtime)}
    if MANIFEST_JSON.exists():
        try:
            saved = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
            if saved.get("signature") == cache_sig and saved.get("segments"):
                return saved["segments"]
        except Exception:
            pass
    manifest = []
    members = src.members()
    if not members:
        raise ValueError("ZIP 안에서 ace_hydrins_YYYYMMDD.csv/txt 파일을 찾지 못했습니다.")
    total = len(members)
    if True:
        for i, member in enumerate(members, 1):
            daily = src.read_daily(member)
            segments, med, gap = split_continuous(daily)
            for seg_i, seg in enumerate(segments):
                manifest.append({
                    "member": member,
                    "segment_index": int(seg_i),
                    "start": seg["timestamp"].iloc[0].isoformat(),
                    "end": seg["timestamp"].iloc[-1].isoformat(),
                    "rows": int(len(seg)),
                    "duration_s": float(seg["elapsed_s"].iloc[-1]),
                    "median_dt": float(med),
                })
            if progress_slot is not None:
                progress_slot.progress(i / total, text=f"항해 날짜 인덱싱: {i}/{total}일 파일")
    if not manifest:
        raise ValueError("연속된 항해 데이터 구간을 찾지 못했습니다.")
    manifest.sort(key=lambda x: (x["start"], x["member"], x["segment_index"]))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps({"signature": cache_sig, "segments": manifest}, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_manifest(archive_path: Path) -> list[dict]:
    return build_manifest(archive_path, None)


def load_segment(archive_path, descriptor: dict) -> pd.DataFrame:
    src = archive_path if hasattr(archive_path, "read_daily") else ZipSource(archive_path)
    daily = src.read_daily(descriptor["member"])
    segs, _, _ = split_continuous(daily)
    idx = int(descriptor["segment_index"])
    if idx >= len(segs):
        raise ValueError(f"데이터 구간을 다시 찾지 못했습니다: {descriptor['member']}")
    seg = segs[idx]
    x = seg["elapsed_s"].to_numpy(dtype=float)
    seg["roll_rate"] = np.gradient(seg["ship_roll"].to_numpy(dtype=float), x)
    seg["pitch_rate"] = np.gradient(seg["ship_pitch"].to_numpy(dtype=float), x)
    return seg


def nearest_wave(timestamp, wave_df: pd.DataFrame | None, tolerance_s=180.0):
    if wave_df is None or wave_df.empty or timestamp is None or pd.isna(timestamp):
        return None
    target = pd.Timestamp(timestamp)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    sec = wave_df.attrs.get("_epoch")
    if sec is None or len(sec) != len(wave_df):
        sec = epoch_seconds(wave_df["timestamp"])
        wave_df.attrs["_epoch"] = sec
    delta = np.abs(sec - target.timestamp())
    i = int(np.argmin(delta))
    if float(delta[i]) > tolerance_s:
        return None
    row = wave_df.iloc[i]
    return {"Hs": float(row["Hs"]) if pd.notna(row["Hs"]) else np.nan,
            "Tp": float(row["Tp"]) if pd.notna(row["Tp"]) else np.nan,
            "timestamp": row["timestamp"], "delta_s": float(delta[i])}


# -----------------------------------------------------------------------------
# Motion and control models
# -----------------------------------------------------------------------------
WMO_CODES = [  # (코드, Hs 하한, Hs 상한, 이름) — WMO Sea State Code(Douglas 척도)
    (0, 0.0, 0.0, "고요(유리면)"), (1, 0.0, 0.1, "고요(잔물결)"), (2, 0.1, 0.5, "평온"), (3, 0.5, 1.25, "약간 거침"),
    (4, 1.25, 2.5, "보통"), (5, 2.5, 4.0, "거침"), (6, 4.0, 6.0, "매우 거침"), (7, 6.0, 9.0, "높음"),
    (8, 9.0, 14.0, "매우 높음"), (9, 14.0, math.inf, "극심"),
]


def classify_sea_state(hs) -> tuple[int | None, str]:
    """유의파고 Hs(m) → WMO 해상상태 코드 0~9. 자료가 없으면 (None, "자료 없음")."""
    try:
        hs = float(hs)
    except (TypeError, ValueError):
        return None, "자료 없음"
    if not np.isfinite(hs) or hs < 0:
        return None, "자료 없음"
    if hs == 0:
        return 0, WMO_CODES[0][3]
    for code, lo, hi, name in WMO_CODES[1:]:
        if hs <= hi:
            return code, name
    return 9, WMO_CODES[-1][3]


def wave_components(tp: float, hs: float, count: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    fp = 1.0 / tp
    freqs = np.linspace(max(0.04, fp * 0.45), fp * 2.0, count)
    phases = rng.uniform(0, 2 * np.pi, count)
    spectrum = freqs**-5 * np.exp(-1.25 * (fp / freqs)**4)
    df = freqs[1] - freqs[0] if count > 1 else 0.1
    amps = np.sqrt(2 * spectrum * df)
    amps = amps / max(np.sqrt(np.sum(amps**2) / 2), 1e-9) * (hs / 4.0)
    spread = np.deg2rad(np.linspace(-18, 18, count))
    return freqs, amps, phases, spread


def ocean_surface(t: float, comps, direction: float, grid=(48, 34)):
    freqs, amps, phases, spread = comps
    X, Y = np.meshgrid(np.linspace(-75, 75, grid[0]), np.linspace(-18, 18, grid[1]))
    Z = np.zeros_like(X)
    th0 = np.deg2rad(direction)
    for f, a, ph, sp in zip(freqs, amps, phases, spread):
        w = 2 * np.pi * f
        k = w**2 / G
        th = th0 + sp
        Z += a * np.cos(k * (X*np.cos(th) + Y*np.sin(th)) - w*t + ph)
    return X, Y, Z


def shock_value(tau, peak, freq=0.5, zeta=0.25):
    if tau < 0: return 0.0, 0.0
    w = 2*np.pi*freq
    A = peak / 0.73
    e = np.exp(-zeta*w*tau)
    return A*e*np.sin(w*tau), A*e*(w*np.cos(w*tau)-zeta*w*np.sin(w*tau))


PHYS_COMPONENTS = 48              # 물리 모델 파도 성분 수 (0.03~0.35 Hz, 롤 공진 0.074 Hz 포함)
PHYS_HEADINGS = (45.0, 90.0, 135.0)   # 한계 분석에서 실현마다 바꾸는 파향(선수 사파·정횡·선미 사파)


def _particle_acc(t, comps, direction, we=None, hull=None):
    """선박의 수평(전후·횡) 가속도 → 가속도계 오염.
    hull=None : 이전 가상 모델의 '수면 입자 가속도 × 0.5' 가정 (근거 약함 — 이전 모델 비교용으로 유지)
    hull=dict(L,B,T): 물리 모델. 긴 파도에서 선체가 물과 함께 움직인다는 Froude–Krylov 극한으로,
        입자 가속도 aω² 를 선체 부피에 대해 평균: sinc(k_x·L/2)·sinc(k_y·B/2)·(1−e^(−kT))/(kT)  [유도·근사]
        → 선체보다 짧은 파도의 가속도는 상쇄되어 거의 0, 긴 파도는 그대로 (부가질량·회절 무시)"""
    freqs, amps, phases, spread = comps
    th = np.deg2rad(direction) + np.asarray(spread)
    w = 2*np.pi*np.asarray(freqs); we = w if we is None else we
    tt = np.atleast_1d(np.asarray(t, float))[:, None]
    acc = np.asarray(amps)*w*w*np.sin(-we*tt + np.asarray(phases))
    if hull is None:
        return 0.5*(acc*np.cos(th)).sum(1), 0.5*(acc*np.sin(th)).sum(1)
    k = w*w/G; kT = np.maximum(k*hull["T"], 1e-9)
    fac = np.sinc(k*np.abs(np.cos(th))*hull["L"]/2/np.pi)*np.sinc(k*np.abs(np.sin(th))*hull["B"]/2/np.pi)*(1 - np.exp(-kT))/kT
    acc = acc*fac[None, :]
    return (acc*np.cos(th)).sum(1), (acc*np.sin(th)).sum(1)


def virtual_motion(t, comps, direction, gain, shocks, phys_terms=None, phys_co=None, imu_h=0.0):
    """가상 파도 한 시점의 선박 운동.
    phys_terms 가 있으면 물리 기반 운동방정식(sea_state.py)의 정상 응답, 없으면 이전 파경사 배율 모델."""
    if phys_terms is not None:
        r = phys.response(float(t), phys_terms)
        roll, pitch = r["roll"]*gain, r["pitch"]*gain
        rr, pr = r["roll_rate"]*gain, r["pitch_rate"]*gain
        ra, pa = math.radians(r["roll_acc"]*gain), math.radians(r["pitch_acc"]*gain)
        for s in shocks:   # 충격 → 선박 고유주기·감쇠비로 울리는 자유 진동 (관성·복원·감쇠가 바로 보이는 시험)
            ax_ = "roll" if s["axis"] == "roll" else "pitch"
            wn, z = (phys_co["w44"], phys_co["z44"]) if ax_ == "roll" else (phys_co["w55"], phys_co["z55"])
            v, rt = phys.impulse_response(t - s["t0"], s["peak"], wn, z)
            if ax_ == "roll": roll += float(v); rr += float(rt)
            else: pitch += float(v); pr += float(rt)
        axp, ayp = _particle_acc(t, comps, direction, phys_terms["we"], phys_co.get("hull"))
        # 레버암: IMU 가 회전 중심 위 h 에 있으면 접선 가속도 h·φ'' 가 가속도계에 더해진다 (롤 → +y, 피치 → −x)
        ax = float(axp[0]) - imu_h*pa; ay = float(ayp[0]) + imu_h*ra
        return {"roll": roll, "pitch": pitch, "roll_rate": rr, "pitch_rate": pr,
                "heave": r["heave"], "ax": ax, "ay": ay,
                "heading": np.nan, "north_speed": np.nan, "east_speed": np.nan, "timestamp": None}
    freqs, amps, phases, spread = comps
    th0 = np.deg2rad(direction)
    roll = pitch = rr = pr = heave = ax = ay = 0.0
    for f, a, ph, sp in zip(freqs, amps, phases, spread):
        w = 2*np.pi*f; k = w*w/G; th = th0+sp; ang = -w*t+ph
        heave += a*np.cos(ang)
        slope = -a*k*np.sin(ang); slope_rate = a*k*w*np.cos(ang)
        roll += slope*np.sin(th); pitch += slope*np.cos(th)
        rr += slope_rate*np.sin(th); pr += slope_rate*np.cos(th)
        acc = a*w*w*np.sin(ang); ax += acc*np.cos(th); ay += acc*np.sin(th)
    scale = np.rad2deg(0.75)*gain
    roll*=scale; pitch*=scale; rr*=scale; pr*=scale
    for s in shocks:
        v, r = shock_value(t-s["t0"], s["peak"])
        if s["axis"] == "roll": roll += v; rr += r
        else: pitch += v; pr += r
    return {"roll": roll, "pitch": pitch, "roll_rate": rr, "pitch_rate": pr,
            "heave": heave, "ax": 0.5*ax, "ay": 0.5*ay,
            "heading": np.nan, "north_speed": np.nan, "east_speed": np.nan, "timestamp": None}


_MOTION_CACHE: dict = {}
_MOTION_COLS = ("ship_roll", "ship_pitch", "roll_rate", "pitch_rate", "heading", "north_speed", "east_speed")


def _motion_arrays(df: pd.DataFrame) -> dict:
    """구간 DataFrame 의 열을 한 번만 numpy 배열로 바꿔 둔다 (매 스텝 전체 배열 변환을 피함 → 깜빡임 원인 제거)."""
    key = id(df)
    c = _MOTION_CACHE.get(key)
    if c is not None and c["n"] == len(df) and c["x0"] == float(df["elapsed_s"].iloc[0]):
        return c
    x = df["elapsed_s"].to_numpy(dtype=float)
    cols = {}
    for col in _MOTION_COLS:
        if col in df:
            v = df[col].to_numpy(dtype=float); good = np.isfinite(v)
            cols[col] = (x[good], v[good]) if good.sum() >= 2 else None
        else:
            cols[col] = None
    if len(_MOTION_CACHE) > 8:
        _MOTION_CACHE.clear()
    c = {"n": len(df), "x0": float(x[0]) if len(x) else 0.0, "x": x, "cols": cols}
    _MOTION_CACHE[key] = c
    return c


def actual_motion_at(t: float, df: pd.DataFrame):
    c = _motion_arrays(df)
    x = c["x"]
    i = int(np.clip(np.searchsorted(x, t, side="left"), 0, len(x)-1))
    def interp(col):
        e = c["cols"].get(col)
        if e is None: return np.nan
        xx, vv = e
        j = int(np.searchsorted(xx, t))                    # 이웃한 두 점만으로 선형 보간 = 전체 배열 np.interp 와 같은 값
        return float(np.interp(t, xx[max(j-1, 0):min(j+1, len(xx))], vv[max(j-1, 0):min(j+1, len(xx))]))
    return {"roll": interp("ship_roll"), "pitch": interp("ship_pitch"),
            "roll_rate": interp("roll_rate"), "pitch_rate": interp("pitch_rate"),
            "heave": 0.0, "ax": 0.0, "ay": 0.0,
            "heading": interp("heading"), "north_speed": interp("north_speed"),
            "east_speed": interp("east_speed"), "timestamp": df["timestamp"].iloc[i]}


def new_sim_state(initial=(0.0, 0.0), seed=7):
    """verified_sim.new_state 와 같은 상태(같은 난수 순서) + 화면용 기록."""
    st_ = vs.new_state(initial, seed)
    st_["history"] = []; st_["shocks"] = []
    return st_


def expm_small(M: np.ndarray) -> np.ndarray:
    """작은 정방행렬의 행렬 지수(스케일링·제곱 + 테일러 20차). numpy만 사용."""
    norm = float(np.abs(M).sum(axis=1).max())
    k = max(0, int(math.ceil(math.log2(norm))) + 1) if norm > 0.5 else 0
    A = M / (2.0 ** k)
    E = np.eye(M.shape[0]); term = np.eye(M.shape[0])
    for i in range(1, 21):
        term = term @ A / i
        E = E + term
    for _ in range(k):
        E = E @ E
    return E


_SERVO2_CACHE: dict = {}


def servo2_coeffs(wn: float, zeta: float, dt: float = CONTROL_DT) -> tuple:
    """2차 서보 j'' = wn²(cmd - j) - 2ζ·wn·j' 를 한 제어 주기(dt) 동안 명령 고정(ZOH)으로 정확히 푼 계수.
    [j, v](k+1) = [[a11, a12], [a21, a22]]·[j, v](k) + [b1, b2]·cmd"""
    key = (round(float(wn), 12), round(float(zeta), 12), dt)
    if key not in _SERVO2_CACHE:
        M = np.array([[0.0, 1.0, 0.0], [-wn*wn, -2*zeta*wn, wn*wn], [0.0, 0.0, 0.0]]) * dt
        E = expm_small(M)
        _SERVO2_CACHE[key] = (E[0, 0], E[0, 1], E[1, 0], E[1, 1], E[0, 2], E[1, 2])
    return _SERVO2_CACHE[key]


def engine_params(p: dict) -> dict:
    """앱 파라미터 dict → verified_sim 파라미터. servo2(이전 키)도 받아 준다."""
    q = vs.params(**{k: v for k, v in p.items() if k in vs.DEFAULTS})
    if p.get("servo2"):
        q["servo_model"] = "zoh2"
    return q


def control_step(state, motion, p, controller_on=True, accel_disturb=0.0, dt=CONTROL_DT):
    """한 제어 주기. 계산은 전부 verified_sim.step (식별 출처는 그 파일 머리말과 수식_출처_원문캡처.pdf)."""
    row = vs.step(state, motion, engine_params(p), controller_on, accel_disturb, dt)
    row.update({"heave": motion.get("heave", 0.0), "timestamp": motion.get("timestamp"),
                "heading": motion.get("heading", np.nan), "north_speed": motion.get("north_speed", np.nan),
                "east_speed": motion.get("east_speed", np.nan)})
    return row


# -----------------------------------------------------------------------------
# Auto tuning: simulation-based random search + local search, evaluated with the live 50 Hz loop
# -----------------------------------------------------------------------------
TUNE_KEYS = ["ff", "kp", "kd", "alpha"]
# Kd 상한: 4단계 Kp×Kd 격자에서 확인한 안정 경계(기본 서보 τ=0.10s에서 Kd≈0.07부터 진동) 바로 위까지만 탐색한다.
# 그 이상은 기본 조건에서도 리밋사이클이므로 탐색할 이유가 없다. 경계 근처는 로그 스케일 샘플링으로 촘촘히 본다.
KD_MAX = 0.12
TUNE_BOUNDS = {"ff": (0.0, 1.2), "kp": (0.0, 4.0), "kd": (0.0, KD_MAX), "alpha": (0.90, 0.999)}
KD_LOG_MIN = 0.002
CASES = {
    "① FF·Kp·Kd·α 전부 튜닝": {},
    "② FF=1 고정, PD+α만 튜닝": {"ff": 1.0},
    "③ FF=0 (PD만 사용), PD+α만 튜닝": {"ff": 0.0},
}
OBJ_WEIGHTED = "가중합 방식 (기존 방식, effort 재정의)"
OBJ_CONSTRAINED = "제약 조건 방식 (포화율 ≤1%, 리밋사이클 없음)"
OBJECTIVES = [OBJ_CONSTRAINED, OBJ_WEIGHTED]

SEARCH_WINDOWS = 30            # 탐색 단계에서 쓰는 학습용 관찰 구간 수(무작위 추출, 시드 고정)
SUBSET_SEED = 20261120         # 탐색용 학습용 관찰 구간 추출 시드(+ 탐색 시드)
NOISE_SEED_BASE = 2026         # 관찰 구간 i의 기본 센서 노이즈 시드 = NOISE_SEED_BASE + i
VALID_NOISE_SEEDS = 3          # 검증 평가에 쓰는 노이즈 시드 수(관찰 구간마다)
SEED_STRIDE = 7919             # 같은 관찰 구간의 r번째 노이즈 시드 = 기본 시드 + SEED_STRIDE*r
CONSISTENCY_TOL = 0.05         # 실시간 재생과 튜닝 평가의 허용 차이(5%)
SERIES_BUDGET = 16_000_000     # p95 계산용 시계열 저장 상한(원소 수, 약 64 MB)

# 리밋사이클·고주파 명령 판정 (임계값은 기본 조건의 안정/진동 사례로 보정)
HF_CUTOFF_HZ = 2.0             # 이 주파수 이상을 고주파 명령으로 본다
LC_HF_RATIO = 0.30             # 관절각 분산 중 2Hz 이상 성분 비율이 이 값을 넘고
LC_HF_RMS = 0.30               # 2Hz 이상 성분 RMS가 이 값(°)을 넘으면 리밋사이클
LC_FLIP_HZ = 3.0               # 또는 관절 각속도 부호 변화로 본 진동 주파수가 이 값(Hz)을 넘으면 리밋사이클
FLIP_DEADBAND = 5.0            # 부호 변화 계산에서 무시할 작은 각속도(°/s, 노이즈 떨림 제외)
SAT_CONSTRAINT = 0.01          # 제약 조건 방식: 포화율 상한
PENALTY = 5.0                  # 제약 위반 페널티
STAB_GAIN_FACTORS = (1.5, 2.0)
STAB_TAU_FACTORS = (0.5, 1.5)
STAB_SAT_MAX = 0.05
STAB_TOP_K = 6                 # 안정성 검사를 한 번에 할 상위 후보 수
STAB_MAX_CHECK = 48            # 통과 후보가 나올 때까지 순위를 내려가며 검사할 최대 후보 수
STAB_WINDOWS = 8               # 안정성 검사에 쓰는 관찰 구간 수


def prior_tilt_rms(seg: pd.DataFrame, start_idx: int) -> dict:
    """관찰 구간 시작 직전 L분(2~5분) 동안의 선박 기울기 RMS(√(roll²+pitch²)). 하부 IMU가 측정할 수 있는 지표에 해당."""
    el = seg["elapsed_s"].to_numpy(dtype=float)
    r = seg["ship_roll"].to_numpy(dtype=float); p = seg["ship_pitch"].to_numpy(dtype=float)
    t0 = el[start_idx]
    out = {}
    for L in (2, 3, 4, 5):
        m = (el >= t0 - 60*L) & (el < t0)
        out[L] = float(np.sqrt(np.mean(r[m]**2 + p[m]**2))) if m.sum() >= 30*L else np.nan
    return out


def label_window(it: dict, wave_df: pd.DataFrame | None) -> None:
    """5-2: 관찰 구간 시작·중앙·끝 시각의 Hs/Tp를 nearest_wave로 붙이고 WMO 코드를 정한다."""
    df = it["df"]
    ts = df["timestamp"]
    picks = [ts.iloc[0], ts.iloc[len(ts)//2], ts.iloc[-1]]
    found = [nearest_wave(x, wave_df) for x in picks]
    ok = [f for f in found if f is not None and np.isfinite(f["Hs"])]
    it["date"] = ts.iloc[0].strftime("%Y-%m-%d")
    it["start_time"] = ts.iloc[0].strftime("%H:%M:%S")
    it["roll_rms"] = float(np.sqrt(np.mean(df["ship_roll"].to_numpy(float)**2)))
    it["pitch_rms"] = float(np.sqrt(np.mean(df["ship_pitch"].to_numpy(float)**2)))
    it["wave_ok"] = bool(found[1] is not None and len(ok) >= 2)
    if it["wave_ok"]:
        hs = np.array([f["Hs"] for f in ok]); tp = np.array([f["Tp"] for f in ok], dtype=float)
        it["Hs"] = float(np.mean(hs)); it["Tp"] = float(np.nanmean(tp)) if np.isfinite(tp).any() else np.nan
        it["delta_max"] = float(max(f["delta_s"] for f in ok))
        codes = {classify_sea_state(h)[0] for h in hs}
        it["wmo"], it["wmo_name"] = classify_sea_state(it["Hs"])
        it["boundary"] = len(codes) > 1
    else:
        it["Hs"] = np.nan; it["Tp"] = np.nan; it["delta_max"] = np.nan
        it["wmo"], it["wmo_name"] = None, "파랑 자료 없음"; it["boundary"] = False


def make_tuning_windows(archive_path: Path, manifest: list[dict], wave_df: pd.DataFrame | None = None,
                        max_per_day=2, window_rows=240, max_total_windows=180) -> dict:
    """대표 4분 관찰 구간을 날짜 전체에서 균등하게 추출한다.
    이전 버전은 앞 날짜부터 읽다가 max_total_windows에 도달하면 멈춰서 뒤쪽 날짜가 빠졌다.
    이제는 사용할 날짜 수(max_total_windows/max_per_day)만큼 전체 기간에서 균등 간격으로 고른다."""
    unique_members = sorted(set(d["member"] for d in manifest), key=lambda name: Path(name).name)
    member_rank = {member: i for i, member in enumerate(unique_members)}
    max_days = max(1, max_total_windows // max_per_day)
    if len(unique_members) > max_days:
        pick = sorted(set(np.linspace(0, len(unique_members)-1, max_days).round().astype(int)))
        chosen = [unique_members[i] for i in pick]
    else:
        chosen = list(unique_members)
    items = []
    src = archive_path if hasattr(archive_path, "read_daily") else ZipSource(archive_path)
    if True:
        for member in chosen:
            try:
                daily = src.read_daily(member)
            except ValueError:
                continue
            segments, _, _ = split_continuous(daily)
            candidates = []
            for seg in segments:
                n = len(seg)
                if n < 60:
                    continue
                length = min(window_rows, n)
                possible = sorted(set(int(x) for x in np.linspace(0, max(0, n - length), min(max_per_day, 3))))
                for start_idx in possible:
                    window = seg.iloc[start_idx:start_idx + length].copy().reset_index(drop=True)
                    if len(window) < 60:
                        continue
                    elapsed = window["elapsed_s"].to_numpy(dtype=float)
                    window["roll_rate"] = np.gradient(window["ship_roll"].to_numpy(dtype=float), elapsed)
                    window["pitch_rate"] = np.gradient(window["ship_pitch"].to_numpy(dtype=float), elapsed)
                    candidates.append((window, prior_tilt_rms(seg, start_idx)))
            if len(candidates) > max_per_day:
                sel = np.linspace(0, len(candidates) - 1, max_per_day).round().astype(int)
                candidates = [candidates[i] for i in sorted(set(sel))]
            is_validation = (member_rank[member] % 5 == 0)
            for window, prior in candidates:
                items.append({"df": window, "member": member, "validation": is_validation, "prior": prior})
    if not items:
        raise ValueError("자동 튜닝을 위한 1분 이상 연속 구간을 만들지 못했습니다.")
    n_val = sum(it["validation"] for it in items)
    if len(items) - n_val < 2 or n_val < 1:   # 작은 자료: 관찰 구간 단위로 나눈다
        for i, it in enumerate(items):
            it["validation"] = (i % 5 == 0)
    for gi, it in enumerate(items):
        it["id"] = gi
        it["seed"] = NOISE_SEED_BASE + gi
        label_window(it, wave_df)
    return {"items": items, "days_used": len(set(it["member"] for it in items)), "days_chosen": len(chosen),
            "days_total": len(unique_members), "n_windows": len(items)}


def window_inputs(df: pd.DataFrame) -> dict:
    """관찰 구간을 실시간 재생과 똑같은 50 Hz 입력으로 만든다.
    - 시간축: live_panel처럼 t=0에서 시작해 CONTROL_DT를 계속 더하고, t < 구간 길이일 때만 스텝을 진행한다.
    - 보간: actual_motion_at과 같은 np.interp(선형 보간)를 roll, pitch, roll_rate, pitch_rate에 적용한다."""
    x = df["elapsed_s"].to_numpy(dtype=float)
    x = x - x[0]
    roll = df["ship_roll"].to_numpy(dtype=float)
    pitch = df["ship_pitch"].to_numpy(dtype=float)
    rr = df["roll_rate"].to_numpy(dtype=float) if "roll_rate" in df else np.gradient(roll, x)
    pr = df["pitch_rate"].to_numpy(dtype=float) if "pitch_rate" in df else np.gradient(pitch, x)
    duration = float(x[-1])
    ts = []
    t = 0.0
    while t < duration:
        ts.append(t)
        t += CONTROL_DT
    t = np.asarray(ts if ts else [0.0], dtype=float)

    def interp(v):
        good = np.isfinite(v)
        return np.interp(t, x[good], v[good]) if good.sum() >= 2 else np.full_like(t, np.nan)
    return {"t": t, "ship": np.stack([interp(roll), interp(pitch)]), "rate": np.stack([interp(rr), interp(pr)]),
            "init": (float(roll[0]), float(pitch[0])), "duration": duration}


def noise_for(seed: int, n: int):
    """new_sim_state(seed)와 control_step이 뽑는 난수를 같은 순서로 미리 만든다.
    bias[축][하부, 상판], z[스텝][축][gyro_b, acc_b, gyro_p, acc_p]"""
    rng = np.random.default_rng(seed)
    bias = np.array([rng.normal(0, 1) for _ in range(4)]).reshape(2, 2)
    z = rng.normal(0.0, 1.0, size=(n, 2, 4))
    return bias, z


def highpass_coeffs(fc: float, fs: float):
    """2차 버터워스 고역통과 필터 계수(쌍선형 변환)."""
    K = math.tan(math.pi * fc / fs)
    q = 1 / math.sqrt(2)
    norm = 1 / (1 + K/q + K*K)
    return norm, -2*norm, norm, 2*(K*K - 1)*norm, (1 - K/q + K*K)*norm


HP_COEF = highpass_coeffs(HF_CUTOFF_HZ, 1 / CONTROL_DT)


def simulate_batch(inputs: list[dict], seeds: list[int], cands: list[dict], fixed: dict, progress_cb=None,
                   keep_p95: bool = True, warmup_s: float = 0.0) -> dict:
    """여러 후보 × 여러 관찰 구간을 한 번에 50 Hz로 시뮬레이션한다.
    식은 control_step(=verified_sim.step)과 한 줄씩 같고, 노이즈 난수 순서도 같다(관찰 구간마다 seeds[w]).
    같음은 verify_engine.py V13 과 앱의 consistency_check 로 확인한다.
    후보 dict에 "tau"가 있으면 그 후보만 서보 시간상수를 바꿔 계산한다(안정성 검사용).
    warmup_s: 처음 몇 초(필터·바이어스 추정이 수렴하는 동안)를 지표에서 뺀다. 튜닝은 0, 보정 한계 분석은 LIMIT_WARMUP_S.
    반환: 지표 이름 -> (후보 C, 관찰 구간 W) 배열"""
    W, C = len(inputs), len(cands)
    lens = np.array([inp["t"].size for inp in inputs])
    N = int(lens.max())
    ship = np.zeros((N, 2, W)); rate = np.zeros((N, 2, W)); tt = np.zeros((N, W)); valid = np.zeros((N, W), bool)
    noise = np.zeros((N, 2, 4, W)); bias = np.zeros((2, 2, W)); init = np.zeros((2, W)); acc_err = np.zeros((N, 2, W))
    for w, (inp, sd) in enumerate(zip(inputs, seeds)):
        n = int(lens[w])
        ship[:n, :, w] = inp["ship"].T; ship[n:, :, w] = inp["ship"][:, -1]
        rate[:n, :, w] = inp["rate"].T
        tt[:n, w] = inp["t"]; valid[:n, w] = True
        if "acc" in inp:            # 가상 파도: 선박 가속도에 의한 가속도계 오염각
            acc_err[:n, :, w] = inp["acc"].T
        b, z = noise_for(int(sd), n)
        bias[:, :, w] = b; noise[:n, :, :, w] = z
        init[:, w] = inp["init"]
    ad = float(fixed.get("accel_disturb", 0.0))
    if ad > 0:  # control_step과 같은 가속도 오염 식
        acc_err[:, 0] = acc_err[:, 0] + ad*np.sin(2*np.pi*0.23*tt)
        acc_err[:, 1] = acc_err[:, 1] + ad*np.sin(2*np.pi*0.19*tt+0.7)

    def col(k, default=None):
        return np.array([c.get(k, default) for c in cands], dtype=float)[None, :, None]
    ff, kp, kd, alpha = col("ff"), col("kp"), col("kd"), col("alpha")
    ki = col("ki", float(fixed.get("ki", 0.0)))         # B9 자이로 바이어스 추정 이득 (0이면 끔)
    W0 = int(round(float(warmup_s) / CONTROL_DT))
    tau = col("tau", float(fixed["tau"]))
    servo2 = bool(fixed.get("servo2", False)) or fixed.get("servo_model") == "zoh2"
    model = "zoh2" if servo2 else fixed.get("servo_model", "zoh1")
    legacy = fixed.get("timing") == "legacy"          # 기존 앱 재현용: 관절각[k+1]과 선박각[k]를 짝지어 기록
    D = int(fixed.get("delay_steps", 0))               # 센서→구동 지연(샘플)
    a_zoh = np.exp(-CONTROL_DT / tau)                  # 1차 서보 정확 이산화 a = e^(−dt/τ)
    if servo2:  # 후보별 τ 변형(안정성 검사)은 2차 모델에서 고유진동수를 τ에 반비례해 바꾼다
        wn_c = float(fixed["wn"]) * (float(fixed["tau"]) / tau)
        zeta_c = float(fixed["zeta"])
        coef = np.array([servo2_coeffs(float(w), zeta_c) for w in wn_c.ravel()])   # (C, 6)
        s11, s12, s21, s22, sb1, sb2 = (coef[:, i][None, :, None] for i in range(6))   # 서보 계수(고역통과 b1,b2와 이름 분리)
    dt = CONTROL_DT
    rlim, limit = float(fixed["rate"]), float(fixed["limit"])
    gn, gbias, an = float(fixed["gyro_noise"]), float(fixed["gyro_bias"]), float(fixed["acc_noise"])
    b0, b1, b2, a1, a2 = HP_COEF
    shape = (2, C, W)
    base_est = np.broadcast_to(init[:, None, :], shape).copy()
    plate_est = base_est.copy()
    joint = np.zeros(shape); jrate = np.zeros(shape); jvel = np.zeros(shape); u_prev = np.zeros(shape)
    qbuf = np.zeros((D,) + shape) if D > 0 else None; qpos = 0
    bb = bias[:, 0][:, None, :]; pb = bias[:, 1][:, None, :]
    bh_b = np.zeros(shape); bh_p = np.zeros(shape)      # 바이어스 추정값 b̂ (하부, 상판)
    ss = np.zeros(shape); mx = np.zeros(shape); sat = np.zeros((C, W))
    jsum = np.zeros(shape); jss = np.zeros(shape); jr_ss = np.zeros(shape); ja_ss = np.zeros(shape)
    hx1 = np.zeros(shape); hx2 = np.zeros(shape); hy1 = np.zeros(shape); hy2 = np.zeros(shape); hp_ss = np.zeros(shape)
    flips = np.zeros(shape); last_sgn = np.zeros(shape)
    series = np.empty((N, 2, C, W), dtype=np.float32) if keep_p95 else None
    for k in range(N):
        sh = ship[k][:, None, :]; sr = rate[k][:, None, :]; ae = acc_err[k][:, None, :]; z = noise[k]
        plate_before = sh + joint
        if legacy or model == "euler1":
            jdot = jrate                                   # 직전 스텝 평균 관절속도 (기존 앱)
        elif model == "zoh1":
            jdot = (u_prev - joint) / tau                  # 1차계 순간 관절속도 (정확)
        else:
            jdot = jvel                                    # 2차계 속도 상태 (정확)
        plate_rate = sr + jdot
        gyro_b = sr + gbias*bb + gn*z[:, 0][:, None, :] - bh_b      # ω_c = ω − b̂[k−1]
        acc_b = sh + ae + an*z[:, 1][:, None, :]
        base_est = alpha*(base_est + gyro_b*dt) + (1-alpha)*acc_b
        bh_b = bh_b - ki*(acc_b - base_est)*dt
        gyro_p = plate_rate + gbias*pb + gn*z[:, 2][:, None, :] - bh_p
        acc_p = plate_before + ae + an*z[:, 3][:, None, :]
        plate_est = alpha*(plate_est + gyro_p*dt) + (1-alpha)*acc_p
        bh_p = bh_p - ki*(acc_p - plate_est)*dt
        cmd = np.clip(-ff*base_est - kp*plate_est - kd*gyro_p, -limit, limit)
        if D > 0:                                          # u[k] = cmd[k−D]
            u = qbuf[qpos].copy(); qbuf[qpos] = cmd; qpos = (qpos + 1) % D
        else:
            u = cmd
        if servo2:
            new_joint = s11*joint + s12*jvel + sb1*u
            jvel = s21*joint + s22*jvel + sb2*u
            jvel = np.clip(jvel, -rlim, rlim)
            new_joint = np.clip(new_joint, -limit, limit)
            jvel = np.where((np.abs(new_joint) >= limit) & (new_joint*jvel > 0), 0.0, jvel)
        else:
            gain = (1.0 - a_zoh) if model == "zoh1" else dt/tau
            d = np.clip(gain*(u - joint), -rlim*dt, rlim*dt)
            new_joint = np.clip(joint + d, -limit, limit)
        new_rate = (new_joint - joint)/dt
        jacc = (new_rate - jrate)/dt
        jrate = new_rate
        rec_joint = new_joint if legacy else joint         # 기록 시점: 같은 시각의 관절각(기본) / 기존 앱 방식
        joint = new_joint
        u_prev = u
        plate = sh + rec_joint
        vk = valid[k] & (k >= W0)
        m = vk[None, None, :]
        ss += np.where(m, plate*plate, 0.0)
        np.maximum(mx, np.where(m, np.abs(plate), 0.0), out=mx)
        sat += ((np.abs(joint) >= limit*0.999).any(axis=0) & vk[None, :])
        jsum += np.where(m, joint, 0.0); jss += np.where(m, joint*joint, 0.0)
        jr_ss += np.where(m, jrate*jrate, 0.0); ja_ss += np.where(m, jacc*jacc, 0.0)
        hy = b0*joint + b1*hx1 + b2*hx2 - a1*hy1 - a2*hy2
        hx2 = hx1; hx1 = joint; hy2 = hy1; hy1 = hy
        hp_ss += np.where(m, hy*hy, 0.0)
        sgn = np.where(np.abs(jrate) > FLIP_DEADBAND, np.sign(jrate), 0.0)
        flips += np.where(m & (sgn != 0) & (last_sgn != 0) & (sgn != last_sgn), 1.0, 0.0)
        last_sgn = np.where(sgn != 0, sgn, last_sgn)
        if keep_p95:
            series[k] = np.abs(plate)
        if progress_cb and k % 600 == 0:
            progress_cb(k / N)
    n = np.maximum(lens - W0, 1)[None, None, :].astype(float)
    if keep_p95:
        p95 = np.empty(shape)
        for w in range(W):
            p95[:, :, w] = np.percentile(series[min(W0, lens[w]-1):lens[w], :, :, w], 95, axis=0)
    else:
        p95 = np.full(shape, np.nan)
    _sh = [inp["ship"][:, min(W0, inp["ship"].shape[1]-1):] for inp in inputs]
    ship_rms = np.stack([np.sqrt(np.mean(x**2, axis=1)) for x in _sh], axis=1)   # (2, W)
    ship_p95 = np.stack([np.percentile(np.abs(x), 95, axis=1) for x in _sh], axis=1)
    ship_max = np.stack([np.max(np.abs(x), axis=1) for x in _sh], axis=1)
    rms_out = np.sqrt(ss / n)
    srms, sp95, smax = (np.maximum(v, 0.1)[:, None, :] for v in (ship_rms, ship_p95, ship_max))
    roll_ratio, pitch_ratio = rms_out[0]/srms[0], rms_out[1]/srms[1]
    rms_ratio = 0.5*(roll_ratio + pitch_ratio)
    p95_ratio = 0.5*(p95[0]/sp95[0] + p95[1]/sp95[1])
    max_ratio = 0.5*(mx[0]/smax[0] + mx[1]/smax[1])
    # 3단계 effort 재정의: 필요한 관절 각도 자체가 아니라 '움직임의 격렬함'에 벌점을 준다.
    rate_rms = np.sqrt(jr_ss / n).mean(axis=0) / rlim                       # 관절 각속도 RMS / 최대 각속도
    acc_rms = np.sqrt(ja_ss / n).mean(axis=0) / (rlim / tau[0])             # 관절 각가속도 RMS / (최대 각속도/τ)
    var = np.maximum(jss/n - (jsum/n)**2, 1e-12)
    hf_axis = np.clip((hp_ss/n) / var, 0.0, 1.0)
    hf_rms_axis = np.sqrt(hp_ss/n)
    flip_axis = flips / (n*dt) / 2.0                                        # 부호 변화 2번 = 진동 1주기
    hf = hf_axis.max(axis=0)
    flip_hz = flip_axis.max(axis=0)
    lc = (((hf_axis > LC_HF_RATIO) & (hf_rms_axis > LC_HF_RMS)) | (flip_axis > LC_FLIP_HZ)).any(axis=0).astype(float)
    saturation = sat / n[0]
    score_w = (0.45*rms_ratio + 0.20*p95_ratio + 0.15*max_ratio
               + 0.05*rate_rms + 0.05*np.minimum(acc_rms, 1.0) + 0.05*hf + 0.05*saturation)
    score_c = rms_ratio + PENALTY*(saturation > SAT_CONSTRAINT) + PENALTY*lc
    rep_ = lambda v: np.broadcast_to(v[None, :], (C, W))
    return {"score_w": score_w, "score_c": score_c, "rms_ratio": rms_ratio,
            "roll_rms": rms_out[0], "pitch_rms": rms_out[1],
            "roll_rms_before": rep_(srms[0, 0]), "pitch_rms_before": rep_(srms[1, 0]),
            "roll_p95": p95[0], "pitch_p95": p95[1], "roll_max": mx[0], "pitch_max": mx[1],
            "roll_ratio": roll_ratio, "pitch_ratio": pitch_ratio,
            "rate_rms": rate_rms, "acc_rms": acc_rms, "hf": hf, "hf_rms": hf_rms_axis.max(axis=0), "flip_hz": flip_hz,
            "lc": lc, "saturation": saturation, "steps": rep_(lens.astype(float))}


METRIC_KEYS = ["score_w", "score_c", "rms_ratio", "roll_rms", "pitch_rms", "roll_rms_before", "pitch_rms_before",
               "roll_p95", "pitch_p95", "roll_max", "pitch_max", "roll_ratio", "pitch_ratio", "rate_rms", "acc_rms",
               "hf", "hf_rms", "flip_hz", "lc", "saturation"]


def as_item(x):
    return {"df": x} if isinstance(x, pd.DataFrame) else x


def prepare_inputs(items: list, n_seeds: int = 1):
    """관찰 구간 목록을 50 Hz 입력과 노이즈 시드로 바꾼다. 관찰 구간마다 n_seeds개의 노이즈 시드를 쓴다.
    반환: inputs, seeds, owners(입력 → 관찰 구간 번호)"""
    inputs, seeds, owners = [], [], []
    for j, raw in enumerate(items):
        it = as_item(raw)
        inp = it if "t" in it else window_inputs(it["df"])
        base = int(it.get("seed", NOISE_SEED_BASE + j))
        for r in range(n_seeds):
            inputs.append(inp); seeds.append(base + SEED_STRIDE*r); owners.append(j)
    return inputs, seeds, np.array(owners, dtype=int)


def objective_key(objective: str) -> str:
    return "score_c" if objective == OBJ_CONSTRAINED else "score_w"


def evaluate(inputs, seeds, cands, fixed, objective=OBJ_CONSTRAINED, owners=None, groups=None,
             aggregate="windows", progress_cb=None) -> dict:
    """후보들을 평가한다.
    - owners: 같은 관찰 구간의 여러 노이즈 시드를 관찰 구간 단위로 평균하기 위한 번호
    - aggregate: "windows"(관찰 구간 평균), "group_mean"(구간 평균의 평균), "group_max"(구간 평균의 최댓값)
    반환: {"summary": 후보별 평균 지표 목록, "per_item": 지표 -> (C, 관찰 구간 수) 배열}"""
    keep = objective != OBJ_CONSTRAINED
    W = len(inputs)
    N = max(inp["t"].size for inp in inputs)
    chunk = max(1, int(SERIES_BUDGET // (2 * W * N))) if keep else max(1, 120_000 // W)
    parts = []
    for a in range(0, len(cands), chunk):
        part = cands[a:a + chunk]
        cb = (lambda f, a=a, k=len(part): progress_cb((a + f*k) / len(cands))) if progress_cb else None
        parts.append(simulate_batch(inputs, seeds, part, fixed, cb, keep_p95=keep))
    m = {k: np.concatenate([p[k] for p in parts], axis=0) for k in METRIC_KEYS}
    if owners is None:
        owners = np.arange(W)
    n_items = int(owners.max()) + 1
    per_item = {}
    for k in METRIC_KEYS:
        acc = np.zeros((len(cands), n_items)); cnt = np.bincount(owners, minlength=n_items).astype(float)
        for j in range(n_items):
            acc[:, j] = m[k][:, owners == j].mean(axis=1)
        per_item[k] = acc
    per_item["lc_any"] = np.zeros((len(cands), n_items))
    for j in range(n_items):
        per_item["lc_any"][:, j] = m["lc"][:, owners == j].max(axis=1)
    sk = objective_key(objective)
    score_items = per_item[sk]
    if aggregate == "windows" or groups is None:
        agg = score_items.mean(axis=1)
    else:
        groups = np.asarray(groups)
        gm = np.stack([score_items[:, groups == g].mean(axis=1) for g in np.unique(groups)], axis=1)
        agg = gm.mean(axis=1) if aggregate == "group_mean" else gm.max(axis=1)
    summary = []
    for c in range(len(cands)):
        s = {k: (float(np.nanmean(per_item[k][c])) if np.isfinite(per_item[k][c]).any() else float("nan")) for k in METRIC_KEYS}
        s["lc_frac"] = float(per_item["lc_any"][c].mean())
        s["score"] = float(agg[c])
        summary.append(s)
    if progress_cb:
        progress_cb(1.0)
    return {"summary": summary, "per_item": per_item}


def evaluate_candidate(windows, p, fixed, objective=OBJ_WEIGHTED):
    """이전 API 호환용: 관찰 구간 목록과 후보 하나를 50 Hz로 평가한다."""
    if not len(windows):
        return {"score": float("inf")}
    inputs, seeds, owners = prepare_inputs(windows)
    return evaluate(inputs, seeds, [p], fixed, objective, owners)["summary"][0]


def stability_variants(params: dict, fixed: dict) -> list[tuple[str, dict]]:
    v = [("기본 조건", dict(params))]
    for g in STAB_GAIN_FACTORS:
        v.append((f"게인 ×{g:g} (Kp {params['kp']*g:.2f}, Kd {params['kd']*g:.3f})", dict(params, kp=params["kp"]*g, kd=params["kd"]*g)))
    for f in STAB_TAU_FACTORS:
        if fixed.get("servo2"):
            name = f"서보 속도 ×{1/f:.2g} (fn {fixed['wn']/(2*math.pi)/f:.2f} Hz)"
        else:
            name = f"서보 τ ×{f:g} ({fixed['tau']*f:.3f}s)"
        v.append((name, dict(params, tau=fixed["tau"]*f)))
    return v


def stability_check_many(cands: list[dict], inputs, seeds, owners, fixed, progress_cb=None) -> list[dict]:
    """3단계 stability_check를 여러 후보에 한 번에 적용한다.
    통과 조건: 기본·변형 조건 모두에서 자세 RMS비(검사 관찰 구간 전체 합산) < 1.0, 포화율 < 5%, 리밋사이클 없음."""
    variants = [stability_variants(c, fixed) for c in cands]
    flat = [vp for vs in variants for _, vp in vs]
    ev = evaluate(inputs, seeds, flat, fixed, OBJ_CONSTRAINED, owners, progress_cb=progress_cb)
    res = ev["summary"]
    pi = ev["per_item"]
    # 검사 관찰 구간 전체 RMS비(에너지 합산): 아주 잔잔한 관찰 구간 하나에서 센서 바이어스 때문에 생기는 큰 비율이 판정을 좌우하지 않게 한다.
    pooled = 0.5*(np.sqrt((pi["roll_rms"]**2).mean(1)) / np.sqrt((pi["roll_rms_before"]**2).mean(1))
                  + np.sqrt((pi["pitch_rms"]**2).mean(1)) / np.sqrt((pi["pitch_rms_before"]**2).mean(1)))
    for idx, s_ in enumerate(res):
        s_["rms_ratio_pooled"] = float(pooled[idx])
    out = []
    i = 0
    for c, vs in zip(cands, variants):
        rows = []
        for j, (name, _) in enumerate(vs):
            s = res[i]; i += 1
            ok = s["rms_ratio_pooled"] < 1.0 and s["saturation"] < STAB_SAT_MAX
            note = []
            if s["rms_ratio_pooled"] >= 1.0: note.append("RMS비 ≥ 1")
            if s["saturation"] >= STAB_SAT_MAX: note.append("포화율 ≥ 5%")
            if s["lc_frac"] > 0:   # 기본 조건뿐 아니라 변형 조건에서 채터링이 생겨도 실패(게인·서보 여유가 부족하다는 뜻)
                ok = False; note.append("리밋사이클 검출")
            rows.append({"조건": name, "RMS비(전체 합산)": s["rms_ratio_pooled"], "RMS비(관찰 구간 평균)": s["rms_ratio"], "포화율(%)": 100*s["saturation"],
                         "2Hz 이상 비율": s["hf"], "2Hz 이상 RMS(°)": s["hf_rms"], "부호변화 진동수(Hz)": s["flip_hz"],
                         "리밋사이클 관찰 구간 비율": s["lc_frac"], "판정": "통과" if ok else "실패", "사유": ", ".join(note)})
        table = pd.DataFrame(rows)
        out.append({"passed": bool((table["판정"] == "통과").all()), "table": table, "base": res[i - len(vs)]})
    return out


def stability_check(params: dict, windows_subset: list, fixed: dict) -> dict:
    inputs, seeds, owners = prepare_inputs(windows_subset)
    return stability_check_many([params], inputs, seeds, owners, fixed)[0]


def sample_candidate(rng, case_fixed: dict, kd_log: bool) -> dict:
    c = {k: float(rng.uniform(*TUNE_BOUNDS[k])) for k in TUNE_BOUNDS}
    if kd_log:
        c["kd"] = 0.0 if rng.uniform() < 0.1 else float(10**rng.uniform(math.log10(KD_LOG_MIN), math.log10(KD_MAX)))
    c.update(case_fixed)
    return c


def perturb_candidate(rng, base: dict, case_fixed: dict, kd_log: bool) -> dict:
    c = {}
    for k, (lo, hi) in TUNE_BOUNDS.items():
        c[k] = float(np.clip(base[k] + rng.normal(0, (hi-lo)*0.08), lo, hi))
    if kd_log:
        kd0 = max(base["kd"], KD_LOG_MIN)
        c["kd"] = float(np.clip(math.exp(math.log(kd0) + rng.normal(0, 0.4)), 0.0, KD_MAX))
    c.update(case_fixed)
    return c


def stratified_subset(n_items: int, groups, n_pick: int, rng) -> list[int]:
    if n_items <= n_pick:
        return list(range(n_items))
    if groups is None:
        return sorted(int(i) for i in rng.choice(n_items, n_pick, replace=False))
    groups = np.asarray(groups)
    ug = np.unique(groups)
    per = max(1, n_pick // len(ug))
    pick = []
    for g in ug:
        idx = np.where(groups == g)[0]
        pick += [int(i) for i in rng.choice(idx, min(per, len(idx)), replace=False)]
    return sorted(pick)


def optimize_params(train_windows, valid_windows, current, fixed, n_trials=36, progress_cb=None,
                    n_search=SEARCH_WINDOWS, seed=42, objective=OBJ_CONSTRAINED, case_fixed=None, kd_log=True,
                    n_valid_seeds=VALID_NOISE_SEEDS, train_groups=None, aggregate="windows", do_stability=True):
    """시뮬레이션 기반 자동 튜닝(무작위 탐색 + 국소 탐색).
    - 탐색 단계: 학습용 관찰 구간 중 n_search개(시드에 따라 정해지는 무작위)만 사용
    - 최종 검증: 검증용 관찰 구간 전체 × 노이즈 시드 n_valid_seeds개 평균
    - 최종 추천값은 stability_check를 통과한 후보 중 점수가 가장 좋은 것
    - 모든 평가는 실시간 재생과 같은 50 Hz 루프(simulate_batch)"""
    case_fixed = dict(case_fixed or {})
    rng = np.random.default_rng(seed)
    sub_rng = np.random.default_rng(SUBSET_SEED + seed)
    search_idx = stratified_subset(len(train_windows), train_groups, n_search, sub_rng)
    search_items = [train_windows[i] for i in search_idx]
    s_groups = None if train_groups is None else np.asarray(train_groups)[search_idx]
    s_in, s_seed, s_own = prepare_inputs(search_items)
    v_in, v_seed, v_own = prepare_inputs(valid_windows, n_valid_seeds)

    start = dict(current); start.update(case_fixed)
    candidates = [start]
    n_random = max(15, n_trials-9)
    for _ in range(n_random):
        candidates.append(sample_candidate(rng, case_fixed, kd_log))
    total = len(candidates) + 8 + (STAB_TOP_K if do_stability else 0) + 2

    def stage(frac0, count, label):
        if not progress_cb:
            return None
        return lambda f: progress_cb(min((frac0 + f*count)/total, 1.0), f"{label}")
    ev = evaluate(s_in, s_seed, candidates, fixed, objective, s_own, s_groups, aggregate,
                  stage(0, len(candidates), f"전역 탐색 · 후보 {len(candidates)}개 · 학습용 관찰 구간 {len(search_items)}개 · 50Hz"))
    records = [(m["score"], c, m, "전역 탐색") for c, m in zip(candidates, ev["summary"])]
    ranked = sorted(records, key=lambda x: x[0])
    local_candidates = [perturb_candidate(rng, base, case_fixed, kd_log) for _, base, _, _ in ranked[:4] for _ in range(2)]
    ev2 = evaluate(s_in, s_seed, local_candidates, fixed, objective, s_own, s_groups, aggregate,
                   stage(len(candidates), len(local_candidates), "상위 후보 국소 탐색"))
    records += [(m["score"], c, m, "국소 탐색") for c, m in zip(local_candidates, ev2["summary"])]
    history, best_so_far = [], float("inf")
    for i, (sc, _, _, _) in enumerate(records, 1):
        best_so_far = min(best_so_far, sc)
        history.append({"평가 횟수": i, "점수": sc, "최고 점수": best_so_far})
    ranked = sorted(records, key=lambda x: x[0])

    stab_results = {}
    best_rec, stable = ranked[0], None
    stab_checked = 0
    if do_stability:
        # 점수 순위대로 STAB_TOP_K개씩 검사하고, 통과 후보가 나오면 멈춘다(최대 STAB_MAX_CHECK개).
        # 점수가 좋은 후보일수록 Kd가 커서 여유가 부족할 수 있으므로 상위 몇 개만 보면 통과 후보를 놓친다.
        k_in, k_seed, k_own = prepare_inputs(search_items[:STAB_WINDOWS])
        passing = []
        limit_n = min(len(ranked), STAB_MAX_CHECK)
        while stab_checked < limit_n and not passing:
            top = ranked[stab_checked:stab_checked + STAB_TOP_K]
            checks = stability_check_many([r[1] for r in top], k_in, k_seed, k_own, fixed,
                                          stage(len(candidates)+8, STAB_TOP_K,
                                                f"안정성 검사 · 순위 {stab_checked+1}~{stab_checked+len(top)}"))
            for r, chk in zip(top, checks):
                stab_results[id(r[1])] = chk
            passing = [r for r in top if stab_results[id(r[1])]["passed"]]
            stab_checked += len(top)
        stable = bool(passing)
        best_rec = passing[0] if passing else ranked[0]
    best_score, best_params, train_metrics, _ = best_rec
    current_train = records[0][2]
    vres = evaluate(v_in, v_seed, [best_params, dict(current)], fixed, objective, v_own,
                    progress_cb=stage(total-2, 2, f"최종 검증 · 검증용 관찰 구간 {len(valid_windows)}개 × 노이즈 시드 {n_valid_seeds}개"))
    best_valid, current_valid = vres["summary"]
    all_rows = []
    for i, (sc, c, m, phase) in enumerate(records, 1):
        chk = stab_results.get(id(c))
        all_rows.append({"평가 순서": i, "단계": phase, "FF": c["ff"], "Kp": c["kp"], "Kd": c["kd"], "α": c["alpha"],
                         "학습 점수": sc, "자세 RMS비": m["rms_ratio"], "Roll RMS(°)": m["roll_rms"], "Pitch RMS(°)": m["pitch_rms"],
                         "Roll 최대각(°)": m["roll_max"], "Pitch 최대각(°)": m["pitch_max"],
                         "서보 각속도 RMS(정규화)": m["rate_rms"], "서보 각가속도 RMS(정규화)": m["acc_rms"],
                         "2Hz 이상 비율": m["hf"], "부호변화 진동수(Hz)": m["flip_hz"], "리밋사이클 관찰 구간 비율": m["lc_frac"],
                         "서보 포화율(%)": 100*m["saturation"],
                         "안정성 검사": "-" if chk is None else ("통과" if chk["passed"] else "실패"),
                         "추천": c is best_params})
    all_df = pd.DataFrame(all_rows)
    top_df = all_df.sort_values("학습 점수").head(10).reset_index(drop=True)
    top_df.insert(0, "순위", range(1, len(top_df)+1))
    usage = {"search_used": len(search_items), "train_total": len(train_windows),
             "valid_used": len(valid_windows), "valid_total": len(valid_windows), "valid_seeds": n_valid_seeds,
             "search_steps": int(sum(i["t"].size for i in s_in)),
             "valid_steps": int(sum(i["t"].size for i in v_in)),
             "search_evals": len(records), "valid_evals": 2,
             "stab_windows": min(STAB_WINDOWS, len(search_items)) if do_stability else 0,
             "stab_checked": stab_checked}
    return {"best": {k: float(best_params[k]) for k in TUNE_KEYS}, "train": train_metrics, "valid": best_valid,
            "current_train": current_train, "current_valid": current_valid, "top": top_df, "all": all_df,
            "history": pd.DataFrame(history), "n_candidates": len(records), "usage": usage, "seed": seed,
            "objective": objective, "case_fixed": case_fixed, "kd_log": kd_log, "stable": stable,
            "stability": stab_results.get(id(best_params)), "fixed": dict(fixed),
            "stab_items": list(search_items[:STAB_WINDOWS])}


def pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """두 지표를 모두 작게 하는 비지배해(파레토 해)."""
    keep = np.ones(len(x), bool)
    for i in range(len(x)):
        dominated = (x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i]))
        if dominated.any():
            keep[i] = False
    return keep


def grid_scan(ff: float, alpha: float, kps, kds, items, fixed, objective, progress_cb=None) -> dict:
    """4단계: FF·α 고정, Kp×Kd 격자 평가 + 각 칸의 stability_check."""
    inputs, seeds, owners = prepare_inputs(items)
    cands = [{"ff": ff, "kp": float(a), "kd": float(b), "alpha": alpha} for a in kps for b in kds]
    checks = stability_check_many(cands, inputs, seeds, owners, fixed, progress_cb)
    shape = (len(kps), len(kds))
    ratio = np.array([c["base"]["rms_ratio_pooled"] for c in checks]).reshape(shape)
    passed = np.array([c["passed"] for c in checks]).reshape(shape)
    sat = np.array([c["base"]["saturation"] for c in checks]).reshape(shape)
    if objective == OBJ_CONSTRAINED:
        score = np.array([c["base"]["score_c"] for c in checks]).reshape(shape)
    else:
        ev = evaluate(inputs, seeds, cands, fixed, objective, owners)
        score = np.array([s["score"] for s in ev["summary"]]).reshape(shape)
    return {"kps": np.asarray(kps), "kds": np.asarray(kds), "ratio": ratio, "passed": passed, "sat": sat,
            "score": score, "ff": ff, "alpha": alpha, "n_windows": len(items)}


def sensitivity_curves(base: dict, items, fixed, objective, n=13, kd_max_view=0.30, progress_cb=None) -> pd.DataFrame:
    """4단계 1D 민감도: 나머지를 base로 고정하고 한 파라미터만 범위 전체에서 바꾼다(Kd는 슬라이더 최대 0.30까지)."""
    inputs, seeds, owners = prepare_inputs(items)
    ranges = {"ff": (0.0, 1.2), "kp": (0.0, 4.0), "kd": (0.0, kd_max_view), "alpha": (0.90, 0.999)}
    cands, meta = [], []
    for k, (lo, hi) in ranges.items():
        for v in np.linspace(lo, hi, n):
            c = dict(base); c[k] = float(v); cands.append(c); meta.append((k, float(v)))
    ev = evaluate(inputs, seeds, cands, fixed, objective, owners, progress_cb=progress_cb)["summary"]
    return pd.DataFrame([{"파라미터": k, "값": v, "점수": s["score"], "자세 RMS비": s["rms_ratio"],
                          "포화율(%)": 100*s["saturation"], "리밋사이클 관찰 구간 비율": s["lc_frac"]}
                         for (k, v), s in zip(meta, ev)])


# -----------------------------------------------------------------------------
# 5단계: 해상상태별 튜닝, 전이 행렬, 공통 게인
# -----------------------------------------------------------------------------
TP_DEFAULT_EDGES = (6.0, 9.0, 12.0)
PRIOR_DEFAULT_EDGES = "0.5, 1.0, 2.0"


def wmo_label(code) -> str:
    if code is None:
        return "파랑 자료 없음"
    c = [w for w in WMO_CODES if w[0] == code][0]
    rng_txt = "Hs 0 m" if code == 0 else (f"Hs > {c[1]:g} m" if not np.isfinite(c[2]) else f"Hs {c[1]:g}~{c[2]:g} m")
    return f"WMO {code} {c[3]} ({rng_txt})"


def tp_bin(tp: float, edges) -> int | None:
    if tp is None or not np.isfinite(tp):
        return None
    return int(np.searchsorted(np.asarray(edges), tp, side="right"))


def tp_label(b: int, edges) -> str:
    e = list(edges)
    if b == 0: return f"Tp < {e[0]:g} s"
    if b == len(e): return f"Tp ≥ {e[-1]:g} s"
    return f"Tp {e[b-1]:g}~{e[b]:g} s"


def parse_edges(text: str) -> list[float]:
    vals = sorted({float(x) for x in re.split(r"[,\s]+", text.strip()) if x})
    if not vals:
        raise ValueError("구간 경계를 하나 이상 입력해 주세요.")
    return vals


def assign_bins(items: list[dict], criterion: str, tp_edges, prior_minutes: int, prior_edges) -> tuple[list, dict]:
    """5-3: 관찰 구간마다 구간 키를 붙인다. 쓸 수 없는 관찰 구간은 None.
    반환: 관찰 구간별 키 목록, 키 -> {"order", "label", "short"}"""
    keys, info = [], {}
    for it in items:
        k = None
        if criterion.startswith("A"):
            if it.get("wave_ok") and it.get("wmo") is not None:
                k = ("WMO", it["wmo"])
                info[k] = {"order": (it["wmo"],), "label": wmo_label(it["wmo"]), "short": f"WMO {it['wmo']}", "codes": [it["wmo"]]}
        elif criterion.startswith("B"):
            b = tp_bin(it.get("Tp"), tp_edges) if it.get("wave_ok") else None
            if b is not None:
                k = ("TP", b)
                info[k] = {"order": (b,), "label": tp_label(b, tp_edges), "short": tp_label(b, tp_edges)}
        elif criterion.startswith("C"):
            b = tp_bin(it.get("Tp"), tp_edges) if it.get("wave_ok") else None
            if b is not None and it.get("wmo") is not None:
                k = ("WMOxTP", it["wmo"], b)
                info[k] = {"order": (it["wmo"], b), "label": f"WMO {it['wmo']} × {tp_label(b, tp_edges)}",
                           "short": f"W{it['wmo']}·{tp_label(b, tp_edges)}"}
        else:
            v = it.get("prior", {}).get(prior_minutes, np.nan)
            if np.isfinite(v):
                b = int(np.searchsorted(np.asarray(prior_edges), v, side="right"))
                e = prior_edges
                lab = (f"직전 {prior_minutes}분 기울기 RMS < {e[0]:g}°" if b == 0 else
                       f"직전 {prior_minutes}분 기울기 RMS ≥ {e[-1]:g}°" if b == len(e) else
                       f"직전 {prior_minutes}분 기울기 RMS {e[b-1]:g}~{e[b]:g}°")
                k = ("IMU", b)
                info[k] = {"order": (b,), "label": lab, "short": lab.replace(f"직전 {prior_minutes}분 기울기 RMS ", "")}
        keys.append(k)
    return keys, info


def merge_bins(keys: list, info: dict, items: list[dict], min_train: int, min_valid: int):
    """학습용 관찰 구간/검증용 관찰 구간이 부족한 구간을 순서상 인접한 구간과 합친다. 반환: 관찰 구간별 그룹 번호, 그룹 정보, 병합 기록"""
    order = sorted(info, key=lambda k: info[k]["order"])
    groups = [[k] for k in order]

    def counts(g):
        tr = sum(1 for kk, it in zip(keys, items) if kk in g and not it["validation"])
        va = sum(1 for kk, it in zip(keys, items) if kk in g and it["validation"])
        return tr, va
    log = []
    while len(groups) > 1:
        bad = [i for i, g in enumerate(groups) if counts(g)[0] < min_train or counts(g)[1] < min_valid]
        if not bad:
            break
        i = bad[0]
        if i == 0:
            j = 1
        elif i == len(groups) - 1:
            j = i - 1
        else:
            j = i - 1 if sum(counts(groups[i-1])) <= sum(counts(groups[i+1])) else i + 1
        a, b = sorted((i, j))
        before = (group_label(groups[a], info), group_label(groups[b], info))
        groups[a] = groups[a] + groups[b]
        del groups[b]
        log.append(f"{before[0]} + {before[1]} → {group_label(groups[a], info)}")
    key_to_group = {k: gi for gi, g in enumerate(groups) for k in g}
    gid = [key_to_group.get(k) if k is not None else None for k in keys]
    ginfo = []
    for gi, g in enumerate(groups):
        tr, va = counts(g)
        ginfo.append({"id": gi, "keys": g, "label": group_label(g, info), "n_train": tr, "n_valid": va,
                      "enough": tr >= min_train and va >= min_valid})
    return gid, ginfo, log


def group_label(g: list, info: dict) -> str:
    if len(g) == 1:
        return info[g[0]]["label"]
    if all(k[0] == "WMO" for k in g):
        codes = sorted(k[1] for k in g)
        lo = [w for w in WMO_CODES if w[0] == codes[0]][0][1]
        hi = [w for w in WMO_CODES if w[0] == codes[-1]][0][2]
        hi_txt = f"> {lo:g}" if not np.isfinite(hi) else f"{lo:g}~{hi:g}"
        return f"WMO {'+'.join(str(c) for c in codes)} 병합 (Hs {hi_txt} m)"
    return " + ".join(info[k]["short"] for k in sorted(g, key=lambda k: info[k]["order"])) + " 병합"


def window_table(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"관찰 구간": it["id"], "날짜": it.get("date", "-"), "시작 시각(UTC)": it.get("start_time", "-"),
                          "Hs(m)": it.get("Hs", np.nan), "Tp(s)": it.get("Tp", np.nan),
                          "WMO 코드": it.get("wmo"), "WMO 이름": it.get("wmo_name", "-"),
                          "매칭 시간차 최대(s)": it.get("delta_max", np.nan), "WMO 경계 걸침": it.get("boundary", False),
                          "구분": "검증" if it["validation"] else "학습",
                          "선박 Roll RMS(°)": it.get("roll_rms", np.nan), "선박 Pitch RMS(°)": it.get("pitch_rms", np.nan),
                          "직전 3분 기울기 RMS(°)": it.get("prior", {}).get(3, np.nan)} for it in items])


def run_bin_analysis(items: list[dict], gid: list, ginfo: list[dict], current: dict, fixed: dict, objective: str,
                     case_fixed: dict, kd_log: bool, n_trials: int, n_rep: int, seed: int, progress_cb=None) -> dict:
    """5-4, 5-5: 구간별 최적화 → 전이 행렬·regret → 공통 게인(평균형/최악형) 비교."""
    groups = [g for g in ginfo if g["enough"]]
    if len(groups) < 1:
        raise ValueError("학습용 관찰 구간/검증용 관찰 구간 최소 개수를 만족하는 구간이 없습니다.")
    gids = [g["id"] for g in groups]
    tr_of = {g: [it for it, k in zip(items, gid) if k == g and not it["validation"]] for g in gids}
    va_of = {g: [it for it, k in zip(items, gid) if k == g and it["validation"]] for g in gids}
    jobs = len(gids) * n_rep + 2 + 1
    done = [0]

    def sub_cb(label):
        def cb(f, msg=""):
            if progress_cb:
                progress_cb(min((done[0] + f) / jobs, 1.0), f"{label} · {msg}")
        return cb
    per_group = {}
    for g in gids:
        reps = []
        for r in range(n_rep):
            res = optimize_params(tr_of[g], va_of[g], current, fixed, n_trials, sub_cb(f"구간 '{ginfo[g]['label']}' 시드 {r+1}/{n_rep}"),
                                  seed=seed + r, objective=objective, case_fixed=case_fixed, kd_log=kd_log)
            reps.append(res); done[0] += 1
        per_group[g] = reps
    all_train = [it for g in gids for it in tr_of[g]]
    all_groups = [g for g in gids for _ in tr_of[g]]
    common = {}
    for name, agg in (("평균형 공통 게인", "group_mean"), ("최악형 공통 게인", "group_max")):
        all_valid = [it for g in gids for it in va_of[g]]
        res = optimize_params(all_train, all_valid, current, fixed, n_trials, sub_cb(name), seed=seed,
                              objective=objective, case_fixed=case_fixed, kd_log=kd_log,
                              train_groups=all_groups, aggregate=agg)
        common[name] = res; done[0] += 1
    # 전이 행렬: 모든 게인 × 모든 구간의 검증용 관찰 구간(노이즈 시드 VALID_NOISE_SEEDS개)
    gains = [(f"g[{ginfo[g]['label']}]", per_group[g][0]["best"]) for g in gids]
    gains += [(n, common[n]["best"]) for n in common] + [("현재 슬라이더 값", dict(current))]
    v_items = [it for g in gids for it in va_of[g]]
    v_group = np.array([g for g in gids for _ in va_of[g]])
    v_in, v_seed, v_own = prepare_inputs(v_items, VALID_NOISE_SEEDS)
    ev = evaluate(v_in, v_seed, [c for _, c in gains], fixed, objective, v_own, progress_cb=sub_cb("전이 행렬 평가"))
    done[0] += 1
    sk = objective_key(objective)
    J = np.stack([ev["per_item"][sk][:, v_group == g].mean(axis=1) for g in gids], axis=1)        # (gain, group)
    R = np.stack([ev["per_item"]["rms_ratio"][:, v_group == g].mean(axis=1) for g in gids], axis=1)
    S = np.stack([ev["per_item"]["saturation"][:, v_group == g].mean(axis=1) for g in gids], axis=1)
    own = np.array([J[i, i] for i in range(len(gids))])
    regret = J / own[None, :] - 1.0
    rows = []
    for i, g in enumerate(gids):
        reps = per_group[g]
        b = np.array([[r["best"][k] for k in TUNE_KEYS] for r in reps])
        gw = [it for it, k in zip(items, gid) if k == g]
        hs = np.array([it["Hs"] for it in gw], float); tp = np.array([it["Tp"] for it in gw], float)
        rows.append({"구간": ginfo[g]["label"], "학습용 관찰 구간": ginfo[g]["n_train"], "검증용 관찰 구간": ginfo[g]["n_valid"],
                     "Hs 평균(m)": np.nanmean(hs) if np.isfinite(hs).any() else np.nan, "Hs 표준편차": np.nanstd(hs) if np.isfinite(hs).any() else np.nan,
                     "Tp 평균(s)": np.nanmean(tp) if np.isfinite(tp).any() else np.nan, "Tp 표준편차": np.nanstd(tp) if np.isfinite(tp).any() else np.nan,
                     "FF": reps[0]["best"]["ff"], "Kp": reps[0]["best"]["kp"], "Kd": reps[0]["best"]["kd"], "α": reps[0]["best"]["alpha"],
                     "FF 시드 표준편차": b[:, 0].std(), "Kp 시드 표준편차": b[:, 1].std(), "Kd 시드 표준편차": b[:, 2].std(), "α 시드 표준편차": b[:, 3].std(),
                     "검증 RMS비": R[i, i], "포화율(%)": 100*S[i, i],
                     "안정성 검사": "통과" if reps[0]["stable"] else "실패(통과 후보 없음)"})
    labels = [ginfo[g]["label"] for g in gids]
    comp = []
    n_g = len(gids)
    for j, g in enumerate(gids):
        row = {"구간": labels[j]}
        for name, gi in (("구간별 게인", j), ("평균형 공통", n_g), ("최악형 공통", n_g + 1), ("현재 슬라이더 값", n_g + 2)):
            row[f"{name} RMS비"] = R[gi, j]; row[f"{name} 포화율(%)"] = 100*S[gi, j]; row[f"{name} regret(%)"] = 100*regret[gi, j]
        comp.append(row)
    comp = pd.DataFrame(comp)
    summary = []
    for name, gi in (("평균형", n_g), ("최악형", n_g + 1), ("현재 슬라이더 값", n_g + 2)):
        j = int(np.argmax(regret[gi])); prefix = "공통 게인" if name != "현재 슬라이더 값" else ""
        summary.append(f"{prefix}({name}) 사용 시 최대 regret {100*regret[gi, j]:.1f}% ({labels[j]})" if prefix else
                       f"현재 슬라이더 값 사용 시 최대 regret {100*regret[gi, j]:.1f}% ({labels[j]})")
    return {"labels": labels, "gain_names": [n for n, _ in gains], "J": J, "R": R, "S": S, "regret": regret,
            "table": pd.DataFrame(rows), "comparison": comp, "summary": summary,
            "common": {n: common[n]["best"] for n in common}, "objective": objective}


# -----------------------------------------------------------------------------
# 5-6: 가상 파도 대조 실험 (가상 파도는 실선 RAO 모델이 아님)
# -----------------------------------------------------------------------------
VIRTUAL_HS = (0.9, 1.9, 3.2, 5.0)     # WMO 3~6 대표값
VIRTUAL_TP = (6.0, 8.0, 10.0, 12.0)


def virtual_gain_for(hs: float) -> float:
    """기존 가상 모드 프리셋(Hs 0.5/2/4 m → 배율 0.6/2.0/3.5)을 Hs에 대해 선형 보간·외삽한 흔들림 배율."""
    xs, ys = (0.5, 2.0, 4.0), (0.6, 2.0, 3.5)
    if hs > xs[-1]:
        return ys[-1] + (hs - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(np.interp(hs, xs, ys))


def virtual_window(hs: float, tp: float, realization: int, duration: float, gain_mult: float, direction: float = 25.0,
                   phys_p: dict | None = None) -> dict:
    """virtual_motion과 같은 식을 시간 배열로 계산한 가상 관찰 구간(충격 없음).
    phys_p 를 주면 물리 기반 운동방정식(sea_state.py)으로 선박 운동을 만든다(이전 배율 virtual_gain_for 는 쓰지 않음)."""
    n = int(round(duration / CONTROL_DT))
    t = np.arange(n) * CONTROL_DT
    if phys_p is not None:
        comps = phys.wave_components_phys(tp, hs, PHYS_COMPONENTS, seed=123 + 31*realization)
        terms = phys.component_terms(comps, direction, phys_p)
        r = phys.response(t, terms)
        g_ = float(gain_mult)
        roll, pitch, rr, pr = r["roll"]*g_, r["pitch"]*g_, r["roll_rate"]*g_, r["pitch_rate"]*g_
        axp, ayp = _particle_acc(t, comps, direction, terms["we"], {"L": phys_p["L"], "B": phys_p["B"], "T": phys_p["T"]})
        h = float(phys_p.get("imu_h", 0.0))
        ay = ayp + h*np.radians(r["roll_acc"]*g_); ax = axp - h*np.radians(r["pitch_acc"]*g_)
        acc_err = np.stack([np.rad2deg(np.arctan(ay/G)), -np.rad2deg(np.arctan(ax/G))])
        return {"t": t, "ship": np.stack([roll, pitch]), "rate": np.stack([rr, pr]), "acc": acc_err,
                "init": (float(roll[0]), float(pitch[0])), "duration": float(t[-1])}
    comps = wave_components(tp, hs, 5, seed=123 + 31*realization)
    freqs, amps, phases, spread = comps
    th = np.deg2rad(direction) + spread
    w = 2*np.pi*freqs; k = w*w/G
    ang = -w[None, :]*t[:, None] + phases[None, :]
    slope = -amps*k*np.sin(ang); slope_rate = amps*k*w*np.cos(ang)
    scale = np.rad2deg(0.75) * virtual_gain_for(hs) * gain_mult
    roll = (slope*np.sin(th)).sum(1)*scale; pitch = (slope*np.cos(th)).sum(1)*scale
    rr = (slope_rate*np.sin(th)).sum(1)*scale; pr = (slope_rate*np.cos(th)).sum(1)*scale
    acc = amps*w*w*np.sin(ang)
    ax = 0.5*(acc*np.cos(th)).sum(1); ay = 0.5*(acc*np.sin(th)).sum(1)
    acc_err = np.stack([np.rad2deg(np.arctan(ay/G)), -np.rad2deg(np.arctan(ax/G))])
    return {"t": t, "ship": np.stack([roll, pitch]), "rate": np.stack([rr, pr]), "acc": acc_err,
            "init": (float(roll[0]), float(pitch[0])), "duration": float(t[-1])}


# -----------------------------------------------------------------------------
# WMO 0~9 가상 해상상태와 보정 한계 분석
# -----------------------------------------------------------------------------
# Pierson–Moskowitz 완전발달 해역 (Stewart, Introduction to Physical Oceanography 16.4절):
#   ω_p = 0.877·g/U19.5,  H1/3 = 0.21·U19.5²/g   →  U를 지우면  Tp = 2π/ω_p = (2π/0.877)·√(Hs/(0.21·g)) ≈ 4.99·√Hs
PM_TP_COEF = (2*math.pi/0.877) * math.sqrt(1.0/(0.21*G))
LIMIT_WARMUP_S = 60.0   # 보정 한계 분석에서 뺄 시작 구간(초): α=0.998·바이어스 추정의 수렴 시간(약 20~35 s)보다 넉넉히
# Akademik INS 관찰 구간 96개로 맞춘 선박 기울기 RMS = c·Hs^p (앱의 fit_tilt_power 결과, 2026-11-26 배포 앱에서 확인).
# 관찰 구간을 준비하기 전(가상 모드)에도 이 값을 기본으로 쓴다. 준비하면 그 자리에서 다시 계산한 값으로 바뀐다.
AKADEMIK_FIT = {"c": 1.879, "p": 0.281, "n": 96, "r2": 0.41, "hs_min": 0.11, "hs_max": 4.89, "saved": True}
WMO_REP_HS = {0: 0.0, 1: 0.05, 2: 0.30, 3: 0.875, 4: 1.875, 5: 3.25, 6: 5.0, 7: 7.5, 8: 11.5, 9: 15.0}   # 각 단계 구간의 가운데 (9는 14m 초과 → 15m)


def pm_tp(hs: float) -> float:
    """완전발달 해역 PM 관계의 첨두주기. 아주 작은 파고에서도 계산이 되도록 2초 아래로는 내리지 않는다."""
    return max(2.0, PM_TP_COEF*math.sqrt(max(float(hs), 0.0)))


def wmo_row(code: int):
    return [w for w in WMO_CODES if w[0] == int(code)][0]     # (코드, Hs 하한, Hs 상한, 이름)


def tilt_rms(inp: dict) -> float:
    """관찰 구간의 선박 기울기 RMS = √(롤 RMS² + 피치 RMS²)."""
    r = np.sqrt(np.mean(inp["ship"][0]**2)); q = np.sqrt(np.mean(inp["ship"][1]**2))
    return float(math.hypot(r, q))


def fit_tilt_power(items: list) -> dict | None:
    """실측 관찰 구간(Akademik)에서 선박 기울기 RMS = c·Hs^p 를 로그-로그 최소제곱으로 맞춘다."""
    pts = [(it["Hs"], math.hypot(it["roll_rms"], it["pitch_rms"])) for it in items
           if it.get("wave_ok") and np.isfinite(it.get("Hs", np.nan)) and it["Hs"] > 0.05]
    if len(pts) < 5:
        return None
    h = np.log([a for a, _ in pts]); y = np.log([max(b, 1e-3) for _, b in pts])
    p_, logc = np.polyfit(h, y, 1)
    pred = logc + p_*h
    r2 = 1 - np.sum((y-pred)**2)/max(np.sum((y-y.mean())**2), 1e-12)
    return {"c": float(np.exp(logc)), "p": float(p_), "n": len(pts), "r2": float(r2),
            "hs_min": float(np.exp(h.min())), "hs_max": float(np.exp(h.max()))}


def motion_mult(hs: float, tp: float, model: str, c: float, p_: float) -> float:
    """가상 관찰 구간의 흔들림 배율. model="assumed": 기존 앱 가정(virtual_gain_for). model="fit": 기울기 RMS가 c·Hs^p 가 되도록 맞춘 배율."""
    if model != "fit" or hs <= 0:      # "phys"(물리 모델)와 "assumed"(이전 배율)는 여기서 배율을 더하지 않는다
        return 1.0
    base = tilt_rms(virtual_window(hs, tp, 0, 180.0, 1.0))
    return float(c*hs**p_/base) if base > 1e-9 else 0.0


def tilt_rms_after(inp: dict, warmup_s: float) -> float:
    """처음 warmup_s 초를 뺀 선박 기울기 RMS (보정 한계 분석에서 상판과 같은 구간으로 비교)."""
    k0 = min(int(round(warmup_s/CONTROL_DT)), inp["ship"].shape[1]-1)
    r = np.sqrt(np.mean(inp["ship"][0, k0:]**2)); q = np.sqrt(np.mean(inp["ship"][1, k0:]**2))
    return float(math.hypot(r, q))


def limit_sweep(conds: list, params: dict, fixed: dict, model: str, c: float, p_: float, gain_user: float,
                n_real: int = 3, duration: float = 300.0, progress_cb=None, warmup_s: float = None,
                phys_p: dict | None = None) -> pd.DataFrame:
    """조건 목록 [(Hs, Tp)] 마다 가상 관찰 구간 n_real 개를 만들어 현재 파라미터로 평가한다(튜닝과 같은 50Hz 루프).
    장치를 켠 직후(warmup_s, 기본 LIMIT_WARMUP_S)는 필터·바이어스 추정이 수렴하는 구간이라 판정에서 뺀다.
    (이전 판은 0초부터 넣어서 '상판 최대'가 시작 순간의 선박 기울기로 채워졌다.)"""
    warmup_s = LIMIT_WARMUP_S if warmup_s is None else float(warmup_s)
    fixed_v = dict(fixed, accel_disturb=0.0)
    rows = []
    for i, (hs, tp) in enumerate(conds):
        if hs <= 0:
            rows.append({"Hs(m)": 0.0, "Tp(s)": tp, "WMO": 0, "선박 기울기 RMS(°)": 0.0, "상판 기울기 RMS(°)": 0.0,
                         "RMS비": np.nan, "상판 최대(°)": 0.0, "서보 포화율(%)": 0.0, "리밋사이클": False})
            continue
        mult = motion_mult(hs, tp, model, c, p_) * gain_user
        if model == "phys":   # 물리 모델: 실현마다 파향을 45°/90°/135°로 바꿔 선수 사파·정횡·선미 사파를 모두 본다
            wins = [virtual_window(hs, tp, r, duration, mult, PHYS_HEADINGS[r % len(PHYS_HEADINGS)], phys_p=phys_p or phys.default_params())
                    for r in range(n_real)]
        else:
            wins = [virtual_window(hs, tp, r, duration, mult) for r in range(n_real)]
        seeds = [NOISE_SEED_BASE + 1000*i + r for r in range(n_real)]
        m = simulate_batch(wins, seeds, [params], fixed_v, keep_p95=False, warmup_s=warmup_s)
        ship = np.mean([tilt_rms_after(w, warmup_s) for w in wins])
        plate = float(np.mean(np.hypot(m["roll_rms"][0], m["pitch_rms"][0])))
        rows.append({"Hs(m)": hs, "Tp(s)": tp, "WMO": classify_sea_state(hs)[0], "선박 기울기 RMS(°)": ship,
                     "상판 기울기 RMS(°)": plate, "RMS비": plate/ship if ship > 1e-9 else np.nan,
                     "상판 최대(°)": float(np.max(np.maximum(m["roll_max"][0], m["pitch_max"][0]))),
                     "서보 포화율(%)": float(100*np.max(m["saturation"][0])), "리밋사이클": bool(np.max(m["lc"][0]) > 0)})
        if progress_cb:
            progress_cb((i+1)/len(conds))
    return pd.DataFrame(rows)


def real_limit_by_wmo(items: list, params: dict, fixed: dict, warmup_s: float = None) -> pd.DataFrame:
    """실측 관찰 구간(Akademik INS)을 WMO 단계별로 묶어 현재 파라미터로 평가한다. 처음 warmup_s 초는 판정에서 뺀다."""
    warmup_s = LIMIT_WARMUP_S if warmup_s is None else float(warmup_s)
    rows = []
    for code in range(10):
        its = [it for it in items if it.get("wmo") == code]
        if not its:
            continue
        inp, sd, own = prepare_inputs(its, 1)
        m = simulate_batch(inp, sd, [params], fixed, keep_p95=False, warmup_s=warmup_s)
        ship = np.array([tilt_rms_after(x, warmup_s) for x in inp]); plate = np.hypot(m["roll_rms"][0], m["pitch_rms"][0])
        rows.append({"WMO": code, "관찰 구간 수": len(its), "Hs 범위(m)": f"{min(i_['Hs'] for i_ in its):.2f}~{max(i_['Hs'] for i_ in its):.2f}",
                     "선박 기울기 RMS(°)": float(ship.mean()), "상판 기울기 RMS(°)": float(plate.mean()),
                     "상판 RMS 최악 관찰 구간(°)": float(plate.max()), "RMS비": float(plate.mean()/max(ship.mean(), 1e-9)),
                     "상판 최대(°)": float(np.max(np.maximum(m["roll_max"][0], m["pitch_max"][0]))),
                     "서보 포화율(%)": float(100*np.max(m["saturation"][0])), "리밋사이클": bool(np.max(m["lc"][0]) > 0)})
    return pd.DataFrame(rows)


def judge(df: pd.DataFrame, rms_lim: float, max_lim: float, sat_lim: float, rms_col: str = "상판 기울기 RMS(°)") -> pd.DataFrame:
    df = df.copy()
    why = []
    for _, r in df.iterrows():
        w = []
        if r[rms_col] > rms_lim: w.append(f"RMS {r[rms_col]:.2f}°>{rms_lim:g}°")
        if r["상판 최대(°)"] > max_lim: w.append(f"최대 {r['상판 최대(°)']:.2f}°>{max_lim:g}°")
        if r["서보 포화율(%)"] > sat_lim: w.append(f"포화 {r['서보 포화율(%)']:.1f}%>{sat_lim:g}%")
        if r["리밋사이클"]: w.append("리밋사이클")
        why.append(", ".join(w))
    df["판정"] = ["통과" if not w else "실패" for w in why]; df["실패 이유"] = why
    return df


def highest_pass(df: pd.DataFrame) -> int | None:
    """0단계부터 올라가며 연속으로 통과한 마지막 WMO 단계 (한 단계 안에 여러 조건이 있으면 모두 통과해야 통과)."""
    best = None
    for code in sorted(df["WMO"].dropna().unique()):
        if (df.loc[df["WMO"] == code, "판정"] == "통과").all():
            best = int(code)
        else:
            break
    return best


def run_virtual_analysis(current: dict, fixed: dict, objective: str, case_fixed: dict, kd_log: bool, n_trials: int,
                         n_train: int, duration: float, gain_mult: float, seed: int, progress_cb=None) -> dict:
    conds = [(hs, tp) for hs in VIRTUAL_HS for tp in VIRTUAL_TP]
    fixed_v = dict(fixed, accel_disturb=0.0)   # 실시간 가상 모드도 추가 가속도 오염은 쓰지 않는다
    train_of, valid_of = {}, {}
    for ci, (hs, tp) in enumerate(conds):
        train_of[ci] = [dict(virtual_window(hs, tp, r, duration, gain_mult), seed=NOISE_SEED_BASE + 100*ci + r) for r in range(n_train)]
        valid_of[ci] = [dict(virtual_window(hs, tp, 50, duration, gain_mult), seed=NOISE_SEED_BASE + 100*ci + 50)]
    best = {}
    for ci, (hs, tp) in enumerate(conds):
        cb = (lambda f, m, ci=ci: progress_cb((ci + f) / (len(conds) + 1), f"Hs {conds[ci][0]} m · Tp {conds[ci][1]:g} s · {m}")) if progress_cb else None
        res = optimize_params(train_of[ci], valid_of[ci], current, fixed_v, n_trials, cb, seed=seed, objective=objective,
                              case_fixed=case_fixed, kd_log=kd_log, n_valid_seeds=1)
        best[ci] = res
    v_items = [valid_of[ci][0] for ci in range(len(conds))]
    v_in, v_seed, v_own = prepare_inputs(v_items, VALID_NOISE_SEEDS)
    ev = evaluate(v_in, v_seed, [best[ci]["best"] for ci in range(len(conds))], fixed_v, objective, v_own,
                  progress_cb=(lambda f: progress_cb((len(conds) + f) / (len(conds) + 1), "전이 행렬 평가")) if progress_cb else None)
    J = ev["per_item"][objective_key(objective)]                # (gain i, condition j)
    regret = J / np.diag(J)[None, :] - 1.0
    rows = []
    for ci, (hs, tp) in enumerate(conds):
        b = best[ci]["best"]; c, name = classify_sea_state(hs)
        rows.append({"Hs(m)": hs, "Tp(s)": tp, "WMO": f"{c} {name}", "FF": b["ff"], "Kp": b["kp"], "Kd": b["kd"], "α": b["alpha"],
                     "검증 RMS비": ev["per_item"]["rms_ratio"][ci, ci], "포화율(%)": 100*ev["per_item"]["saturation"][ci, ci],
                     "안정성 검사": "통과" if best[ci]["stable"] else "실패"})
    table = pd.DataFrame(rows)
    same_hs = [regret[i, j] for i, (h1, t1) in enumerate(conds) for j, (h2, t2) in enumerate(conds) if h1 == h2 and t1 != t2]
    same_tp = [regret[i, j] for i, (h1, t1) in enumerate(conds) for j, (h2, t2) in enumerate(conds) if t1 == t2 and h1 != h2]
    infl = []
    for k, span in (("FF", 1.2), ("Kp", 4.0), ("Kd", KD_MAX), ("α", 0.099)):
        piv = table.pivot(index="Hs(m)", columns="Tp(s)", values=k).to_numpy()
        infl.append({"파라미터": k, "Hs 고정·Tp 변화 시 표준편차(범위 대비 %)": 100*np.mean(piv.std(axis=1))/span,
                     "Tp 고정·Hs 변화 시 표준편차(범위 대비 %)": 100*np.mean(piv.std(axis=0))/span})
    infl = pd.DataFrame(infl)
    m_hs, m_tp = 100*float(np.mean(same_hs)), 100*float(np.mean(same_tp))
    verdict = ("Tp(주기)" if m_hs > m_tp else "Hs(파고)")
    summary = (f"같은 Hs에서 Tp가 다른 조건의 게인을 쓰면 평균 regret {m_hs:.1f}%, 같은 Tp에서 Hs가 다른 조건의 게인을 쓰면 평균 regret {m_tp:.1f}% → "
               f"이 가상 모델에서는 {verdict} 쪽이 최적 게인에 더 큰 영향을 줍니다.")
    return {"conds": conds, "table": table, "J": J, "regret": regret, "influence": infl, "summary": summary,
            "regret_same_hs": m_hs, "regret_same_tp": m_tp}


def kd_step_response(base: dict, kd: float, fixed: dict, step_deg: float = 2.0, noise: bool = False,
                     ramp: float = 0.2, T: float = 8.0, t0: float = 1.0, seed: int = 7):
    """D 게인 비교용 계단 응답: 선박 Roll이 t0초에 ramp초 동안 step_deg만큼 기울고 그대로 유지된다.
    실시간 재생과 같은 control_step 루프를 쓴다. 나머지 파라미터(FF, Kp, α)는 base로 고정."""
    f = dict(fixed)
    if not noise:
        f.update(gyro_noise=0.0, gyro_bias=0.0, acc_noise=0.0)
    p = {"ff": base["ff"], "kp": base["kp"], "kd": float(kd), "alpha": base["alpha"], "tau": f["tau"],
         "rate": float(f["rate"]), "limit": f["limit"], "gyro_noise": f["gyro_noise"], "gyro_bias": f["gyro_bias"],
         "acc_noise": f["acc_noise"], "servo2": bool(f.get("servo2", False)), "wn": f.get("wn", 0.0), "zeta": f.get("zeta", 0.0), "servo_model": f.get("servo_model", "zoh1"), "delay_steps": int(f.get("delay_steps", 0)), "timing": f.get("timing", "consistent"),
         "ki": float(f.get("ki", 0.0))}
    sim = new_sim_state((0.0, 0.0), seed=seed)
    rows = []
    while sim["t"] < T:
        t = sim["t"]
        r = float(np.clip((t - t0) / ramp, 0.0, 1.0)) * step_deg
        rr = step_deg / ramp if t0 <= t < t0 + ramp else 0.0
        row = control_step(sim, {"roll": r, "pitch": 0.0, "roll_rate": rr, "pitch_rate": 0.0}, p, True, 0.0)
        rows.append({"time": t, "ship_roll": row["ship_roll"], "plate_roll": row["plate_roll"],
                     "joint_roll": row["joint_roll"], "joint_rate": sim["roll"]["joint_rate"]})
    df = pd.DataFrame(rows)
    m = df["time"] >= t0
    y = df.loc[m, "plate_roll"].to_numpy(); tt = df.loc[m, "time"].to_numpy()
    final = float(np.mean(y[tt >= T - 1.0]))
    e = y - final
    ipk = int(np.argmax(np.abs(e)))
    peak = float(e[ipk])
    after = e[ipk:]
    over = float(max(0.0, -np.sign(peak) * after.min() if peak > 0 else after.max()))
    band = 0.05 * abs(step_deg)
    outside = np.where(np.abs(e) > band)[0]
    settle = float(tt[outside[-1]] - t0) if len(outside) else 0.0
    big = after[np.abs(after) > 0.01 * abs(step_deg)]
    osc = int(np.sum(np.diff(np.sign(big)) != 0)) if len(big) > 1 else 0
    metrics = {"Kd": float(kd), "상판 최대 편차(°)": abs(peak), "수평을 지나친 폭(°)": over,
               "넘어감 비율(%)": 100 * over / max(abs(peak), 1e-9), "5% 정착 시간(s)": settle,
               "흔들림 횟수(부호 변화)": osc, "정상상태 잔차(°)": final,
               "서보 최대 각속도(°/s)": float(df.loc[m, "joint_rate"].abs().max())}
    return df, metrics


def consistency_check(segment: pd.DataFrame, params: dict, fixed: dict, seed: int = NOISE_SEED_BASE) -> pd.DataFrame:
    """같은 구간·같은 파라미터·같은 노이즈 시드에서
    (1) 자동 튜닝 평가(simulate_batch) RMS와 (2) 실시간 재생(actual_motion_at + control_step, live_panel과 같은 루프) RMS를 비교한다.
    RMS 계산식은 live_performance와 같다: sqrt(mean(x*x))."""
    seg = segment.copy().reset_index(drop=True)
    el = seg["elapsed_s"].to_numpy(dtype=float)
    seg["elapsed_s"] = el - el[0]
    for c in ("heading", "north_speed", "east_speed"):
        if c not in seg:
            seg[c] = np.nan
    if "timestamp" not in seg:
        seg["timestamp"] = None
    if "roll_rate" not in seg:
        seg["roll_rate"] = np.gradient(seg["ship_roll"].to_numpy(float), seg["elapsed_s"].to_numpy(float))
        seg["pitch_rate"] = np.gradient(seg["ship_pitch"].to_numpy(float), seg["elapsed_s"].to_numpy(float))
    inp = window_inputs(seg)
    m = simulate_batch([inp], [seed], [params], fixed)
    sim = new_sim_state((seg["ship_roll"].iloc[0], seg["ship_pitch"].iloc[0]), seed=seed)
    duration = float(seg["elapsed_s"].iloc[-1])
    p = {"ff": params["ff"], "kp": params["kp"], "kd": params["kd"], "alpha": params["alpha"],
         "tau": fixed["tau"], "rate": float(fixed["rate"]), "limit": fixed["limit"],
         "gyro_noise": fixed["gyro_noise"], "gyro_bias": fixed["gyro_bias"], "acc_noise": fixed["acc_noise"],
         "servo2": bool(fixed.get("servo2", False)), "wn": fixed.get("wn", 0.0), "zeta": fixed.get("zeta", 0.0), "servo_model": fixed.get("servo_model", "zoh1"), "delay_steps": int(fixed.get("delay_steps", 0)), "timing": fixed.get("timing", "consistent"),
         "ki": float(params.get("ki", fixed.get("ki", 0.0)))}
    rows = []
    while sim["t"] < duration:
        motion = actual_motion_at(sim["t"], seg)
        rows.append(control_step(sim, motion, p, True, fixed.get("accel_disturb", 0.0), CONTROL_DT))
    live = pd.DataFrame(rows)
    out = []
    for axis, label in (("roll", "Roll"), ("pitch", "Pitch")):
        s_ = live[f"ship_{axis}"].to_numpy(float); p_ = live[f"plate_{axis}"].to_numpy(float)
        live_rms = float(np.sqrt(np.mean(p_*p_))); live_ship = float(np.sqrt(np.mean(s_*s_)))
        tune_rms = float(m[f"{axis}_rms"][0, 0])
        diff = abs(tune_rms - live_rms) / max(live_rms, 1e-12)
        out.append({"축": label, "스텝 수(튜닝/실시간)": f"{int(m['steps'][0, 0]):,} / {len(live):,}",
                    "보정 전 RMS(°)": live_ship, "튜닝 평가 RMS(°)": tune_rms, "실시간 재생 RMS(°)": live_rms,
                    "튜닝 평가 RMS비": tune_rms / max(live_ship, 1e-12), "실시간 RMS비": live_rms / max(live_ship, 1e-12),
                    "차이(%)": 100*diff, "판정(≤5%)": "통과" if diff <= CONSISTENCY_TOL else "실패"})
    return pd.DataFrame(out)


def consistency_windows(seg: pd.DataFrame, count: int, window_rows: int = 240) -> list[pd.DataFrame]:
    """현재 재생 구간에서 관찰 구간과 같은 방식(최대 240개 기록, 속도는 관찰 구간 안에서 다시 계산)으로 검증 관찰 구간을 뽑는다."""
    n = len(seg)
    if n < 60:
        return []
    length = min(window_rows, n)
    starts = sorted(set(int(x) for x in np.linspace(0, n - length, count)))
    out = []
    for a in starts:
        w = seg.iloc[a:a + length].copy().reset_index(drop=True)
        el = w["elapsed_s"].to_numpy(float)
        w["roll_rate"] = np.gradient(w["ship_roll"].to_numpy(float), el)
        w["pitch_rate"] = np.gradient(w["ship_pitch"].to_numpy(float), el)
        out.append(w)
    return out


def apply_best_parameters():
    result = st.session_state.get("optimization_result")
    if result:
        for key, value in result["best"].items():
            st.session_state[f"param_{key}"] = float(value)
        st.session_state["parameters_applied"] = True


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
BOX_FACES = [(0,1,3),(0,3,2),(4,6,7),(4,7,5),(0,4,5),(0,5,1),
             (2,3,7),(2,7,6),(0,2,6),(0,6,4),(1,5,7),(1,7,3)]


def box(cx, cy, cz, lx, ly, lz):
    V = np.array([[x,y,z] for x in (cx-lx/2,cx+lx/2) for y in (cy-ly/2,cy+ly/2) for z in (cz-lz/2,cz+lz/2)], float)
    f = np.array(BOX_FACES)
    return V, f[:,0], f[:,1], f[:,2]


def rotation(roll, pitch):
    r,p = np.deg2rad(roll), np.deg2rad(pitch)
    rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    return ry @ rx


def mesh(V, I, J, K, color, name, legend=True, opacity=1.0):
    return go.Mesh3d(x=V[:,0], y=V[:,1], z=V[:,2], i=I, j=J, k=K, color=color,
                     opacity=opacity, flatshading=True, name=name, showlegend=legend, hoverinfo="skip")


def hull_mesh():
    length, breadth = SHIP["length"], SHIP["breadth"]
    xs = np.linspace(-length/2, length/2, 45)
    vertices=[]
    for x in xs:
        ratio=abs(x/(length/2)); half=(breadth/2)*np.sqrt(max(0.015,1-ratio**2.3))
        vertices += [[x,-half,1.2],[x,-half*0.88,-0.35],[x,0,-1.8],[x,half*0.88,-0.35],[x,half,1.2]]
    ii=[];jj=[];kk=[]
    for i in range(len(xs)-1):
        for j in range(4):
            a=i*5+j;c=(i+1)*5+j
            ii += [a,a+1]; jj += [a+1,c+1]; kk += [c,c]
    return np.array(vertices),np.array(ii),np.array(jj),np.array(kk)

_HULL_CACHE: dict = {}


def hull_for(length: float, breadth: float):
    key = (round(float(length), 2), round(float(breadth), 2))
    if key not in _HULL_CACHE:
        global SHIP
        saved = SHIP
        SHIP = dict(saved, length=key[0], breadth=key[1])
        try:
            _HULL_CACHE[key] = hull_mesh()
        finally:
            SHIP = saved
    return _HULL_CACHE[key]


SHIPS = {
    "Akademik Tryoshnikov (항해 데이터 선박)": dict(SHIP),
    "한나라호 (한국해양대 실습선 · 최소 선박 기준)": {"name": "한나라호 HANNARA", "length": 133.0, "breadth": 19.4,
                                         "displacement": None, "draft": None, "gross_tonnage": 9196,
                                         "source": "VesselFinder IMO 9807279 (전장 133.0 m, 폭 19.4 m, 총톤수 9196)"},
}
HULL = hull_for(SHIP["length"], SHIP["breadth"])


def scene_figure(row, actual_mode, comps=None, direction=0.0, controller_on=True, height=520, plate_scale=1.0):
    fig=go.Figure()
    if actual_mode:
        X,Y=np.meshgrid(np.linspace(-82,82,20),np.linspace(-24,24,12)); Z=np.full_like(X,-2.5)
        surf_name="수면(개념 표시 · 실측 파면 아님)"
    else:
        X,Y,Z=ocean_surface(float(row["time"]),comps,direction); surf_name="가상 파도"
    fig.add_trace(go.Surface(x=X,y=Y,z=Z,showscale=False,opacity=0.86,name=surf_name,hoverinfo="skip",
                             colorscale=[[0,"#042f55"],[0.5,"#087eae"],[1,"#a8e8f0"]]))
    V,I,J,K=hull_for(SHIP["length"],SHIP["breadth"])
    Rs=rotation(float(row["ship_roll"]),float(row["ship_pitch"]))
    lift=np.array([0,0,float(row.get("heave",0.0))])
    tf=lambda vertices:(Rs@vertices.T).T+lift
    fig.add_trace(mesh(tf(V),I,J,K,"#123f70","Akademik Tryoshnikov 선체"))
    sx=SHIP["length"]/133.53; sy=SHIP["breadth"]/23.25          # 선박 크기에 맞춰 상부 구조물·짐벌 위치를 비례 배치
    structures=[box(39*sx,0,3.1,20*sx,12*sy,3.6),box(42*sx,0,6.1,10*sx,9*sy,2.5),box(31*sx,0,5.5,2,2,6),box(31*sx,0,8.8,5*sy,0.6,0.4)]
    for idx,(v,i,j,k) in enumerate(structures): fig.add_trace(mesh(tf(v),i,j,k,"#f1f1e9","상부 구조물",legend=idx==0))
    pivot_ship=np.array([-42.0*sx,0.0,1.7]); pivot=tf(pivot_ship[None,:])[0]
    # 상판·지지대·적재물은 표시용으로 plate_scale 배 키운다 (계산과 무관, 실제 크기는 2.8×1.7m)
    sc=float(max(plate_scale,1.0)); post_h=2.0+0.6*(sc-1.0)
    post=box(0,0,post_h/2,1.3*(1+0.35*(sc-1)),1.3*(1+0.35*(sc-1)),post_h); v,i,j,k=post
    fig.add_trace(mesh((Rs@v.T).T+pivot,i,j,k,"#3b4a5a","짐벌 지지대"))
    Rp=rotation(float(row["plate_roll"]),float(row["plate_pitch"]))
    th=0.18*sc*0.7
    plate=box(0,0,post_h+th/2,2.8*sc,1.7*sc,th); v,i,j,k=plate
    name_plate="안정화 상판"+(f" (표시 ×{sc:g})" if sc>1 else "")
    fig.add_trace(mesh((Rp@v.T).T+pivot,i,j,k,PLATE_COLOR if controller_on else PLATE_OFF_COLOR,name_plate))
    ch=1.7*sc*0.45
    cargo=box(0,0,post_h+th+ch/2,0.55*sc,0.55*sc,ch); v,i,j,k=cargo
    fig.add_trace(mesh((Rp@v.T).T+pivot,i,j,k,"#F2C14E","상판 적재물"))
    # 수평 기준선(점선 테두리): 상판이 이 테두리와 겹치면 수평
    ref=np.array([[-1.4,-0.85],[1.4,-0.85],[1.4,0.85],[-1.4,0.85],[-1.4,-0.85]])*sc
    fig.add_trace(go.Scatter3d(x=ref[:,0]+pivot[0],y=ref[:,1]+pivot[1],z=np.full(5,pivot[2]+post_h+th),mode="lines",
                               line=dict(color="#E4572E",width=4,dash="dash"),name="수평 기준",hoverinfo="skip"))
    top=(Rp@np.array([0,0,post_h+th+ch+0.8*sc]))+pivot
    fig.update_layout(height=height,margin=dict(l=0,r=0,t=0,b=0),uirevision="ship-scene",
        legend=dict(orientation="h",yanchor="bottom",y=0.01,x=0.01,bgcolor="rgba(255,255,255,0.55)"),
        scene=dict(xaxis_title="선박 길이 방향(m)",yaxis_title="선박 폭 방향(m)",zaxis_title="높이(m)",
                   aspectmode="manual",aspectratio=dict(x=2.4,y=0.55,z=0.42),
                   camera=dict(eye=dict(x=-1.65,y=1.75,z=0.9)),bgcolor="#eaf5fb",
                   annotations=[dict(x=float(top[0]),y=float(top[1]),z=float(top[2]),showarrow=False,
                                     text=f"<b>상판</b> Roll {float(row['plate_roll']):+.2f}° · Pitch {float(row['plate_pitch']):+.2f}°",
                                     bgcolor="rgba(255,255,255,0.85)",bordercolor=PLATE_COLOR,font=dict(size=12,color="#0b3d3a"))]))
    return fig


def angle_figure(df, axis, title, height=260, window_s=HISTORY_S):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["time"],y=df[f"ship_{axis}"],name="보정 전(선박 자세)",line=dict(color=SHIP_COLOR,width=1.6)))
    fig.add_trace(go.Scatter(x=df["time"],y=df[f"plate_{axis}"],name="보정 후(상판 자세)",line=dict(color=PLATE_COLOR,width=2.2)))
    fig.add_hrect(y0=-1,y1=1,fillcolor=PLATE_COLOR,opacity=0.08,line_width=0)
    t_end=float(df["time"].iloc[-1]) if len(df) else 0.0
    fig.update_layout(title=dict(text=title,font=dict(size=14)),height=height,margin=dict(l=10,r=10,t=34,b=6),
                      yaxis_title="각도(°)",hovermode="x unified",showlegend=False)
    fig.update_xaxes(range=[max(0.0,t_end-window_s),max(window_s,t_end)])   # 최근 구간만 흘러가게 (시간축 고정 폭)
    fig.update_yaxes(title_standoff=4)
    return fig


def sensor_figure(df):
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=df["time"],y=df["acc_roll"],name="가속도계 환산각(하부)",line=dict(color="#9AA5B1",width=1)))
    fig.add_trace(go.Scatter(x=df["time"],y=df["est_roll"],name="상보필터 추정각",line=dict(color=PLATE_COLOR,width=2)))
    fig.add_trace(go.Scatter(x=df["time"],y=df["ship_roll"],name="INS Roll 기준",line=dict(color=SHIP_COLOR,width=1.4,dash="dot")))
    fig.update_layout(title="하부 IMU 모델: 측정값과 INS 자세 기준 비교(Roll)",height=260,
                      margin=dict(l=10,r=10,t=40,b=10),yaxis_title="각도(°)",xaxis_title="현재 구간 경과 시간(s)",hovermode="x unified")
    return fig


def live_performance(df, limit):
    rows=[]
    for axis,label in (("roll","Roll"),("pitch","Pitch")):
        s=df[f"ship_{axis}"].to_numpy(float); p=df[f"plate_{axis}"].to_numpy(float)
        sr=float(np.sqrt(np.mean(s*s))); pr=float(np.sqrt(np.mean(p*p)))
        rows.append({"축":label,"보정 전 RMS":f"{sr:.2f}°","보정 후 RMS":f"{pr:.2f}°",
                     "RMS 감소율":f"{100*(1-pr/sr):.1f}%" if sr>1e-8 else "-",
                     "보정 전 최대":f"{np.max(np.abs(s)):.2f}°","보정 후 최대":f"{np.max(np.abs(p)):.2f}°",
                     "서보 최대 사용각":f"{np.max(np.abs(df[f'joint_{axis}'])):.2f}° / ±{limit:.0f}°"})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit application
# -----------------------------------------------------------------------------
st.set_page_config(page_title="선박 2축 수평유지 시뮬레이터",page_icon="🌊",layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2.2rem !important; padding-bottom: 1rem !important; }
[data-stale="true"] { opacity: 1 !important; transition: none !important; }
.stPlotlyChart { transition: none !important; }
[data-testid="stMetric"] { padding: 6px 10px; border-left: 3px solid rgba(128,128,128,.30); }
[data-testid="stMetricLabel"] p { font-size: .85rem; }
</style>
""",unsafe_allow_html=True)
st.markdown("#### 🌊 선박 2축(롤·피치) 수평유지 시뮬레이터")
st.caption("가상 파도 시험 + Akademik Tryoshnikov 전체 항해 기록 자동 재생 + 제어 파라미터 자동 최적화")
st.caption("빌드 버전: V3-VERIFIED-ENGINE · 계산식 = verified_sim.py (verify_engine.py 56/56 통과) · 2026-11-27")

# Apply saved optimizer values before creating keyed widgets.
if st.session_state.get("apply_best_on_next_run"):
    res=st.session_state.get("optimization_result")
    if res:
        for key,value in res["best"].items(): st.session_state[f"param_{key}"]=float(value)
        st.session_state["parameters_applied"]=True
    st.session_state["apply_best_on_next_run"]=False

sb=st.sidebar
sb.header("운용 모드")
mode=sb.radio("데이터 입력 방식",["가상 파도 시뮬레이션","Akademik 전체 항해 데이터"],index=0,key="mode_select")
actual_mode=(mode=="Akademik 전체 항해 데이터")

# Persistent fixed/hardware + assumed sensor parameters
sb.header("안정화 장치 모델")
HW_PRESETS={  # 다음 학기 부품 데이터시트 기반 (근거는 앱 '계산식 출처' 표와 발표 자료)
    # 구매한 센서: GY-9250 모듈의 MPU-9250 = MPU-6500(자이로·가속도 다이) + AK8963 지자계.
    # PS-MPU-9250A-01 Rev 1.1: 자이로 Total RMS Noise 0.1 º/s-rms(92 Hz), ZRO 온도 변화 ±30 º/s(−40~+85°C → 약 ±0.24 º/s/°C),
    #   가속도 노이즈 300 µg/√Hz (8·9쪽). 지자계는 방위(요)용이라 롤·피치 수평유지에는 쓰지 않는다.
    "MPU-9250 + DS51150":   dict(gyro_noise=0.1,gyro_bias=0.3,acc_noise=0.2)}
def _apply_hw(name):
    v=HW_PRESETS[name]
    st.session_state.update({"fixed_tau":0.10,"fixed_rate":230,"fixed_limit":15.0,"sensor_gyro_noise":v["gyro_noise"],
                             "sensor_gyro_bias":v["gyro_bias"],"sensor_acc_noise":v["acc_noise"],"sensor_acc_disturb":1.0,
                             "servo_choice":"1차 지연 (정확 이산화)","delay_ms":20})
with sb.expander("서보 및 센서 고정 조건",expanded=False):
    st.caption("부품 프리셋 (서보 DS51150 12V: 230°/s·±15°·τ 0.10 s, 지연 20 ms, 가속도 오염 1°)")
    st.button("MPU-9250(GY-9250) + DS51150 값으로 되돌리기",on_click=_apply_hw,args=("MPU-9250 + DS51150",),key="btn_hw_mpu",width="stretch",
              help="자이로 노이즈 0.1°/s · 바이어스 0.3°/s(ZRO 온도 변화 ±30°/s/125°C ≈ ±0.24°/s/°C, 켠 뒤 약 1°C 변화 가정) · 가속도계 0.2°")
    tau=st.slider("서보 응답 시간상수 τ(s)",0.02,0.4,0.10,0.01,key="fixed_tau")
    rate=st.slider("서보 최대 각속도(°/s)",50,600,230,10,key="fixed_rate",help="기본값 = DS51150 12V (0.21 s/60° 무부하 → 부하 80% 가정)")
    limit=st.slider("서보 회전 한계(±°)",5.0,30.0,15.0,1.0,key="fixed_limit")
    gyro_noise=st.slider("가상 자이로 노이즈(°/s)",0.0,2.0,0.1,0.01,key="sensor_gyro_noise",help="기본값 = MPU-9250 Total RMS Noise 0.1 º/s-rms")
    gyro_bias=st.slider("가상 자이로 바이어스(°/s)",0.0,2.0,0.3,0.01,key="sensor_gyro_bias",
                        help="켤 때 정지 보정을 한 뒤에도 남는 바이어스(주로 온도 변화 × 온도계수). 데이터시트의 ZRO 온도계수로 정합니다.")
    acc_noise=st.slider("가속도계 환산각 노이즈(°)",0.0,3.0,0.2,0.05,key="sensor_acc_noise",help="기본값 = MPU-9250 300 µg/√Hz(≈0.1°)에 배 진동 여유 2배")
    accel_disturb=st.slider("추가 가속도 오염 진폭(°)",0.0,5.0,0.0,0.1,key="sensor_acc_disturb")
    servo_choice=st.radio("서보 모델",["1차 지연 (정확 이산화)","2차 (상판·적재물 관성 포함)"],index=0,key="servo_choice")
    servo2=servo_choice.startswith("2차")
    legacy_mode=False   # '기존 앱 결과 재현'(오일러 근사)은 UI에서 제거 — 엔진에는 남겨 verify_engine V11/V13 에서만 사용
    servo_fn=st.slider("2차 모델 고유진동수 fn(Hz)",1.0,8.0,3.0,0.5,key="servo_fn",disabled=not servo2)
    servo_zeta=st.slider("2차 모델 감쇠비 ζ",0.05,1.0,0.3,0.05,key="servo_zeta",disabled=not servo2)
    delay_ms=st.select_slider("센서→구동 지연(ms)",[0,20,40,60,80,100],value=20,key="delay_ms",disabled=legacy_mode,
                              help="50Hz 제어라 20ms 단위. 다음 학기에 명령→서보 움직임 시각 차이를 재서 넣으세요.")
    st.caption("2차 모델을 켜면 서보가 관성 때문에 목표각을 지나쳤다가 돌아올 수 있습니다(1차 모델은 넘어감 없음). "
               "fn·ζ는 가정값이며 τ 대신 쓰입니다. 실제 서보에 상판을 얹고 계단 응답을 재면 실측값으로 바꿀 수 있습니다.")
SERVO={"servo2":bool(servo2),"wn":2*math.pi*float(servo_fn),"zeta":float(servo_zeta),
       "servo_model":"zoh2" if servo2 else ("euler1" if legacy_mode else "zoh1"),
       "delay_steps":0 if legacy_mode else int(round(delay_ms/1000/CONTROL_DT)),
       "timing":"legacy" if legacy_mode else "consistent"}
sb.caption("센서 오차는 원본 ACE INS에서 직접 측정한 값이 아니라 시뮬레이터 가정값입니다.")

# Adjustable control params; can be filled by optimizer callback.
sb.header("선박")
_ship_choice=sb.selectbox("선박 (3D 표시·제원)",list(SHIPS.keys())+["직접 입력"],index=0,key="ship_choice",
                          help="선박 크기는 3D 그림과 제원 표시에만 쓰입니다. 흔들림 크기는 Akademik 실측(또는 그 회귀식)을 씁니다. "
                               "한나라호(133.0m)와 Akademik(133.5m)은 길이가 거의 같아 실측을 최소 선박 기준으로 쓸 수 있습니다.")
if _ship_choice=="직접 입력":
    SHIP={"name":"사용자 입력 선박","length":float(sb.number_input("전장(m)",20.0,400.0,133.0,1.0,key="ship_len")),
          "breadth":float(sb.number_input("폭(m)",5.0,60.0,19.4,0.1,key="ship_beam")),"displacement":None,"draft":None}
else:
    SHIP=dict(SHIPS[_ship_choice])
sb.caption("선박 크기를 바꿔도 계산된 흔들림은 바뀌지 않습니다. 물리 기반 운동 모델은 Akademik 제원과 실측으로 맞춘 계수를 씁니다.")
PHYS_TERMS=None
_PD=phys.default_params()
with sb.expander("선박 운동 물리 계수 (물리 기반 모델)",expanded=False):
    st.caption("[제원] AARI 공식값 · [실측] Akademik INS에서 계산 · [추정] 자료가 없어 가정한 값. 기본값으로 실측 롤 주기 13.5 s와 Hs별 롤·피치 RMS를 맞췄습니다.")
    _pg1=st.slider("GM_T 횡메타센터 높이 (m) [실측 역산]",0.5,4.0,float(_PD["GM_T"]),0.01,key="phys_gm",
                   help="실측 롤 주기 13.5 s를 IMO IS Code 2.3.4의 T=2CB/√GM 에 넣어 거꾸로 구한 값(약 1.70 m). 바꾸면 C44와 롤 고유주기가 바뀝니다.")
    _pg2=st.slider("롤 감쇠비 ζ44 [실측 맞춤]",0.01,0.40,float(_PD["zeta44"]),0.01,key="phys_z44",
                   help="Hs 구간별 실측 롤 RMS 중앙값에 맞춘 값(선속 10 kn, 파향 0~360° 평균 가정).")
    _pg3=st.slider("피치 감쇠비 ζ55 [실측 맞춤]",0.02,0.80,float(_PD["zeta55"]),0.01,key="phys_z55")
    _pg4=st.slider("롤 관성반경 kxx/B [추정]",0.25,0.50,float(_PD["kxx_B"]),0.01,key="phys_kxx",
                   help="건조 관성만. 부가관성 A44는 롤 주기가 13.5 s가 되도록 자동으로 정해집니다(바꾸면 A44가 같이 바뀜).")
    _pg5=st.slider("피치 관성반경 kyy/L [추정]",0.18,0.32,float(_PD["kyy_L"]),0.01,key="phys_kyy")
    _pg6=st.slider("선속 U (knots) [추정]",0.0,16.0,float(_PD["U_kn"]),0.5,key="phys_U",
                   help="만남 주파수 ωe = ω − kU·cosμ 에 들어갑니다. 0이면 정지 상태.")
    _pg7=st.slider("IMU 설치 높이 h (m, 회전 중심 위) [추정]",0.0,15.0,float(_PD["imu_h"]),0.5,key="phys_h",
                   help="높을수록 회전 가속도 h·φ''가 가속도계에 더 크게 섞입니다(레버암 오염).")
    st.caption("민감도 시험용 배율 (1이 기준)")
    _pm1,_pm2,_pm3=st.columns(3)
    _mI=_pm1.number_input("관성 ×",0.25,4.0,1.0,0.25,key="phys_mI")
    _mC=_pm2.number_input("복원 ×",0.25,4.0,1.0,0.25,key="phys_mC")
    _mB=_pm3.number_input("감쇠 ×",0.25,4.0,1.0,0.25,key="phys_mB")
PHYS_P=dict(_PD,GM_T=float(_pg1),zeta44=float(_pg2),zeta55=float(_pg3),kxx_B=float(_pg4),kyy_L=float(_pg5),
            U_kn=float(_pg6),imu_h=float(_pg7),mult_I=float(_mI),mult_C=float(_mC),mult_B=float(_mB))
PHYS_CO=phys.coefficients(PHYS_P)
sb.caption(f"물리 모델: 롤 고유주기 {PHYS_CO['T44']:.1f} s (실측 13.5 s) · 피치 {PHYS_CO['T55']:.1f} s · ζ44 {PHYS_CO['z44']:.2f} · ζ55 {PHYS_CO['z55']:.2f}")

sb.header("제어 파라미터")
with sb.expander("FF + PD + 상보필터",expanded=True):
    ff=st.slider("선행 보정 FF",0.0,1.2,1.0,0.01,key="param_ff")
    # 기본값 = 'WMO 6 설정' (MPU-9250 + DS51150 에서 WMO 6 을 통과하는 조합. 바이어스 추정을 끄면 잔잔한 바다에서
    #          자이로 바이어스 때문에 상판이 선박보다 더 기울 수 있음 — Hs 1.9 m: 선박 0.66° vs 상판 0.76°)
    kp=st.slider("P 이득 Kp",0.0,4.0,1.27,0.01,key="param_kp")
    kd=st.slider("D 이득 Kd(s)",0.0,0.30,0.02,0.005,key="param_kd")
    alpha=st.slider("상보필터 계수 α (20ms 기준)",0.90,0.999,0.998,0.001,format="%.3f",key="param_alpha")
    ki=st.slider("자이로 바이어스 추정 kᵢ (1/s²)",0.0,0.02,0.005,0.001,format="%.3f",key="param_ki",
                 help="0이면 끔. 켜면 상보필터가 자이로 바이어스를 스스로 추정해 뺍니다(PI형 상보필터, Mahony 외 2005 부록 A 식 22·23). "
                      "바이어스 오차가 사라지므로 α를 더 크게(자이로를 더 믿게) 해서 가속도 오염을 줄일 수 있습니다. 켠 뒤 수렴에 약 20~40초.")
    def _apply_wmo6():
        st.session_state.update({"param_ff":1.0,"param_kp":1.27,"param_kd":0.02,"param_alpha":0.998,"param_ki":0.005})
    st.button("제품 설정으로 되돌리기 (WMO 6 목표)",on_click=_apply_wmo6,key="btn_wmo6",width="stretch",
              help="FF 1.0 · Kp 1.27 · Kd 0.02 · α 0.998 · kᵢ 0.005 — 부품 교체 없이 MPU-9250(구매품)으로 WMO 6을 목표로 한 설정")
SERVO["ki"]=float(ki)   # 추정기 설정도 모든 평가(실시간·튜닝·한계 분석)에 같이 넘긴다

controller_on=sb.toggle("보정 작동",value=True,key="controller_on")

# Actual dataset mode: prominent main-page and sidebar download controls.
manifest=None; wave_df=None; archive_path=None; current_df=None; current_descriptor=None
if actual_mode:
    use_compact=compact_available()
    if use_compact:
        archive_path=CompactSource()
        manifest=build_manifest(archive_path)
        _wave_file=REPO_WAVE_CSV if REPO_WAVE_CSV.exists() else WAVE_CSV
        wave_df=read_wave_csv(_wave_file) if _wave_file.exists() else pd.DataFrame(columns=["timestamp","Hs","Tp"])
        st.success(f"저장소의 경량 데이터(data/)를 사용합니다 · 다운로드 없음 · 연속 재생 구간 {len(manifest):,}개 · 파랑 기록 {len(wave_df):,}개")
    else:
        st.subheader("Akademik 전체 항해 데이터")
        st.info(
            "실제 INS 전체 항해 기록과 재구성 Hs/Tp 자료를 한 번에 자동 다운로드합니다. "
            "준비가 끝나면 날짜순 연속 구간을 자동 재생하고, 각 구간 종료 후 다음 기록으로 넘어갑니다."
        )
        ready_now = dataset_ready()
        with st.container(border=True):
            st.markdown("#### 전체 항해 데이터 자동 다운로드 및 등록")
            status_cols = st.columns(2)
            status_cols[0].metric("INS 원본 ZIP", "준비 완료" if ready_now else "미다운로드", "약 646.9 MB")
            status_cols[1].metric("파고·주기 CSV", "준비 완료" if ready_now else "미다운로드", "약 1.6 MB")
            if ready_now:
                st.success("이 실행 환경의 캐시에 전체 데이터가 있습니다. 아래에서 인덱스 상태를 확인합니다.")
                download_clicked = False
            elif download_in_progress():
                info_dl = download_in_progress()
                st.info(f"다른 화면(세션)에서 다운로드가 진행 중입니다 (현재 {info_dl['mb']:.0f} MB / 약 647 MB). "
                        "동시에 받으면 파일이 섞이므로 버튼을 숨겼습니다. 1~2분 뒤 새로고침해 주세요.")
                st.button("새로고침", key="refresh_download_state")
                download_clicked = False
            else:
                st.warning("아직 이 실행 환경에 전체 데이터가 없습니다. 아래 버튼을 누르면 INS와 Hs/Tp 파일을 자동으로 받습니다.")
                main_download = st.button(
                    "전체 항해 데이터 자동 다운로드 및 등록 시작",
                    type="primary", width="stretch", key="download_full_archive_main",
                )
                side_download = sb.button(
                    "전체 항해 데이터 다운로드 시작",
                    type="primary", width="stretch", key="download_full_archive_sidebar",
                )
                download_clicked = bool(main_download or side_download)
            st.caption(
                "공식 출처: Zenodo ACE INS (646.9 MB) · Zenodo 재구성 Hs/Tp (1.6 MB). "
                "원본 ZIP은 실행 환경 캐시에 저장합니다. 다시 받지 않으려면 아래 '🔁' 안내대로 경량 데이터를 저장소 data/ 에 올리세요."
            )

        if not ready_now and download_clicked:
            try:
                with st.status("공개 ACE 데이터 다운로드 및 등록 중…", expanded=True) as status:
                    status.write("INS 전체 항해 ZIP과 파고·주기 CSV를 받고 체크섬(MD5)을 확인합니다. 체크섬이 틀리면 자동으로 한 번 더 받습니다.")
                    archive_path, wave_path = ensure_dataset(status)
                    status.write("② 날짜별 CSV를 읽어 시간 공백이 없는 연속 구간을 인덱싱합니다.")
                    pbar = st.progress(0.0, text="항해 기록 인덱싱 시작")
                    manifest = build_manifest(archive_path, pbar)
                    wave_df = read_wave_csv(wave_path)
                    status.update(
                        label=f"완료: {len(manifest):,}개 연속 구간 · 파랑 기록 {len(wave_df):,}개 등록",
                        state="complete", expanded=False,
                    )
                st.success("전체 항해 데이터 준비가 끝났습니다. 화면을 새로 고친 뒤 자동 재생을 시작합니다.")
                st.rerun()
            except Exception as exc:
                st.error(f"데이터 자동 다운로드/등록에 실패했습니다: {type(exc).__name__}: {exc}")
                st.info("네트워크 연결 또는 실행 환경의 디스크 용량을 확인한 뒤 다시 시도해 주세요. 불완전한 임시 파일은 자동 삭제됩니다.")
                st.stop()

        if not ready_now:
            st.stop()

        archive_path=ZipSource(INS_ARCHIVE)
        try:
            manifest=load_manifest(archive_path)
            wave_df=read_wave_csv(WAVE_CSV)
        except Exception as exc:
            st.error(f"저장된 데이터 인덱스를 읽지 못했습니다: {type(exc).__name__}: {exc}")
            if st.button("항해 데이터 인덱스 다시 만들기", key="rebuild_manifest"):
                MANIFEST_JSON.unlink(missing_ok=True)
                st.rerun()
            st.stop()
        st.success(
            f"전체 데이터 준비 완료 · 일별 파일 {len(set(x['member'] for x in manifest)):,}개 · "
            f"연속 재생 구간 {len(manifest):,}개 · 파랑 기록 {len(wave_df):,}개"
        )

    labels=[f"{i+1}/{len(manifest)} · {pd.Timestamp(d['start']).strftime('%Y-%m-%d %H:%M:%S')} UTC · {d['rows']:,} samples · {Path(d['member']).name}" for i,d in enumerate(manifest)]
    start_idx=sb.selectbox("처음 재생할 항해 구간",range(len(manifest)),format_func=lambda i:labels[i],index=0,key="actual_start_idx")
    auto_advance=sb.checkbox("구간 종료 시 다음 항해 기록 자동 재생",value=True,key="auto_advance")
    playback_speed=sb.slider("전체 항해 재생 배속",0.25,1000.0,20.0,0.25,key="actual_speed")
    refresh=sb.select_slider("화면 갱신 간격(s)",[0.15,0.25,0.4,0.6,0.8,1.0,1.5,2.0,3.0],value=0.6,key="refresh_actual",
                             help="깜빡이면 늘리세요. 간격을 늘려도 재생 배속(시뮬레이션 시간)은 그대로이고, 화면만 덜 자주 바뀝니다.")

    signature=f"actual|{archive_path.signature()}|{len(manifest)}|{start_idx}"
    if st.session_state.get("source_signature")!=signature:
        st.session_state["source_signature"]=signature
        st.session_state["current_segment_idx"]=int(start_idx)
        st.session_state["current_actual_df"]=load_segment(archive_path,manifest[start_idx])
        first=st.session_state["current_actual_df"].iloc[0]
        st.session_state["sim"]=new_sim_state((first.ship_roll,first.ship_pitch))
        st.session_state["last_tick"]=time.perf_counter()
        st.session_state["actual_finished"]=False
    current_idx=int(st.session_state.get("current_segment_idx",start_idx))
    if "current_actual_df" not in st.session_state:
        st.session_state["current_actual_df"]=load_segment(archive_path,manifest[current_idx])
    current_df=st.session_state["current_actual_df"]
    current_descriptor=manifest[current_idx]
    duration=float(current_df["elapsed_s"].iloc[-1])
    sb.caption(f"현재 구간 {current_idx+1}/{len(manifest)} · {len(current_df):,}개 INS 기록 · 중앙 샘플 간격 {current_descriptor['median_dt']:.2f}s")
    if len(manifest)>1:
        sb.caption("자동 다음 구간 이동이 켜져 있으면 선택한 시작 구간부터 마지막 기록까지 순서대로 재생합니다.")

else:
    # Virtual conditions
    sb.header("가상 해양환경")
    wmo_code=sb.radio("WMO 해상상태 단계",list(range(10)),index=4,horizontal=True,key="virtual_wmo",
                      help="WMO 해상상태 코드(Douglas 척도). 각 단계 Hs 구간의 가운데 값을 쓰고, 주기는 완전발달 해역(Pierson–Moskowitz) 관계 Tp≈4.99·√Hs로 정합니다.")
    _w=wmo_row(wmo_code); sea_name=f"WMO {wmo_code} {_w[3]}"
    _hs=WMO_REP_HS[wmo_code]
    sea=dict(Hs=_hs,Tp=round(pm_tp(_hs),2),direction=25,components=5,shock=min(15.0,2.0+2.5*_hs))
    _rng=f"{_w[1]:g}~{_w[2]:g} m" if math.isfinite(_w[2]) else f"{_w[1]:g} m 초과"
    sb.caption(f"**{sea_name}** · Hs 구간 {_rng} → 대표 Hs {_hs:g} m · Tp {sea['Tp']:.1f} s (PM)")
    if sb.toggle("파도 조건 직접 변경",value=False,key="custom_wave"):
        sea["Hs"]=sb.slider("유의파고 Hs(m)",0.0,16.0,float(sea["Hs"]),0.05,key="virtual_hs")
        sea["Tp"]=sb.slider("피크 주기 Tp(s)",2.0,20.0,float(max(2.0,sea["Tp"])),0.5,key="virtual_tp")
        sea["direction"]=sb.slider("파도 방향(°)",0,360,int(sea["direction"]),5,key="virtual_dir",
                                   help="선수에서 잰 파향: 0° = 선수 정면(맞바람 파도), 90° = 정횡, 180° = 선미(뒤에서 오는 파도)")
        sea_name=f"직접 설정 (WMO {classify_sea_state(sea['Hs'])[0]})"
    # 제품 기준: 선박 운동은 물리 기반 운동방정식만 사용 (가정 배율·실측 회귀식·흔들림 배율 제거)
    motion_model_key="phys"; fc,fp=1.0,0.5; gain=1.0
    sea["gain"]=1.0 if sea["Hs"]>0 else 0.0
    sb.caption("선박 운동: 물리 기반 운동방정식 (I+A)φ''+Bφ'+Cφ=M(t) — 계수는 '선박 운동 물리 계수'")
    comps=phys.wave_components_phys(sea["Tp"],sea["Hs"],PHYS_COMPONENTS)
    PHYS_TERMS=phys.component_terms(comps,sea["direction"],PHYS_P,PHYS_CO)
    playback_speed=sb.slider("재생 배속",0.25,20.0,1.0,0.25,key="virtual_speed")
    refresh=sb.select_slider("화면 갱신 간격(s)",[0.15,0.25,0.4,0.6,0.8,1.0,1.5,2.0,3.0],value=0.25,key="refresh_virtual",
                             help="깜빡이면 늘리세요. 간격을 늘려도 재생 배속(시뮬레이션 시간)은 그대로이고, 화면만 덜 자주 바뀝니다.")
    shock_roll=sb.button("Roll 충격",key="virtual_roll_shock")
    shock_pitch=sb.button("Pitch 충격",key="virtual_pitch_shock")
    signature=f"virtual|{sea_name}|{sea['Hs']}|{sea['Tp']}|{sea['direction']}|{gain}|{sea['gain']:.6f}|{motion_model_key}|{sorted(PHYS_P.items())}"
    if st.session_state.get("source_signature")!=signature:
        st.session_state["source_signature"]=signature
        st.session_state["sim"]=new_sim_state()
        st.session_state["last_tick"]=time.perf_counter()
    if "sim" not in st.session_state: st.session_state["sim"]=new_sim_state()
    if shock_roll: st.session_state["sim"]["shocks"].append({"t0":st.session_state["sim"]["t"],"axis":"roll","peak":sea["shock"]})
    if shock_pitch: st.session_state["sim"]["shocks"].append({"t0":st.session_state["sim"]["t"],"axis":"pitch","peak":0.7*sea["shock"]})
    st.session_state["actual_finished"]=False

# Reset button
if sb.button("현재 구간 / 가상 시험 처음부터",width="stretch",key="reset_sim"):
    if actual_mode and current_df is not None:
        first=current_df.iloc[0]
        st.session_state["sim"]=new_sim_state((first.ship_roll,first.ship_pitch))
        st.session_state["actual_finished"]=False
    else:
        st.session_state["sim"]=new_sim_state()
    st.session_state["last_tick"]=time.perf_counter()

# Main ship data and provenance
with st.expander(f"선박 기준 · 데이터 출처 ({SHIP['name']}, 전장 {SHIP['length']:.1f} m)",expanded=False):
  cols=st.columns(5)
  cols[0].metric("선박",SHIP["name"])
  cols[1].metric("전장",f"{SHIP['length']:.2f} m")
  cols[2].metric("선폭",f"{SHIP['breadth']:.2f} m")
  cols[3].metric("배수량",f"{SHIP['displacement']:,} t" if SHIP.get("displacement") else "-")
  cols[4].metric("흘수",f"{SHIP['draft']:.1f} m" if SHIP.get("draft") else "-")
  if SHIP.get("source"): st.caption("제원 출처: "+SHIP["source"])
  if actual_mode:
      st.info("실제 항해 모드: 선박 Roll/Pitch는 ACE INS 기록값입니다. IMU 측정·필터·서보·상판 응답은 캡스톤 장치의 수학적 모델이며 실제 하드웨어 측정값이 아닙니다.")
      st.caption("INS: Thomas & Pina Estany (2020), Zenodo DOI 10.5281/zenodo.3733228, CC BY 4.0. 공개 원자료는 보정되지 않았고 품질검사도 수행되지 않았습니다.")
      st.caption("Hs/Tp: Derkani et al. (2021), Zenodo DOI 10.5281/zenodo.4541564, CC BY 4.0. 선박 Heave 5분 구간과 RAO로 재구성한 해상상태 추정값입니다.")
  else:
      code,label=classify_sea_state(sea["Hs"])
      st.caption(f"가상 조건: {sea_name} · Hs {sea['Hs']:.2f}m · Tp {sea['Tp']:.1f}s · WMO {code if code is not None else '-'} ({label})")

playing=sb.toggle("자동 재생",value=True,key="auto_play")
SCENE_EVERY=sb.select_slider("3D 화면 갱신 간격(s)",["그래프와 같이",0.5,1.0,2.0,3.0],value=1.0,key="scene_every",
                             help="3D는 그리는 데 오래 걸려 가장 많이 깜빡입니다. 그래프·숫자는 '화면 갱신 간격'대로 흐르고, 3D만 이 간격으로 다시 그립니다.")
PLATE_SCALE=sb.slider("3D 상판 표시 배율 (×)",1,15,6,1,key="plate_scale",help="3D 화면에서만 상판·지지대·적재물을 키워 보입니다. 계산(제어·기울기)과는 무관하며 실제 상판은 2.8×1.7m입니다.")
VIEW_HEIGHT={"작게 (노트북)":400,"보통":470,"크게 (큰 모니터)":560}[sb.radio("3D·그래프 화면 높이",["작게 (노트북)","보통","크게 (큰 모니터)"],index=1,horizontal=True,key="view_height")]

# Live simulation fragment
@st.fragment(run_every=refresh)
def live_panel():
    now=time.perf_counter()
    elapsed_wall=min(max(now-st.session_state.get("last_tick",now),0.0),1.0)
    st.session_state["last_tick"]=now
    if "sim" not in st.session_state:
        st.session_state["sim"]=new_sim_state()
    sim=st.session_state["sim"]
    if actual_mode:
        sim_df=st.session_state["current_actual_df"]
        seg_idx=int(st.session_state["current_segment_idx"])
        seg_desc=manifest[seg_idx]
        seg_duration=float(sim_df["elapsed_s"].iloc[-1])
    else:
        sim_df=None; seg_duration=float("inf"); seg_idx=0; seg_desc=None

    if playing and not st.session_state.get("actual_finished",False):
        n_steps=min(10000,max(1,int(round(elapsed_wall*playback_speed/CONTROL_DT))))
        for _ in range(n_steps):
            if actual_mode:
                sim_df=st.session_state["current_actual_df"]
                seg_idx=int(st.session_state["current_segment_idx"])
                seg_duration=float(sim_df["elapsed_s"].iloc[-1])
                if sim["t"] >= seg_duration:
                    if auto_advance and seg_idx+1 < len(manifest):
                        next_idx=seg_idx+1
                        try:
                            next_df=load_segment(archive_path,manifest[next_idx])
                        except Exception as exc:
                            st.error(f"다음 항해 구간 로딩 실패: {exc}")
                            st.session_state["actual_finished"]=True
                            break
                        st.session_state["current_segment_idx"]=next_idx
                        st.session_state["current_actual_df"]=next_df
                        f=next_df.iloc[0]
                        sim=new_sim_state((f.ship_roll,f.ship_pitch))
                        st.session_state["sim"]=sim
                        continue
                    else:
                        st.session_state["actual_finished"]=True
                        break
                motion=actual_motion_at(sim["t"],sim_df)
            else:
                motion=virtual_motion(sim["t"],comps,sea["direction"],sea["gain"]*gain,sim["shocks"],PHYS_TERMS,PHYS_CO,PHYS_P["imu_h"])
            p={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha,"tau":tau,"rate":float(rate),"limit":limit,
               "gyro_noise":gyro_noise,"gyro_bias":gyro_bias,"acc_noise":acc_noise,**SERVO}
            row=control_step(sim,motion,p,controller_on,accel_disturb if actual_mode else 0.0,CONTROL_DT)
            sim["history"].append(row)
            keep=int(HISTORY_S/CONTROL_DT)
            if len(sim["history"])>keep: del sim["history"][:-keep]
        st.session_state["sim"]=sim

    sim=st.session_state["sim"]
    if not sim["history"]:
        if actual_mode:
            sim_df=st.session_state["current_actual_df"]
            motion=actual_motion_at(0.0,sim_df)
        else:
            motion=virtual_motion(0.0,comps,sea["direction"],sea["gain"]*gain,sim["shocks"],PHYS_TERMS,PHYS_CO,PHYS_P["imu_h"])
        p={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha,"tau":tau,"rate":float(rate),"limit":limit,
           "gyro_noise":gyro_noise,"gyro_bias":gyro_bias,"acc_noise":acc_noise,**SERVO}
        sim["history"].append(control_step(sim,motion,p,controller_on,accel_disturb if actual_mode else 0.0,CONTROL_DT))
        st.session_state["sim"]=sim
    df=pd.DataFrame(sim["history"])
    row=df.iloc[-1]
    def _rms(x): x=np.asarray(x,float); return float(np.sqrt(np.mean(x*x)))
    rr=0.5*(_rms(df.plate_roll)/max(_rms(df.ship_roll),0.1)+_rms(df.plate_pitch)/max(_rms(df.ship_pitch),0.1))
    metric_cols=st.columns(6)
    metric_cols[0].metric("선박 Roll",f"{row.ship_roll:+.2f}°")
    metric_cols[1].metric("선박 Pitch",f"{row.ship_pitch:+.2f}°")
    metric_cols[2].metric("상판 Roll",f"{row.plate_roll:+.2f}°")
    metric_cols[3].metric("상판 Pitch",f"{row.plate_pitch:+.2f}°")
    metric_cols[4].metric("RMS비 (최근 30초)",f"{rr:.3f}",help="상판 RMS ÷ 선박 RMS, 롤·피치 평균. 0이면 완전 수평, 1이면 효과 없음")
    info_line=""
    if actual_mode:
        idx=int(st.session_state["current_segment_idx"])
        dnow=st.session_state["current_actual_df"]
        durnow=float(dnow["elapsed_s"].iloc[-1])
        metric_cols[5].metric("항해 구간",f"{idx+1:,}/{len(manifest):,}")
        parts=[]
        if row.timestamp is not None:
            parts.append(f"INS {row.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        wave=nearest_wave(row.timestamp,wave_df)
        if wave and np.isfinite(wave["Hs"]):
            code,label=classify_sea_state(wave["Hs"])
            tp=f"Tp {wave['Tp']:.1f}s" if np.isfinite(wave["Tp"]) else "Tp 없음"
            parts.append(f"**Hs {wave['Hs']:.2f}m · {tp} · WMO {code if code is not None else '-'} ({label})**")
        else:
            parts.append("파랑 기록 없음(±180s)")
        parts.append(f"경과 {min(row.time,durnow):.0f}/{durnow:.0f}s · 배속 {playback_speed:g}x")
        info_line=" · ".join(parts)
    else:
        metric_cols[5].metric("진행 시간",f"{row.time:.1f}s")
    if info_line:
        st.caption(info_line)

    # 3D 화면과 Roll·Pitch 그래프를 한 화면에 (스크롤 없이)
    H=VIEW_HEIGHT
    plotted=df.iloc[::2] if len(df)>1 else df
    left,right=st.columns([1.35,1],gap="small")
    with left:
        _now=time.perf_counter(); _sig=(actual_mode,H,PLATE_SCALE,controller_on,st.session_state.get("source_signature"))
        _cache=st.session_state.get("scene_cache")
        if (_cache is None or SCENE_EVERY=="그래프와 같이" or _cache[2]!=_sig or _now-_cache[0]>=float(SCENE_EVERY)):
            _fig=scene_figure(row,actual_mode,comps if not actual_mode else None,sea["direction"] if not actual_mode else 0.0,controller_on,height=H,plate_scale=PLATE_SCALE)
            st.session_state["scene_cache"]=(_now,_fig,_sig)
        else:
            _fig=_cache[1]                      # 같은 그림을 그대로 다시 보내면 브라우저가 다시 그리지 않아 깜빡이지 않음
        st.plotly_chart(_fig,width="stretch",key="ship_scene_live",config={"displayModeBar":False})
    with right:
        st.plotly_chart(angle_figure(plotted,"roll","Roll: 선박(주황) vs 상판(청록)",height=H//2-4),width="stretch",key="roll_live")
        st.plotly_chart(angle_figure(plotted,"pitch","Pitch: 선박(주황) vs 상판(청록)",height=H//2-4),width="stretch",key="pitch_live")
    a,b=st.columns([1.1,1])
    with a:
        st.markdown("**안정화 성능 지표 (최근 30초)**")
        st.dataframe(live_performance(df,limit),hide_index=True,width="stretch")
        if (df[["joint_roll","joint_pitch"]].abs()>=limit-1e-6).any().any():
            st.warning("서보 회전 한계에 도달했습니다. 이 구간에서는 상판이 수평을 유지하지 못할 수 있습니다.")
        if actual_mode:
            st.caption("주의: INS 원자료는 1Hz입니다. 실시간 화면은 50Hz 제어 모델 입력을 위해 선형 보간했으며 실제 고주파 흔들림을 복원하지 않습니다.")
        else:
            st.caption("가상 파도는 파도 경사 기반 단순화 모델이며 실선 RAO 모델이 아닙니다.")
    with b:
        st.plotly_chart(sensor_figure(plotted),width="stretch",key="sensor_live")
    st.download_button("현재 구간 시뮬레이션 CSV 저장",df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="stabilizer_current_segment.csv",mime="text/csv",key=f"dl_current_{len(df)}_{row.time:.2f}")
    if actual_mode and st.session_state.get("actual_finished",False):
        st.success("전체 항해 데이터의 마지막 연속 구간까지 재생했습니다.")

live_panel()

with st.expander("📐 계산식 출처와 자체 검증 (발표·질의응답용)",expanded=False):
    st.markdown("""
| 블록 | 식 | 출처 (원문 위치) |
|---|---|---|
| 플랜트 구조 | 상판각 = 선박각 + 관절각 | Åström & Murray, *Feedback Systems* 2판, **12-20쪽 그림 12.13** (P₂=1) |
| 제어 법칙 | u = −(Kp·θ̂상판 + Kd·ω상판) − Kff·θ̂선박 | 같은 책 **11-2쪽 식 (11.1)·그림 11.1(b)**, 12-20쪽 그림 12.13 |
| 잔차(해석식) | (1 − Kff·G)/(1 + G(Kp + Kd·s)) | 같은 책 **12-19쪽 식 (12.11)** |
| Kff = 1 근거 | F_v = 1/P₁ 의 저주파 근사 | 같은 책 12-19쪽 식 (12.12), 12-20쪽 식 (12.13) |
| 보완필터 | θ̂ = α(θ̂ + ω·dt) + (1−α)·θ_acc, α = τ/(τ+dt) | Higgins (1975) **322쪽 그림 2(B)** 후진 오일러 이산화 · arXiv 1508.04886 **10쪽 식 (3)** |
| 가속도계 | θ_acc = atan2(f_y, f_z), 가속 오염 포함 | NXP AN3461 **5쪽 식 1, 10쪽 식 25** |
| 선박 운동방정식 | (I+A)φ'' + Bφ' + Cφ = M(t),  ω0² = C/(I+A),  ζ = B/(2√((I+A)C)) | Journée & Massie, Offshore Hydromechanics (TU Delft 2001) **식 6.28 (6-13쪽), 식 6.38b·6.39 (6-15쪽)** |
| 복원 계수 | C44 = ρg∇·GM_T,  C55 = ρg∇·GM_L | 같은 책 **6-15쪽** |
| 만남 주파수 | ωe = ω − kU·cos μ | 같은 책 **식 6.6 (6-3쪽)** |
| 롤 주기 ↔ GM | T = 2CB/√GM,  C = 0.373 + 0.023(B/d) − 0.043(L/100) | IMO 2008 IS Code (MSC.267(85)) **Part A 2.3.4, 14쪽** — 실측 롤 주기 13.5 s로 GM 역산 |
| 선박 제원 | L 133.53 m · B 23.25 m · T 8.5 m · Δ 16,539 t | AARI(선주) 공식 제원표 |
| 파랑 외력 | M = C·k·a·(sinβ|cosβ)·F(x)·e^(−kT),  F(x) = 3(sin x − x cos x)/x³ | 정적 극한(긴 파도에서 수면 기울기 추종)과 상자형 수선면 평균을 이 앱에서 유도 · e^(−kT)는 J&M 식 6.75의 인자 (근사, V15에서 검산) |
| 바이어스 추정 (kᵢ>0) | ω_c = ω − b̂,  b̂ ← b̂ − kᵢ(θ_acc − θ̂)·dt  (연속형 x̂' = y_u − b̂ + k(y_x − x̂), b̂' = −(y_x − x̂)/λ) | Mahony·Hamel·Pflimlin, 44th IEEE CDC-ECC 2005, **부록 A 식 (22)·(23), 1484쪽** (보상기 C(s) = k + 1/(λs)) |
| 1차 서보 | j[k+1] = a·j[k] + (1−a)·u[k], a = e^(−dt/τ) | UW ME547 **슬라이드 9** (ZOH 이산화), 시정수 63.2% |
| 2차 서보 | j'' = ωn²(u − j) − 2ζωn·j' (정확 이산화) | Toronto Metropolitan 교재 **7.2.1절 식 7-7, 7-10** |
| 지연 | u[k] = cmd[k − n] | 12-20쪽 'Difficulties with Feedforward' (지연이 있으면 이상적 보상 불가) |
""")
    st.caption("원문 캡처는 수식_출처_원문캡처.pdf, 전체 시험(56개)은 verify_engine.py 를 실행하면 됩니다. "
               "이 식들로 확인되는 것은 '시뮬레이션이 근거 있는 식을 맞게 계산한다'까지이고, 실제 장치와의 일치는 실측 비교가 필요합니다.")
    if st.button("지금 설정으로 자체 검증 실행",key="run_self_test"):
        cur={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha,"tau":tau,"wn":2*math.pi*float(servo_fn),"zeta":float(servo_zeta),
             "servo_model":SERVO["servo_model"],"delay_steps":SERVO["delay_steps"],"timing":SERVO["timing"]}
        with st.spinner("시험 중…"):
            res=vs.self_test(cur)
        st.dataframe(pd.DataFrame([{"시험":a,"출처":b,"코드 값":float(c),"해석 값":float(d),"판정":"통과" if e else "실패"} for a,b,c,d,e in res]),
                     hide_index=True,width="stretch")

with st.expander("🚢 물리 기반 선박 운동 모델 — 계수·검증·이전 모델과 비교",expanded=False):
    _co=PHYS_CO; _P=PHYS_P; _M=phys.AKADEMIK_MEASURED
    st.markdown("Roll·Pitch를 각각 **(I+A)·φ'' + B·φ' + C·φ = M(t)** 로 풉니다. 파랑 외력 M = C × (유효 파경사)이고, "
                "선형 방정식이라 파도 성분마다의 정상해를 더하면 정확한 해가 됩니다. 계산 결과(각도·각속도·각가속도)가 "
                "기존 IMU 모델 → 상보필터 → FF+PD → DS51150 서보 → 상판으로 그대로 들어갑니다.")
    # ① 계수 표
    st.markdown("##### ① 계수와 출처 구분")
    _rows=[("배수량 Δ",f"{_P['disp_t']:,.0f}","t","제원","AARI 공식 제원표"),
           ("길이 L · 폭 B · 흘수 T",f"{_P['L']:g} · {_P['B']:g} · {_P['T']:g}","m","제원","AARI 공식 제원표"),
           ("배수 체적 ∇ = Δ/ρ",f"{_co['vol']:,.0f}","m³","계산","ρ = 1025 kg/m³(표준 해수)"),
           ("GM_T",f"{_P['GM_T']:.2f}","m","실측 역산","실측 롤 주기 13.5 s + IMO IS Code 2.3.4 (T = 2CB/√GM)"),
           ("C44 = ρg∇·GM_T",f"{_co['C44']:.3e}","N·m/rad","계산","Journée & Massie 6-15쪽"),
           ("I44 = Δ·(kxx·B)²",f"{_co['I44']:.3e}","kg·m²","추정",f"kxx = {_P['kxx_B']:.2f}·B (관례 범위 0.35~0.40B)"),
           ("A44 부가관성",f"{_co['A44']:.3e} (I44의 {_co['A44']/max(_co['I44'],1):.0%})","kg·m²","실측 역산","총 관성 = C44·(13.5/2π)² 에서 I44를 뺀 값"),
           ("B44 = 2ζ44·√((I+A)·C)",f"{_co['B44']:.3e}","N·m·s/rad","실측 맞춤",f"ζ44 = {_P['zeta44']:.2f} (Hs별 롤 RMS에 맞춤)"),
           ("롤 고유주기",f"{_co['T44']:.2f}","s","계산",f"실측 스펙트럼 첨두 {_M['roll_period_s']} s"),
           ("GM_L ≈ BM_L = CIL·B·L³/∇",f"{_co['GM_L']:.0f}","m","추정",f"CIL = {_P['CIL']} (상자형 1/12보다 작게)"),
           ("C55 = ρg∇·GM_L",f"{_co['C55']:.3e}","N·m/rad","계산","Journée & Massie 6-15쪽"),
           ("I55 = Δ·(kyy·L)²",f"{_co['I55']:.3e}","kg·m²","추정",f"kyy = {_P['kyy_L']:.2f}·L"),
           ("A55 부가관성",f"{_co['A55']:.3e}","kg·m²","추정",f"I55 × {_P['a55_ratio']:g}"),
           ("B55",f"{_co['B55']:.3e}","N·m·s/rad","실측 맞춤",f"ζ55 = {_P['zeta55']:.2f} (Hs별 피치 RMS에 맞춤)"),
           ("피치 고유주기",f"{_co['T55']:.2f}","s","계산","실측에서는 공진이 보이지 않음(강한 감쇠) — 확인 불가"),
           ("선속 U",f"{_P['U_kn']:g}","kn","추정","압축 INS 자료에 속도 없음, ACE 순항 속도 가정"),
           ("IMU 높이 h",f"{_P['imu_h']:g}","m","추정","레버암 가속도 오염 h·φ''")]
    st.dataframe(pd.DataFrame(_rows,columns=["항목","값","단위","구분","근거"]),hide_index=True,width="stretch")
    if any(abs(_P[k]-1.0)>1e-9 for k in ("mult_I","mult_C","mult_B")):
        st.warning(f"민감도 배율 적용 중: 관성 ×{_P['mult_I']:g} · 복원 ×{_P['mult_C']:g} · 감쇠 ×{_P['mult_B']:g} → 롤 고유주기 {_co['T44']:.1f} s, ζ44 {_co['z44']:.3f}")
    # ② 물리적 타당성 지표
    st.markdown("##### ② 물리적 타당성 지표")
    _t=np.arange(0,150,CONTROL_DT); _x,_=phys.free_decay(_t,5.0,_co["w44"],_co["z44"])
    _Td=phys.zero_cross_period(_t,_x,demean=False); _dl,_zd=phys.log_decrement(_t,_x)
    _long=phys.rao_curves(_P,90.0,[200.0],_co)["roll_per_slope"][0]
    _rp=[];_mm=[]
    for (lo,hi),n,tpm,rm,pm in _M["rms_by_hs"]:
        r_,p_=phys.rms_expected(0.5*(lo+hi),tpm,0.0,_P,headings=range(0,360,30)); _rp.append((r_,p_)); _mm.append((rm,pm))
    _err=np.mean([abs(math.log(a/b)) for (a,_),(b,_) in zip(_rp,_mm)])
    q1,q2,q3,q4=st.columns(4)
    q1.metric(f"롤 고유주기 (실측 {_M['roll_period_s']} s)",f"{_co['T44']:.1f} s")
    q2.metric("자유감쇠 시험 ζ (대수감쇠율)",f"{_zd:.3f}",help=f"감쇠 주기 {_Td:.2f} s. 입력 ζ44 {_co['z44']:.3f}와 같아야 정상")
    q3.metric("긴 파도에서 롤/파경사",f"{_long:.3f}",help="파장이 아주 길면 선박이 수면 기울기를 그대로 따라가야 하므로 1에 가까워야 정상")
    q4.metric("롤 RMS 평균 오차 (Hs 4구간)",f"{(math.exp(_err)-1)*100:.0f}%",help="실측 30분 블록 중앙값 대비 모델(선속·파향 평균) — 로그 오차의 평균")
    _cmp=pd.DataFrame([{"Hs 구간(m)":f"{lo:g}~{hi:g}","실측 블록 수":n,"실측 Tp(s)":tpm,"롤 RMS 실측(°)":rm,"롤 RMS 모델(°)":round(r_,2),
                        "피치 RMS 실측(°)":pm,"피치 RMS 모델(°)":round(p_,2)} for ((lo,hi),n,tpm,rm,pm),(r_,p_) in zip(_M["rms_by_hs"],_rp)])
    st.dataframe(_cmp,hide_index=True,width="stretch")
    st.caption("실측: Akademik INS 30분 블록(평균 경사를 뺀 표준편차)의 중앙값. 모델: 같은 Hs·Tp, 선속 사이드바 값, 파향 0~360° 평균.")
    # ③ RAO와 민감도
    st.markdown("##### ③ 주파수 응답 (관성·복원·감쇠를 바꿨을 때)")
    r1,r2=st.columns([1,1.4])
    _beta=r1.select_slider("RAO 파향 (선수에서, °)",[0,30,45,60,90,120,135,150,180],value=45,key="phys_rao_beta",
                           help="0°(선수)는 피치만, 90°(정횡)는 롤만 생깁니다. 45°에서는 둘 다 보입니다.")
    _sens=r2.radio("민감도 곡선",["관성","복원","감쇠"],horizontal=True,key="phys_rao_sens")
    _Tg=np.linspace(4,25,160); _fr=go.Figure(); _fq=go.Figure()
    for m_,dash in ((0.5,"dot"),(1.0,"solid"),(2.0,"dash")):
        _q=dict(_P,**{{"관성":"mult_I","복원":"mult_C","감쇠":"mult_B"}[_sens]:m_*_P[{"관성":"mult_I","복원":"mult_C","감쇠":"mult_B"}[_sens]]})
        _c=phys.rao_curves(_q,_beta,_Tg)
        _fr.add_trace(go.Scatter(x=_Tg,y=_c["roll_per_slope"],name=f"{_sens} ×{m_:g}",line=dict(dash=dash,color=SHIP_COLOR)))
        _fq.add_trace(go.Scatter(x=_Tg,y=_c["pitch_per_slope"],name=f"{_sens} ×{m_:g}",line=dict(dash=dash,color=PLATE_COLOR)))
    _fr.add_vline(x=_M["roll_period_s"],line_dash="dot",line_color="#888",annotation_text="실측 롤 주기 13.5 s",annotation_font_size=10)
    for f_,ttl in ((_fr,"Roll: 롤 진폭 / 파경사 진폭"),(_fq,"Pitch: 피치 진폭 / 파경사 진폭")):
        f_.update_layout(height=280,margin=dict(l=10,r=10,t=36,b=10),title=dict(text=ttl,font=dict(size=13)),
                         xaxis_title="파도 주기(s) — 선속이 있으면 만남 주기와 다름",yaxis_title="배",legend=dict(orientation="h",y=-0.3))
    g1,g2=st.columns(2); g1.plotly_chart(_fr,width="stretch",key="phys_rao_roll"); g2.plotly_chart(_fq,width="stretch",key="phys_rao_pitch")
    st.caption("관성↑ → 고유주기가 길어져 공진이 긴 파도 쪽으로 이동 · 복원↑ → 고유주기가 짧아짐 · 감쇠↑ → 공진 봉우리가 낮아짐. "
               "긴 파도(오른쪽 끝)에서 1에 가까워지는 것은 선박이 수면 기울기를 따라가는 정적 극한입니다.")
    # ④ 이전 모델과 시간 응답 비교
    st.markdown("##### ④ 같은 파도에서 이전 모델 vs 물리 모델")
    _d1,_d2,_d3=st.columns(3)
    _hs0=float(sea["Hs"]) if not actual_mode else 3.25
    _hs=_d1.number_input("Hs (m)",0.1,16.0,_hs0 if _hs0>0 else 3.25,0.25,key="phys_cmp_hs")
    _tp=_d2.number_input("Tp (s)",2.0,20.0,round(pm_tp(_hs),1),0.5,key="phys_cmp_tp")
    _d0=90 if actual_mode else int(sea["direction"])%360
    _dir=_d3.number_input("파향 (선수에서, °)",0,180,_d0 if _d0<=180 else 360-_d0,15,key="phys_cmp_dir")
    _wins={"가정 배율 (이전 앱)":virtual_window(_hs,_tp,0,300.0,1.0,_dir),
           "실측 회귀식 (이전)":virtual_window(_hs,_tp,0,300.0,motion_mult(_hs,_tp,"fit",AKADEMIK_FIT["c"],AKADEMIK_FIT["p"]),_dir),
           "물리 기반 운동방정식":virtual_window(_hs,_tp,0,300.0,1.0,_dir,phys_p=_P)}
    _cols={"가정 배율 (이전 앱)":"#9AA5B1","실측 회귀식 (이전)":"#E4A72E","물리 기반 운동방정식":"#123f70"}
    _ft=go.Figure(); _rows2=[]
    for nm,w_ in _wins.items():
        m_=w_["t"]<=120
        _ft.add_trace(go.Scatter(x=w_["t"][m_],y=w_["ship"][0][m_],name=f"{nm} Roll",line=dict(color=_cols[nm],width=2 if "물리" in nm else 1)))
        _ft.add_trace(go.Scatter(x=w_["t"][m_],y=w_["ship"][1][m_],name=f"{nm} Pitch",line=dict(color=_cols[nm],dash="dot",width=1.5 if "물리" in nm else 1),visible="legendonly"))
        _rows2.append({"모델":nm,"롤 RMS(°)":float(np.sqrt(np.mean(w_["ship"][0]**2))),"피치 RMS(°)":float(np.sqrt(np.mean(w_["ship"][1]**2))),
                       "롤 최대(°)":float(np.max(np.abs(w_["ship"][0]))),"롤 평균 주기(s)":phys.zero_cross_period(w_["t"],w_["ship"][0]),
                       "피치 평균 주기(s)":phys.zero_cross_period(w_["t"],w_["ship"][1]),"가속도 오염 RMS(°)":float(np.sqrt(np.mean(w_["acc"][0]**2)))})
    _ft.update_layout(height=300,margin=dict(l=10,r=10,t=30,b=10),xaxis_title="시간(s)",yaxis_title="각도(°)",legend=dict(orientation="h",y=-0.25))
    st.plotly_chart(_ft,width="stretch",key="phys_cmp_time")
    st.dataframe(pd.DataFrame(_rows2).round(2),hide_index=True,width="stretch")
    st.caption("이전 모델은 파도 기울기에 배율만 곱해서 짧은 파도까지 그대로 따라갑니다. 물리 모델은 롤 고유주기(13.5 s) 근처 성분을 증폭하고 "
               "선체보다 짧은 파도는 관성 때문에 걸러냅니다. 그래서 파도가 짧고 선수 쪽에서 오면(기본값 Hs 1.9 m·25°) 거의 흔들리지 않고, "
               "정횡·긴 파도일수록 크게 흔들립니다. 실측(30분 블록)에서 롤 스펙트럼 첨두가 13~14 s에 모이는 것과 같은 성질입니다.")
    # ⑤ Hs·Tp 변화에 따른 응답
    st.markdown("##### ⑤ 파고·주기 변화에 따른 응답")
    _hsg=np.array([0.3,0.875,1.875,3.25,5.0,7.5,11.5]); _ph=[phys.rms_expected(h,pm_tp(h),0.0,_P,headings=range(0,360,30)) for h in _hsg]
    _fh=go.Figure()
    _fh.add_trace(go.Scatter(x=_hsg,y=[math.hypot(a,b) for a,b in _ph],name="물리 모델 (파향 평균, Tp=PM)",mode="lines+markers",line=dict(color="#123f70",width=3)))
    _fh.add_trace(go.Scatter(x=_hsg,y=AKADEMIK_FIT["c"]*_hsg**AKADEMIK_FIT["p"],name="실측 회귀식 c·Hsᵖ (이전)",line=dict(color="#E4A72E",dash="dash")))
    _fh.add_trace(go.Scatter(x=_hsg,y=[tilt_rms(virtual_window(h,pm_tp(h),0,180.0,1.0)) for h in _hsg],name="가정 배율 (이전 앱)",line=dict(color="#9AA5B1",dash="dot")))
    _fh.add_trace(go.Scatter(x=[0.5*(lo+hi) for (lo,hi),*_ in _M["rms_by_hs"]],y=[math.hypot(rm,pm) for _,_,_,rm,pm in _M["rms_by_hs"]],
                             name="실측 (30분 블록 중앙값)",mode="markers",marker=dict(size=11,color="#E4572E",symbol="diamond")))
    _fh.update_layout(height=300,margin=dict(l=10,r=10,t=36,b=10),title=dict(text="선박 기울기 RMS vs Hs",font=dict(size=13)),
                      xaxis_title="Hs(m)",yaxis_title="√(롤²+피치²) RMS (°)",legend=dict(orientation="h",y=-0.3))
    _tpg=np.linspace(4,18,29); _pt=[phys.rms_expected(_hs,tp_,_dir,_P) for tp_ in _tpg]
    _ftp=go.Figure()
    _ftp.add_trace(go.Scatter(x=_tpg,y=[a for a,_ in _pt],name="Roll RMS",line=dict(color=SHIP_COLOR)))
    _ftp.add_trace(go.Scatter(x=_tpg,y=[b for _,b in _pt],name="Pitch RMS",line=dict(color=PLATE_COLOR)))
    _ftp.update_layout(height=300,margin=dict(l=10,r=10,t=36,b=10),title=dict(text=f"Hs {_hs:g} m·파향 {_dir}°에서 Tp만 바꿀 때",font=dict(size=13)),
                       xaxis_title="Tp(s)",yaxis_title="RMS(°)",legend=dict(orientation="h",y=-0.3))
    h1,h2=st.columns(2); h1.plotly_chart(_fh,width="stretch",key="phys_hs_fig"); h2.plotly_chart(_ftp,width="stretch",key="phys_tp_fig")
    # ⑥ 자유 감쇠
    st.markdown("##### ⑥ 롤 자유 감쇠 시험 (관성·복원·감쇠가 한눈에 보이는 시험)")
    _fd=go.Figure(); _fd.add_trace(go.Scatter(x=_t[_t<=90],y=_x[_t<=90],name="롤 (초기 5°)",line=dict(color="#123f70")))
    _env=5.0*np.exp(-_co["z44"]*_co["w44"]*_t[_t<=90])
    _fd.add_trace(go.Scatter(x=_t[_t<=90],y=_env,name="감쇠 포락선 e^(−ζωt)",line=dict(color="#E4572E",dash="dot")))
    _fd.update_layout(height=260,margin=dict(l=10,r=10,t=30,b=10),xaxis_title="시간(s)",yaxis_title="롤(°)",legend=dict(orientation="h",y=-0.3))
    st.plotly_chart(_fd,width="stretch",key="phys_decay")
    st.caption(f"측정 주기 {_Td:.2f} s (이론 감쇠 주기 {2*math.pi/(_co['w44']*math.sqrt(1-_co['z44']**2)):.2f} s) · 대수감쇠율 δ {_dl:.3f} → ζ {_zd:.3f}. "
               "실제 배에서는 롤 감쇠 시험(기울였다 놓기)으로 이 값을 확인합니다.")
    # ⑦ 보상 성능 비교
    st.markdown("##### ⑦ 2축 플랫폼 보상 성능: 이전 모델 vs 물리 모델 (현재 제어·센서·서보 설정)")
    if st.button("보상 성능 비교 실행",key="phys_comp_run"):
        _par={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha}
        _fx={"tau":tau,"rate":float(rate),"limit":limit,"gyro_noise":gyro_noise,"gyro_bias":gyro_bias,"acc_noise":acc_noise,"accel_disturb":0.0,**SERVO}
        _out=[]
        for nm in _wins:
            if "물리" in nm:
                ws=[virtual_window(_hs,_tp,r,300.0,1.0,PHYS_HEADINGS[r],phys_p=_P) for r in range(3)]
            elif "회귀" in nm:
                ws=[virtual_window(_hs,_tp,r,300.0,motion_mult(_hs,_tp,"fit",AKADEMIK_FIT["c"],AKADEMIK_FIT["p"]),PHYS_HEADINGS[r]) for r in range(3)]
            else:
                ws=[virtual_window(_hs,_tp,r,300.0,1.0,PHYS_HEADINGS[r]) for r in range(3)]
            mm=simulate_batch(ws,[NOISE_SEED_BASE+r for r in range(3)],[_par],_fx,keep_p95=False,warmup_s=LIMIT_WARMUP_S)
            sr=np.mean([tilt_rms_after(w_,LIMIT_WARMUP_S) for w_ in ws]); pr_=float(np.mean(np.hypot(mm["roll_rms"][0],mm["pitch_rms"][0])))
            _out.append({"선박 모델":nm,"선박 기울기 RMS(°)":sr,"상판 잔류 RMS(°)":pr_,"RMS비":pr_/max(sr,1e-9),
                         "롤 잔류 RMS(°)":float(np.mean(mm["roll_rms"][0])),"피치 잔류 RMS(°)":float(np.mean(mm["pitch_rms"][0])),
                         "상판 최대(°)":float(np.max(np.maximum(mm["roll_max"][0],mm["pitch_max"][0]))),"서보 포화율(%)":float(100*np.max(mm["saturation"][0]))})
        st.session_state["phys_comp"]=(pd.DataFrame(_out),_hs,_tp)
    if st.session_state.get("phys_comp"):
        _dfc,_h_,_t_=st.session_state["phys_comp"]
        st.dataframe(_dfc.round(3),hide_index=True,width="stretch")
        st.caption(f"Hs {_h_:g} m · Tp {_t_:g} s · 파향 45/90/135° 3회 · 300 s(처음 {LIMIT_WARMUP_S:.0f} s 제외). "
                   "보정 전 = 선박 기울기, 보정 후 = 상판 잔류 기울기.")

with st.expander("🌊 WMO 단계별 보정 한계 — 우리 장치는 몇 단계까지 수평을 유지하나",expanded=False):
    st.markdown("현재 사이드바의 제어 파라미터·서보·센서 설정 그대로, 해상상태만 바꿔 가며 상판이 기준을 만족하는지 봅니다. "
                "**한 번에 하나만 바꿉니다**(Hs만, 또는 Tp만). 판정 기준은 아래에서 정합니다.")
    lc1,lc2,lc3=st.columns(3)
    crit_rms=lc1.number_input("상판 기울기 RMS 허용(°)",0.05,10.0,1.0,0.05,key="lim_rms",help="√(롤 RMS²+피치 RMS²)")
    crit_max=lc2.number_input("상판 최대 기울기 허용(°)",0.1,30.0,3.0,0.1,key="lim_max")
    crit_sat=lc3.number_input("서보 포화율 허용(%)",0.0,50.0,1.0,0.5,key="lim_sat")
    lim_params={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha}
    lim_fixed={"tau":tau,"rate":float(rate),"limit":limit,"gyro_noise":gyro_noise,"gyro_bias":gyro_bias,
               "acc_noise":acc_noise,"accel_disturb":0.0,**SERVO}
    st.caption(f"평가 파라미터: FF {ff:.2f} · Kp {kp:.2f} · Kd {kd:.3f} · α {alpha:.3f} · 서보 {SERVO['servo_model']} · 지연 {SERVO['delay_steps']*20} ms")
    _pack=st.session_state.get("tune_pack")
    _fit_live=fit_tilt_power(_pack["items"]) if _pack else None
    _fit=_fit_live or AKADEMIK_FIT

    st.markdown("#### ① 가상 해상상태 기준")
    v1,v2=st.columns([1.3,1])
    sweep=v1.radio("바꿀 변수",["WMO 0~9 대표값 (Hs 구간 가운데, Tp=PM)","Hs 촘촘히 0.25~16 m (Tp=PM)","Hs만 변경 (Tp 고정)","Tp만 변경 (Hs 고정)"],key="lim_sweep")
    lm_key="phys"; lc_,lp_=_fit["c"],_fit["p"]   # 제품 기준: 물리 기반 운동방정식만
    v2.caption("선박 흔들림: 물리 기반 운동방정식 · 실현마다 파향 45°·90°·135°(선수 사파·정횡·선미 사파)")
    v3,v4,v5=st.columns(3)
    n_real=v3.number_input("조건당 파도 실현 수",1,10,3,1,key="lim_nreal")
    dur=v4.number_input("조건당 길이(s)",120,900,300,30,key="lim_dur",help=f"처음 {LIMIT_WARMUP_S:.0f}초는 필터·바이어스 추정 수렴 구간이라 판정에서 뺍니다.")
    if sweep.startswith("Hs만"):
        tp_fix=v5.number_input("고정 Tp(s)",2.0,20.0,8.0,0.5,key="lim_tpfix")
        conds=[(h,tp_fix) for h in (0.25,0.5,0.75,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0,8.0,10.0,12.0,14.0,16.0)]
    elif sweep.startswith("Tp만"):
        hs_fix=v5.number_input("고정 Hs(m)",0.1,16.0,2.0,0.1,key="lim_hsfix")
        conds=[(hs_fix,t) for t in (4,5,6,7,8,9,10,11,12,13,14,16,18)]
    elif sweep.startswith("Hs 촘촘히"):
        conds=[(h,pm_tp(h)) for h in (0.25,0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0,5.0,6.0,7.0,8.0,9.0,10.0,12.0,14.0,16.0)]
    else:
        conds=[(WMO_REP_HS[c],pm_tp(WMO_REP_HS[c])) for c in range(10)]
    if st.button("가상 해상상태 한계 분석 실행",key="run_limit_virtual",type="primary"):
        pb=st.progress(0.0,text="평가 중")
        dfv=limit_sweep(conds,lim_params,lim_fixed,lm_key,lc_,lp_,1.0,int(n_real),float(dur),lambda f: pb.progress(f,text=f"평가 중 {f*100:.0f}%"),
                        phys_p=PHYS_P)
        st.session_state["lim_virtual"]=(dfv,sweep,lm_key)
    if st.session_state.get("lim_virtual"):
        dfv,sw_used,lm_used=st.session_state["lim_virtual"]
        dfv=judge(dfv,crit_rms,crit_max,crit_sat)
        fig=go.Figure()
        xcol="Tp(s)" if sw_used.startswith("Tp만") else "Hs(m)"
        dplot=dfv[dfv[xcol]>0]   # 로그축에 Hs=0(WMO 0)을 그리면 축 범위가 무한대로 늘어나므로 표에만 남긴다
        if xcol=="Hs(m)":
            for code,lo,hi,name in WMO_CODES[1:]:
                hi_=min(hi,17.0)
                if lo<hi_: fig.add_vrect(x0=max(lo,0.01),x1=hi_,fillcolor="#2EC4B6" if code%2 else "#9AA5B1",opacity=0.07,line_width=0,
                                         annotation_text=f"WMO {code}",annotation_position="top left",annotation_font_size=10)
        fig.add_trace(go.Scatter(x=dplot[xcol],y=dplot["선박 기울기 RMS(°)"],name="선박 기울기 RMS",mode="lines+markers",line=dict(color=SHIP_COLOR)))
        fig.add_trace(go.Scatter(x=dplot[xcol],y=dplot["상판 기울기 RMS(°)"],name="상판 기울기 RMS",mode="lines+markers",line=dict(color=PLATE_COLOR,width=3),
                                 marker=dict(color=["#2EC4B6" if j=="통과" else "#E4572E" for j in dplot["판정"]],size=9)))
        fig.add_hline(y=crit_rms,line_dash="dash",line_color="#E4572E",annotation_text=f"허용 {crit_rms:g}°")
        _xr=[math.log10(0.03),math.log10(17.0)] if xcol=="Hs(m)" else None
        fig.update_layout(height=360,margin=dict(l=10,r=10,t=30,b=10),xaxis_title=xcol,yaxis_title="기울기 RMS(°)",
                          xaxis_type="log" if xcol=="Hs(m)" else "linear",legend=dict(orientation="h",y=1.12))
        if _xr: fig.update_xaxes(range=_xr)
        st.plotly_chart(fig,width="stretch",key="lim_fig")
        st.dataframe(dfv,hide_index=True,width="stretch")
        if not sw_used.startswith("Tp만"):
            hp=highest_pass(dfv)
            if hp is None:
                st.error("가상 모델 기준: WMO 1단계부터 기준을 만족하지 못합니다.")
            else:
                _r=wmo_row(hp); first_fail=dfv[(dfv["WMO"]>hp)&(dfv["판정"]=="실패")].head(1)
                msg=(f"**가상 모델 기준 결론: 현재 설정으로 WMO {hp} ({_r[3]}, Hs ≤ {_r[2]:g} m)까지 기준을 만족합니다.**")
                if len(first_fail): msg+=f" 처음 실패: Hs {first_fail['Hs(m)'].iloc[0]:g} m — {first_fail['실패 이유'].iloc[0]}"
                st.success(msg)
        st.caption(("선박 흔들림 크기: " + ({"fit":"Akademik 실측 회귀식(실측 Hs 범위 밖은 외삽)","phys":f"물리 기반 운동방정식(롤 {PHYS_CO['T44']:.1f} s·ζ {PHYS_CO['z44']:.2f}, 피치 {PHYS_CO['T55']:.1f} s·ζ {PHYS_CO['z55']:.2f}, 선속 {PHYS_P['U_kn']:g} kn, 파향 45/90/135°)"}
                                            .get(lm_used,"이전 앱의 가정 배율(근거 없음 — 결론은 참고용)")))
                   + f" · 주기: Pierson–Moskowitz 완전발달 해역 관계 · 처음 {LIMIT_WARMUP_S:.0f}초(켠 직후 수렴 구간)는 판정에서 뺌"
                   + " · 가상 파도는 실선 RAO가 아닌 파경사 기반 단순 모델입니다.")

    st.markdown("#### ② 실측 항해 데이터 기준 (Akademik INS, WMO 라벨별)")
    if not _pack:
        st.info("실제 항해 모드에서 자동 튜닝 화면의 관찰 구간을 한 번 준비하면(전체 항해 데이터로 자동 최적값 찾기 또는 관찰 구간 준비) 여기서 실측 기준 판정을 할 수 있습니다.")
    else:
        if st.button("실측 데이터 한계 분석 실행",key="run_limit_real"):
            with st.spinner("WMO 단계별 실측 관찰 구간 평가 중…"):
                st.session_state["lim_real"]=real_limit_by_wmo(_pack["items"],lim_params,dict(lim_fixed,accel_disturb=float(accel_disturb)))
        if st.session_state.get("lim_real") is not None and len(st.session_state["lim_real"]):
            dfr=judge(st.session_state["lim_real"],crit_rms,crit_max,crit_sat)
            st.dataframe(dfr,hide_index=True,width="stretch")
            st.caption(f"실측 관찰 구간(4분)에서 처음 {LIMIT_WARMUP_S:.0f}초는 판정에서 뺌 · 사이드바 '추가 가속도 오염 진폭' {float(accel_disturb):g}° 적용")
            hp=highest_pass(dfr); have=", ".join(str(int(c)) for c in dfr["WMO"])
            if hp is None:
                st.error(f"실측 기준: 데이터에 있는 가장 낮은 단계부터 기준을 만족하지 못합니다. (데이터에 있는 단계: {have})")
            else:
                _r=wmo_row(hp)
                st.success(f"**실측 기준 결론: 데이터에 있는 WMO {have} 중 WMO {hp} ({_r[3]})까지 기준을 만족합니다.** "
                           "데이터에 없는 단계는 판정할 수 없으므로 ① 가상 기준과 함께 보고하세요.")
            if _fit_live:
                _fit=_fit_live
                its=[it for it in _pack["items"] if it.get("wave_ok") and np.isfinite(it.get("Hs",np.nan)) and it["Hs"]>0.05]
                ff_=go.Figure()
                ff_.add_trace(go.Scatter(x=[it["Hs"] for it in its],y=[math.hypot(it["roll_rms"],it["pitch_rms"]) for it in its],mode="markers",name="실측 관찰 구간",marker=dict(color=SHIP_COLOR,size=6,opacity=0.6)))
                hx=np.geomspace(0.1,16,60); ff_.add_trace(go.Scatter(x=hx,y=_fit["c"]*hx**_fit["p"],name=f"회귀 {_fit['c']:.2f}·Hs^{_fit['p']:.2f} (R²={_fit['r2']:.2f})",line=dict(color="#123f70")))
                ff_.update_layout(height=300,margin=dict(l=10,r=10,t=30,b=10),xaxis_type="log",yaxis_type="log",xaxis_title="Hs(m)",yaxis_title="선박 기울기 RMS(°)",
                                  title=dict(text="실측: 파고와 선박 흔들림의 관계 (가상 모델 보정에 사용)",font=dict(size=13)),legend=dict(orientation="h",y=-0.25))
                st.plotly_chart(ff_,width="stretch",key="fit_fig")

# -----------------------------------------------------------------------------
# Auto tuner UI (available in actual mode only)
# -----------------------------------------------------------------------------
fixed={"tau":tau,"rate":float(rate),"limit":limit,"gyro_noise":gyro_noise,
       "gyro_bias":gyro_bias,"acc_noise":acc_noise,"accel_disturb":accel_disturb,**SERVO}
current={"ff":ff,"kp":kp,"kd":kd,"alpha":alpha}
HS_NOTE=("Hs/Tp는 Derkani et al.(2021)이 선박 Heave 5분 구간 스펙트럼과 RAO로 재구성한 추정값이며, "
         "같은 시각 부표가 직접 측정한 값이 아닙니다.")


def fmt_params(p):
    return f"FF {p['ff']:.3f} · Kp {p['kp']:.3f} · Kd {p['kd']:.4f} · α {p['alpha']:.4f}"


def set_kd_max(v: float) -> None:
    """Kd 탐색 상한을 바꾼다. 기본 0.12는 1차 서보 모델의 진동 경계 기준이며, 2차 모델에서는 더 큰 Kd가 유효할 수 있다."""
    global KD_MAX
    KD_MAX = float(v)
    TUNE_BOUNDS["kd"] = (0.0, KD_MAX)


def tuning_options(prefix: str, expanded=False):
    """자동 튜닝 공통 옵션(실험 케이스, 목적함수, 시드, Kd 로그 샘플링, 검증 노이즈 시드 수)."""
    with st.expander("자동 튜닝 옵션", expanded=expanded):
        o1,o2=st.columns(2)
        case=o1.selectbox("실험 케이스",list(CASES.keys()),index=0,key=f"{prefix}_case")
        objective=o2.selectbox("목적함수",OBJECTIVES,index=0,key=f"{prefix}_objective")
        o3,o4,o5=st.columns(3)
        seed=int(o3.number_input("탐색 시드",0,99999,42,1,key=f"{prefix}_seed"))
        kd_log=o4.toggle("Kd 로그 스케일 샘플링",value=True,key=f"{prefix}_kdlog")
        n_vs=int(o5.slider("검증 노이즈 시드 수",3,6,VALID_NOISE_SEEDS,1,key=f"{prefix}_vseeds"))
        kd_max=float(st.select_slider("Kd 탐색 상한",[0.06,0.09,0.12,0.2,0.3],value=0.12,key=f"{prefix}_kdmax"))
        set_kd_max(kd_max)
        st.caption(f"탐색 범위: FF {TUNE_BOUNDS['ff']}, Kp {TUNE_BOUNDS['kp']}, Kd {TUNE_BOUNDS['kd']}, α {TUNE_BOUNDS['alpha']}. "
                   f"Kd 상한 기본값 0.12는 서보 1차 모델의 안정 경계(기본 τ에서 약 0.07~0.08) 바로 위입니다. 서보 2차 모델에서는 ⑥ D 게인 비교와 ④ 격자로 경계를 확인한 뒤 상한을 올리세요. "
                   "제약 조건 방식은 자세 RMS비를 최소화하고 포화율 >1% 또는 리밋사이클이면 큰 페널티를 줍니다. "
                   "가중합 방식은 0.45·RMS비 + 0.20·95% 잔차비 + 0.15·최대각비 + 0.05·(각속도 RMS + 각가속도 RMS + 2Hz 이상 비율 + 포화율)입니다.")
    return {"case":case,"case_fixed":CASES[case],"objective":objective,"seed":seed,"kd_log":kd_log,"n_valid_seeds":n_vs}


def heatmap(z, x, y, title, zlabel, zmin=None, zmax=None, text_fmt=".2f", colorscale="Viridis_r"):
    fig=go.Figure(go.Heatmap(z=z,x=x,y=y,colorscale=colorscale,zmin=zmin,zmax=zmax,colorbar=dict(title=zlabel),
                             text=np.vectorize(lambda v: format(v,text_fmt))(z),texttemplate="%{text}",hoverongaps=False))
    fig.update_layout(title=title,height=max(320,60+36*len(y)),margin=dict(l=10,r=10,t=40,b=10))
    return fig


if actual_mode:
    st.divider()
    st.header("시뮬레이션 기반 자동 튜닝(무작위 탐색 + 국소 탐색)")
    st.write("현재 준비된 전체 INS 일별 파일에서 여러 날짜·연속 구간의 대표 4분 관찰 구간을 자동 추출합니다. FF, Kp, Kd, α 후보들을 반복 시뮬레이션하고 별도 날짜의 검증 데이터로 결과를 확인합니다.")
    st.warning("자동 튜닝 평가는 실시간 재생과 같은 50Hz 제어 루프, 같은 선형 보간, 같은 센서 노이즈 생성 순서를 사용합니다. "
               "입력은 1Hz 원자료를 보간한 값이므로 기록되지 않은 고주파 운동은 포함되지 않습니다. 결과는 시뮬레이터 추천값이며 실제 장치의 검증 완료값은 아닙니다.")
    st.caption(f"계산량 때문에 탐색 단계는 학습용 관찰 구간 중 최대 {SEARCH_WINDOWS}개(탐색 시드로 정해지는 무작위 추출)만 쓰고, 최종 검증만 검증용 관찰 구간 전체 × 노이즈 시드 여러 개를 씁니다.")
    c1,c2,c3=st.columns(3)
    c1.metric("튜닝 가능 일별 파일",f"{len(set(x['member'] for x in manifest)):,}")
    c2.metric("연속 구간",f"{len(manifest):,}")
    c3.metric("조정 파라미터","FF · Kp · Kd · α")
    trial_count=st.slider("자동 탐색 후보 수",24,80,40,4,key="tune_trials")
    opts=tuning_options("main")

    def get_pack():
        key=f"{archive_path.signature()}|{len(manifest)}|{len(wave_df)}"
        if st.session_state.get("tune_pack_key")!=key:
            with st.status("전체 항해 기간에서 관찰 구간 준비 중…",expanded=True) as status:
                status.write("일별 파일을 전체 기간에서 균등하게 골라 대표 4분 관찰 구간을 추출하고 파랑 라벨을 붙입니다.")
                pack=make_tuning_windows(archive_path,manifest,wave_df)
                status.update(label=f"관찰 구간 {pack['n_windows']}개 준비 완료",state="complete",expanded=False)
            st.session_state["tune_pack"]=pack; st.session_state["tune_pack_key"]=key
        return st.session_state["tune_pack"]

    def split_items(pack):
        tr=[it for it in pack["items"] if not it["validation"]]
        va=[it for it in pack["items"] if it["validation"]]
        return tr,va

    def run_opt(pack, o, seed, progress=None):
        tr,va=split_items(pack)
        return optimize_params(tr,va,current,fixed,trial_count,progress,seed=seed,objective=o["objective"],
                               case_fixed=o["case_fixed"],kd_log=o["kd_log"],n_valid_seeds=o["n_valid_seeds"])

    if st.button("전체 항해 데이터로 자동 최적값 찾기",type="primary",width="stretch"):
        try:
            pack=get_pack()
            tr,va=split_items(pack)
            st.write(f"준비 완료: 일별 파일 {pack['days_used']}/{pack['days_total']}일 사용 · 4분 관찰 구간 {pack['n_windows']}개 · 학습 {len(tr)}개 · 검증 {len(va)}개")
            st.write(f"탐색 단계: 학습용 관찰 구간 {min(SEARCH_WINDOWS,len(tr))}/{len(tr)}개 · 최종 검증: 검증용 관찰 구간 {len(va)}/{len(va)}개 × 노이즈 시드 {opts['n_valid_seeds']}개 · 50Hz 평가")
            bar=st.progress(0.0)
            def progress(frac,msg): bar.progress(min(frac,1.0),text=msg)
            result=run_opt(pack,opts,opts["seed"],progress)
            result["n_days"]=pack["days_used"]; result["n_windows"]=pack["n_windows"]; result["case"]=opts["case"]
            st.session_state["optimization_result"]=result
            bar.progress(1.0,text="자동 튜닝 완료")
        except Exception as exc:
            st.error(f"자동 튜닝 중 오류: {type(exc).__name__}: {exc}")
    result=st.session_state.get("optimization_result")
    pack=st.session_state.get("tune_pack")
    if result:
        st.subheader("자동 튜닝 결과")
        best=result["best"]
        u=result["usage"]
        st.info(f"평가 관찰 구간 사용: 탐색 단계 학습용 관찰 구간 {u['search_used']}/{u['train_total']}개 (후보 {u['search_evals']}개 × 50Hz {u['search_steps']:,}스텝) · "
                f"안정성 검사 학습용 관찰 구간 {u['stab_windows']}개 · 최종 검증 검증용 관찰 구간 {u['valid_used']}/{u['valid_total']}개 × 노이즈 시드 {u['valid_seeds']}개 · "
                f"일별 파일 {result.get('n_days','-')}/{pack['days_total'] if pack else '-'}일 사용")
        st.caption(f"실험 케이스: {result.get('case','-')} · 목적함수: {result['objective']} · 탐색 시드 {result['seed']} · Kd 로그 샘플링 {'ON' if result['kd_log'] else 'OFF'}")
        n_chk=u.get("stab_checked",STAB_TOP_K)
        if result["stable"] is False:
            st.error(f"점수 상위 {n_chk}개 후보를 모두 검사했지만 안정성 검사(게인 ×1.5/×2, 서보 속도 변화, 리밋사이클)를 통과한 후보가 없습니다. "
                     "아래 값은 점수 1위이며 안정성이 보장되지 않습니다. 후보 수를 늘리거나 Kd 탐색 상한을 낮춰 다시 실행해 보세요.")
        elif result["stable"]:
            rank_best=int(result["all"].sort_values("학습 점수").reset_index(drop=True).index[result["all"].sort_values("학습 점수")["추천"].to_numpy()][0])+1
            st.success(f"추천 파라미터는 점수 상위 {n_chk}개를 순서대로 검사해 안정성 검사를 통과한 후보 중 점수가 가장 좋은 값입니다(학습 점수 순위 {rank_best}위). "
                       "아래 값은 시뮬레이터의 현재 센서·서보 가정값에 종속됩니다.")
        mc=st.columns(4)
        mc[0].metric("추천 FF",f"{best['ff']:.3f}")
        mc[1].metric("추천 Kp",f"{best['kp']:.3f}")
        mc[2].metric("추천 Kd",f"{best['kd']:.4f}")
        mc[3].metric("추천 α",f"{best['alpha']:.4f}")
        comp=pd.DataFrame([
            {"검증 구분":name,"검증 점수":m["score"],"자세 RMS비":m["rms_ratio"],
             "Roll RMS":"%.3f°"%m["roll_rms"],"Pitch RMS":"%.3f°"%m["pitch_rms"],
             "Roll 최대각":"%.3f°"%m["roll_max"],"Pitch 최대각":"%.3f°"%m["pitch_max"],
             "서보 포화율":"%.2f%%"%(100*m["saturation"]),"리밋사이클 관찰 구간 비율":"%.2f"%m["lc_frac"]}
            for name,m in (("현재 파라미터",result["current_valid"]),("추천 파라미터",result["valid"]))])
        st.markdown(f"**학습에 사용하지 않은 검증 구간 결과** (노이즈 시드 {u['valid_seeds']}개 평균)")
        st.dataframe(comp,hide_index=True,width="stretch")
        cur=result["current_valid"]; opt=result["valid"]
        if cur["score"]>0:
            st.metric("검증 점수 개선율",f"{100*(cur['score']-opt['score'])/cur['score']:.1f}%")
        st.markdown("**상위 후보 10개 (탐색 단계 학습용 관찰 구간 기준)**")
        st.dataframe(result["top"].round(4),hide_index=True,width="stretch")
        st.button("추천 파라미터를 제어 슬라이더에 적용",on_click=lambda: st.session_state.update({"apply_best_on_next_run":True}),
                  type="primary",width="stretch",key="apply_best_button")
        st.download_button("최적화 후보 및 결과 CSV 저장",result["top"].to_csv(index=False).encode("utf-8-sig"),
                           file_name="akademik_auto_tuning_candidates.csv",mime="text/csv")
        st.caption("서보 τ·최대 각속도·회전한계와 센서 노이즈/바이어스 설정은 최적화 중 고정했습니다. 이 고정값이 실제 하드웨어 사양과 측정치가 아니면 추천값의 신뢰도가 낮아집니다.")

    st.subheader("튜닝 분석")
    t1,t2,t3,t4,t6,t7=st.tabs(["① 수렴·재현성","② 전체 후보·파레토","③ 안정성 검사","④ Kp×Kd 격자·민감도","⑤ 관찰 구간 목록·분포","⑥ D 게인 비교"])

    # ---------------------------------------------------------------- ① convergence & reproducibility
    with t1:
        if result:
            h=result["history"]
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=h["평가 횟수"],y=h["점수"],mode="markers",name="각 후보 점수",marker=dict(color="#9AA5B1",size=6)))
            fig.add_trace(go.Scatter(x=h["평가 횟수"],y=h["최고 점수"],mode="lines",name="지금까지 최고 점수",line=dict(color=PLATE_COLOR,width=3,shape="hv")))
            fig.update_layout(title="평가 횟수 vs 최고 점수 (수렴 곡선)",height=330,yaxis_type="log",xaxis_title="평가 횟수",yaxis_title="학습 점수(로그)",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="conv_fig")
        else:
            st.info("자동 튜닝을 먼저 실행하면 수렴 곡선이 나타납니다.")
        st.markdown("**시드 반복 실험**: 같은 옵션으로 탐색 시드만 바꿔 최적화를 반복합니다(탐색용 학습용 관찰 구간 추출도 시드에 따라 달라집니다).")
        n_rep=st.slider("반복할 시드 수",2,10,5,1,key="seedrep_n")
        if st.button("시드 반복 실험",key="run_seedrep"):
            try:
                pk=get_pack(); runs=[]
                bar=st.progress(0.0)
                for r in range(n_rep):
                    res_=run_opt(pk,opts,opts["seed"]+r,lambda f,m,r=r: bar.progress(min((r+f)/n_rep,1.0),text=f"시드 {opts['seed']+r} · {m}"))
                    runs.append(res_)
                rows=[{"탐색 시드":r_["seed"],"FF":r_["best"]["ff"],"Kp":r_["best"]["kp"],"Kd":r_["best"]["kd"],"α":r_["best"]["alpha"],
                       "학습 점수":r_["train"]["score"],"검증 점수":r_["valid"]["score"],"검증 RMS비":r_["valid"]["rms_ratio"],
                       "안정성 검사":"통과" if r_["stable"] else "실패"} for r_ in runs]
                per=pd.DataFrame(rows)
                stats=pd.DataFrame([{"항목":k,"평균":per[k].mean(),"표준편차":per[k].std(ddof=0),"최솟값":per[k].min(),"최댓값":per[k].max()}
                                    for k in ("FF","Kp","Kd","α","검증 점수")])
                st.session_state["seedrep"]={"per":per,"stats":stats,"hist":[(r_["seed"],r_["history"]) for r_ in runs],"case":opts["case"],"objective":opts["objective"]}
            except Exception as exc:
                st.error(f"시드 반복 실험 오류: {type(exc).__name__}: {exc}")
        sr=st.session_state.get("seedrep")
        if sr:
            st.caption(f"실험 케이스: {sr['case']} · 목적함수: {sr['objective']}")
            st.dataframe(sr["stats"].round(4),hide_index=True,width="stretch")
            st.dataframe(sr["per"].round(4),hide_index=True,width="stretch")
            fig=go.Figure()
            for sd,h in sr["hist"]:
                fig.add_trace(go.Scatter(x=h["평가 횟수"],y=h["최고 점수"],mode="lines",name=f"시드 {sd}",line=dict(shape="hv")))
            fig.update_layout(title="시드별 수렴 곡선",height=320,yaxis_type="log",xaxis_title="평가 횟수",yaxis_title="최고 점수(로그)",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="seedrep_fig")
            st.download_button("시드 반복 결과 CSV 저장",sr["per"].to_csv(index=False).encode("utf-8-sig"),file_name="seed_repeat.csv",mime="text/csv",key="dl_seedrep")

    # ---------------------------------------------------------------- ② all candidates & Pareto
    with t2:
        if result:
            alldf=result["all"]
            x=alldf["서보 각속도 RMS(정규화)"].to_numpy(float); y=alldf["자세 RMS비"].to_numpy(float)
            pm=pareto_mask(x,y)
            fig=go.Figure()
            lc_mask=alldf["리밋사이클 관찰 구간 비율"].to_numpy(float)>0
            fig.add_trace(go.Scatter(x=x[~lc_mask],y=y[~lc_mask],mode="markers",name="후보",marker=dict(color="#9AA5B1",size=7),
                                     text=[f"FF {a:.2f}, Kp {b:.2f}, Kd {c:.3f}, α {d:.3f}" for a,b,c,d in alldf.loc[~lc_mask,["FF","Kp","Kd","α"]].to_numpy()]))
            fig.add_trace(go.Scatter(x=x[lc_mask],y=y[lc_mask],mode="markers",name="리밋사이클 검출 후보",marker=dict(color=PLATE_OFF_COLOR,size=7,symbol="x")))
            order=np.argsort(x[pm])
            fig.add_trace(go.Scatter(x=x[pm][order],y=y[pm][order],mode="lines+markers",name="파레토 해(비지배해)",line=dict(color=PLATE_COLOR,width=2.5),marker=dict(size=9)))
            rec=alldf["추천"].to_numpy(bool)
            fig.add_trace(go.Scatter(x=x[rec],y=y[rec],mode="markers",name="추천값",marker=dict(color="#F2C14E",size=16,symbol="star",line=dict(color="black",width=1))))
            fig.update_layout(title="서보 각속도 RMS(정규화) vs 자세 RMS비",height=420,xaxis_title="서보 각속도 RMS / 최대 각속도",yaxis_title="자세 RMS비(보정 후/보정 전)",
                              yaxis_type="log",xaxis_type="log",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="pareto_fig")
            st.caption(f"평가한 후보 {len(alldf)}개 중 파레토 해 {int(pm.sum())}개. 왼쪽 아래로 갈수록 적은 서보 움직임으로 수평을 잘 유지합니다.")
            st.dataframe(alldf.round(4),hide_index=True,width="stretch")
            st.download_button("전체 후보 CSV 저장",alldf.to_csv(index=False).encode("utf-8-sig"),file_name="all_candidates.csv",mime="text/csv",key="dl_all")
        else:
            st.info("자동 튜닝을 먼저 실행하면 평가한 모든 후보가 나타납니다.")

    # ---------------------------------------------------------------- ③ stability check
    with t3:
        st.markdown(f"**판정 기준**: 게인 (Kp, Kd) ×{STAB_GAIN_FACTORS[0]:g}·×{STAB_GAIN_FACTORS[1]:g}, 서보 τ ×{STAB_TAU_FACTORS[0]:g}·×{STAB_TAU_FACTORS[1]:g} 각각에서 "
                    f"자세 RMS비(검사 관찰 구간 전체 합산) < 1.0 이고 포화율 < {100*STAB_SAT_MAX:.0f}%이며, 모든 조건에서 리밋사이클이 없어야 함 "
                    f"(관절 각속도 부호변화 진동수 > {LC_FLIP_HZ:g} Hz, 또는 2Hz 이상 성분 비율 > {LC_HF_RATIO:g} 이면서 RMS > {LC_HF_RMS:g}°이면 리밋사이클).")
        if result and result.get("stability"):
            st.markdown(f"**추천 파라미터** ({fmt_params(result['best'])})")
            st.dataframe(result["stability"]["table"].round(4),hide_index=True,width="stretch")
        if st.button("현재 슬라이더 값 안정성 검사",key="run_stab_current"):
            try:
                pk=get_pack(); tr,_=split_items(pk)
                st.session_state["stab_current"]={"params":dict(current),"res":stability_check(current,(result.get("stab_items") if result and result.get("stab_items") else tr[:STAB_WINDOWS]),fixed)}
            except Exception as exc:
                st.error(f"안정성 검사 오류: {type(exc).__name__}: {exc}")
        sc_=st.session_state.get("stab_current")
        if sc_:
            st.markdown(f"**현재 슬라이더 값** ({fmt_params(sc_['params'])}) → {'통과' if sc_['res']['passed'] else '실패'}")
            st.dataframe(sc_["res"]["table"].round(4),hide_index=True,width="stretch")

    # ---------------------------------------------------------------- ④ grid & sensitivity
    with t4:
        st.write("FF와 α는 사이드바 슬라이더 값으로 고정하고 Kp×Kd 격자를 평가합니다. 색은 자세 RMS비(검사 관찰 구간 전체 합산), ✕는 안정성 검사 실패 칸입니다.")
        g1,g2,g3=st.columns(3)
        gn=g1.select_slider("격자 크기",[8,10,12],value=12,key="grid_n")
        kd_axis=g2.radio("Kd 축",["선형","로그"],horizontal=True,key="grid_kdaxis")
        kd_view=g3.select_slider("Kd 최대",[0.06,0.09,0.12,0.2,0.3],value=0.12,key="grid_kdmax")
        if st.button("Kp×Kd 격자 탐색 실행",key="run_grid"):
            try:
                pk=get_pack(); tr,_=split_items(pk)
                sub=tr[:6]
                kps=np.linspace(0.0,4.0,gn)
                kds=np.linspace(0.0,kd_view,gn) if kd_axis=="선형" else np.r_[0.0,np.geomspace(KD_LOG_MIN,kd_view,gn-1)]
                bar=st.progress(0.0)
                gr=grid_scan(ff,alpha,kps,kds,sub,fixed,opts["objective"],lambda f: bar.progress(min(f,1.0),text="격자 평가 중"))
                if result:
                    inp_,sd_,ow_=prepare_inputs(sub)
                    gr["random_best"]=dict(result["best"])
                    gr["random_score"]=evaluate(inp_,sd_,[result["best"]],fixed,opts["objective"],ow_)["summary"][0]["score"]
                gr["objective"]=opts["objective"]
                st.session_state["grid_result"]=gr
            except Exception as exc:
                st.error(f"격자 탐색 오류: {type(exc).__name__}: {exc}")
        gr=st.session_state.get("grid_result")
        if gr:
            ratio=np.minimum(gr["ratio"],1.5)
            xlab=[f"{v:.4f}" for v in gr["kds"]]; ylab=[f"{v:.2f}" for v in gr["kps"]]
            fig=go.Figure(go.Heatmap(z=ratio,x=xlab,y=ylab,colorscale="Viridis_r",zmin=0,zmax=1.5,colorbar=dict(title="RMS비")))
            fi,fj=np.where(~gr["passed"])
            fig.add_trace(go.Scatter(x=[xlab[j] for j in fj],y=[ylab[i] for i in fi],mode="markers",name="안정성 검사 실패",
                                     marker=dict(symbol="x",color="white",size=9,line=dict(color="black",width=1))))
            sc=np.where(gr["passed"],gr["score"],np.inf)
            if not np.isfinite(sc).any(): sc=gr["score"]
            bi,bj=np.unravel_index(int(np.argmin(sc)),sc.shape)
            fig.add_trace(go.Scatter(x=[xlab[bj]],y=[ylab[bi]],mode="markers",name="격자 최적값(안정 통과 중)",marker=dict(symbol="diamond",color=PLATE_COLOR,size=14,line=dict(color="black",width=1))))
            txt=f"격자 최적값: Kp {gr['kps'][bi]:.2f}, Kd {gr['kds'][bj]:.4f}, 점수 {gr['score'][bi,bj]:.4f}"
            if "random_best" in gr:
                rb=gr["random_best"]
                jx=int(np.argmin(np.abs(gr["kds"]-rb["kd"]))); iy=int(np.argmin(np.abs(gr["kps"]-rb["kp"])))
                fig.add_trace(go.Scatter(x=[xlab[jx]],y=[ylab[iy]],mode="markers",name="무작위 탐색 최적값(가장 가까운 칸)",marker=dict(symbol="star",color="#F2C14E",size=16,line=dict(color="black",width=1))))
                diff=100*(gr["random_score"]-gr["score"][bi,bj])/max(abs(gr["score"][bi,bj]),1e-12)
                txt+=f" · 무작위 탐색 최적값({fmt_params(rb)}) 점수 {gr['random_score']:.4f} → 격자 최적 대비 {diff:+.1f}%"
                txt+=" (무작위 탐색값은 FF·α도 튜닝했을 수 있어 음수가 나올 수 있습니다)"
            fig.update_layout(title=f"Kp×Kd 안정 지도 (FF {gr['ff']:.2f}, α {gr['alpha']:.3f} 고정, 관찰 구간 {gr['n_windows']}개)",height=480,
                              xaxis_title="Kd (s)",yaxis_title="Kp",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="grid_fig")
            st.caption(txt+f" · 목적함수: {gr['objective']}")
            grid_df=pd.DataFrame(gr["ratio"],index=[f"Kp={v}" for v in ylab],columns=[f"Kd={v}" for v in xlab])
            st.download_button("격자 결과 CSV 저장",grid_df.to_csv().encode("utf-8-sig"),file_name="kp_kd_grid.csv",mime="text/csv",key="dl_grid")
        st.markdown("**1D 민감도**: 나머지 파라미터를 자동 튜닝 최적값(없으면 현재 슬라이더 값)에 고정하고 한 파라미터만 바꿉니다. Kd는 슬라이더 최대 0.30까지 봅니다.")
        if st.button("1D 민감도 곡선 계산",key="run_sens"):
            try:
                pk=get_pack(); tr,_=split_items(pk)
                base=dict(result["best"]) if result else dict(current)
                bar=st.progress(0.0)
                st.session_state["sens_result"]={"base":base,"df":sensitivity_curves(base,tr[:6],fixed,opts["objective"],progress_cb=lambda f: bar.progress(min(f,1.0),text="민감도 계산 중"))}
            except Exception as exc:
                st.error(f"민감도 계산 오류: {type(exc).__name__}: {exc}")
        sres=st.session_state.get("sens_result")
        if sres:
            sd=sres["df"]
            fig=make_subplots(rows=1,cols=4,subplot_titles=["FF","Kp","Kd","α"],shared_yaxes=True)
            for i,k in enumerate(["ff","kp","kd","alpha"],1):
                part=sd[sd["파라미터"]==k]
                fig.add_trace(go.Scatter(x=part["값"],y=part["자세 RMS비"],mode="lines+markers",name="자세 RMS비",showlegend=i==1,line=dict(color=PLATE_COLOR)),row=1,col=i)
                lcp=part[part["리밋사이클 관찰 구간 비율"]>0]
                fig.add_trace(go.Scatter(x=lcp["값"],y=lcp["자세 RMS비"],mode="markers",name="리밋사이클",showlegend=i==1,marker=dict(color=PLATE_OFF_COLOR,symbol="x",size=9)),row=1,col=i)
                fig.add_vline(x=sres["base"][k],line_dash="dot",line_color="gray",row=1,col=i)
            fig.update_yaxes(type="log",title_text="자세 RMS비",row=1,col=1)
            fig.update_layout(height=340,margin=dict(l=10,r=10,t=40,b=10),title=f"1D 민감도 (기준값 {fmt_params(sres['base'])})")
            st.plotly_chart(fig,width="stretch",key="sens_fig")
            st.download_button("민감도 CSV 저장",sd.to_csv(index=False).encode("utf-8-sig"),file_name="sensitivity.csv",mime="text/csv",key="dl_sens")

    # ---------------------------------------------------------------- ⑥ windows & distribution
    with t6:
        if st.button("관찰 구간 준비",key="prep_pack"):
            try: get_pack()
            except Exception as exc: st.error(f"관찰 구간 준비 오류: {type(exc).__name__}: {exc}")
        pack=st.session_state.get("tune_pack")
        if pack:
            wt=window_table(pack["items"])
            st.markdown(f"**관찰 구간 목록**: 일별 파일 {pack['days_used']}/{pack['days_total']}일 사용 (균등 추출 {pack['days_chosen']}일) · 관찰 구간 {pack['n_windows']}개 · 파랑 자료 없음 {int(wt['WMO 코드'].isna().sum())}개")
            if pack["days_used"]<pack["days_total"]:
                st.caption("관찰 구간 수 상한(180개) 때문에 일부 날짜만 쓰지만, 날짜를 전체 기간에서 균등 간격으로 골라 특정 기간이 빠지지 않습니다.")
            st.dataframe(wt.round(3),hide_index=True,width="stretch")
            st.download_button("관찰 구간 목록 CSV",wt.to_csv(index=False).encode("utf-8-sig"),file_name="tuning_windows.csv",mime="text/csv",key="dl_windows")
            cats=[str(c) for c,_,_,_ in WMO_CODES]+["자료 없음"]
            wt["WMO 라벨"]=wt["WMO 코드"].apply(lambda v: "자료 없음" if pd.isna(v) else str(int(v)))
            fig=go.Figure()
            for part,color in (("학습",SHIP_COLOR),("검증",PLATE_COLOR)):
                cnt=wt[wt["구분"]==part]["WMO 라벨"].value_counts()
                fig.add_trace(go.Bar(x=cats,y=[int(cnt.get(c,0)) for c in cats],name=part,marker_color=color))
            fig.update_layout(barmode="group",title="WMO 코드별 관찰 구간 개수",height=320,xaxis_title="WMO 해상상태 코드",yaxis_title="관찰 구간 개수",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="wmo_bar")
            ok=wt.dropna(subset=["Hs(m)"])
            fig=go.Figure(go.Scatter(x=ok["Tp(s)"],y=ok["Hs(m)"],mode="markers",marker=dict(color=ok["WMO 코드"],colorscale="Turbo",cmin=0,cmax=9,size=8,
                                     colorbar=dict(title="WMO")),text=ok["날짜"]+" "+ok["시작 시각(UTC)"]))
            fig.update_layout(title="Hs–Tp 분포 (색 = WMO 코드)",height=360,xaxis_title="Tp (s)",yaxis_title="Hs (m)",margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="hs_tp_fig")
            st.caption(HS_NOTE)
        else:
            st.info("자동 튜닝을 실행하거나 '관찰 구간 준비'를 누르면 관찰 구간 목록이 나타납니다.")

    # ---------------------------------------------------------------- ⑦ D gain only
    with t7:
        st.write("FF·Kp·α를 고정하고 **Kd만 바꿔** 비교합니다. ① 계단 응답(선박이 갑자기 기울었을 때 상판이 수평을 지나쳐 되돌아오는 폭), "
                 "② 실제 항해 검증용 관찰 구간 성능, ③ 안정성 검사를 같은 조건에서 계산합니다.")
        model_txt=(f"서보 2차 모델 ON (fn {SERVO['wn']/(2*math.pi):.1f} Hz, ζ {SERVO['zeta']:.2f})" if SERVO["servo2"]
                   else f"서보 1차 모델 (τ {tau:.2f}s)")
        if SERVO["servo2"]:
            st.success(f"현재 {model_txt}. 상판·적재물 관성 때문에 넘어감이 생기므로 D의 감쇠 효과를 볼 수 있습니다.")
        else:
            st.warning(f"현재 {model_txt}. 1차 모델에는 관성이 없어 계단 응답에서 넘어감이 생기지 않으므로 D의 감쇠 효과가 드러나지 않습니다. "
                       "사이드바 '서보 및 센서 고정 조건'에서 **서보 2차 모델**을 켜고 실험하는 것을 권장합니다.")
        base_opts=(["자동 튜닝 추천값"] if result else [])+["현재 슬라이더 값"]
        k1,k2=st.columns(2)
        base_src=k1.radio("고정할 FF·Kp·α",base_opts,horizontal=True,key="kd_base")
        default_kd="0, 0.0079, 0.02, 0.04, 0.06, 0.1" if not result else f"0, {result['best']['kd']:.4f}, 0.02, 0.04, 0.06, 0.1"
        kd_txt=k2.text_input("비교할 Kd 목록(쉼표 구분)",default_kd,key="kd_list")
        k3,k4,k5=st.columns(3)
        step_deg=k3.slider("계단 크기(선박 Roll, °)",0.5,5.0,2.0,0.5,key="kd_step")
        step_noise=k4.toggle("계단 응답에 센서 노이즈 포함",value=False,key="kd_noise")
        kd_valid=k5.toggle("실제 항해 검증용 관찰 구간 평가도 실행",value=True,key="kd_valid")
        if st.button("D 게인 비교 실행",key="run_kd"):
            try:
                kds=sorted({max(0.0,float(x)) for x in re.split(r"[,\s]+",kd_txt.strip()) if x})
                if not kds: raise ValueError("Kd 값을 하나 이상 입력해 주세요.")
                base=dict(result["best"]) if (base_src=="자동 튜닝 추천값" and result) else dict(current)
                bar=st.progress(0.0,text="계단 응답 계산 중")
                curves,rows=[],[]
                for i,kv in enumerate(kds):
                    df_,m_=kd_step_response(base,kv,fixed,step_deg,step_noise)
                    curves.append((kv,df_)); rows.append(m_)
                    bar.progress((i+1)/len(kds)*0.3,text=f"계단 응답 Kd={kv:g}")
                step_tab=pd.DataFrame(rows)
                val_tab=None
                if kd_valid:
                    pk=get_pack(); tr,va=split_items(pk)
                    cands=[dict(base,kd=kv) for kv in kds]
                    v_in,v_sd,v_ow=prepare_inputs(va,VALID_NOISE_SEEDS)
                    ev=evaluate(v_in,v_sd,cands,fixed,OBJ_WEIGHTED,v_ow,progress_cb=lambda f: bar.progress(0.3+0.4*f,text="검증용 관찰 구간 평가 중"))["summary"]
                    _ro=st.session_state.get("optimization_result")
                    k_in,k_sd,k_ow=prepare_inputs(_ro.get("stab_items") if _ro and _ro.get("stab_items") else tr[:STAB_WINDOWS])
                    chk=stability_check_many(cands,k_in,k_sd,k_ow,fixed,lambda f: bar.progress(0.7+0.3*f,text="안정성 검사 중"))
                    val_tab=pd.DataFrame([{"Kd":kv,"자세 RMS비":e["rms_ratio"],"Roll RMS(°)":e["roll_rms"],"Pitch RMS(°)":e["pitch_rms"],
                                           "Roll 최대각(°)":e["roll_max"],"Pitch 최대각(°)":e["pitch_max"],
                                           "Roll 95%(°)":e["roll_p95"],"Pitch 95%(°)":e["pitch_p95"],
                                           "서보 각속도 RMS(정규화)":e["rate_rms"],"서보 각가속도 RMS(정규화)":e["acc_rms"],
                                           "2Hz 이상 비율":e["hf"],"리밋사이클 관찰 구간 비율":e["lc_frac"],"포화율(%)":100*e["saturation"],
                                           "안정성 검사":"통과" if c["passed"] else "실패",
                                           "실패 조건":", ".join(c["table"].loc[c["table"]["판정"]=="실패","조건"].tolist())}
                                          for kv,e,c in zip(kds,ev,chk)])
                    val_tab.attrs["n"]=(len(va),VALID_NOISE_SEEDS)
                bar.progress(1.0,text="완료")
                st.session_state["kd_result"]={"base":base,"model":model_txt,"step":step_tab,"curves":curves,"val":val_tab,
                                               "step_deg":step_deg,"noise":step_noise,"accel":accel_disturb}
            except Exception as exc:
                st.error(f"D 게인 비교 오류: {type(exc).__name__}: {exc}")
        kr=st.session_state.get("kd_result")
        if kr:
            st.caption(f"고정값: {fmt_params(dict(kr['base'],kd=0.0)).replace(' · Kd 0.0000','')} · {kr['model']} · "
                       f"계단 {kr['step_deg']:g}° · 센서 노이즈 {'포함' if kr['noise'] else '제외'} · 가속도 오염 {kr['accel']:g}°")
            fig=go.Figure()
            pal=["#1f77b4","#2EC4B6","#F2C14E","#F28C38","#E4572E","#7B2CBF","#6c757d","#2b9348"]
            first=kr["curves"][0][1]
            fig.add_trace(go.Scatter(x=first["time"],y=first["ship_roll"],name="선박 Roll(입력)",line=dict(color="#9AA5B1",dash="dot")))
            for i,(kv,df_) in enumerate(kr["curves"]):
                fig.add_trace(go.Scatter(x=df_["time"],y=df_["plate_roll"],name=f"상판 Roll · Kd {kv:g}",line=dict(color=pal[i%len(pal)],width=2)))
            fig.add_hline(y=0,line_color="black",line_width=1)
            fig.update_layout(title="계단 응답: 상판 Roll (0 = 수평)",height=380,xaxis_title="시간(s)",yaxis_title="각도(°)",
                              yaxis_range=[-0.6*kr["step_deg"],1.1*kr["step_deg"]],margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig,width="stretch",key="kd_step_fig")
            st.markdown("**계단 응답 지표**")
            st.dataframe(kr["step"].round(4),hide_index=True,width="stretch")
            stp=kr["step"]
            lo,hi=stp.iloc[0],stp.iloc[-1]
            best_i=int(stp["수평을 지나친 폭(°)"].idxmin())
            st.info(f"Kd {lo['Kd']:g} → {hi['Kd']:g}: 수평을 지나친 폭 {lo['수평을 지나친 폭(°)']:.3f}° → {hi['수평을 지나친 폭(°)']:.3f}°, "
                    f"흔들림 횟수 {int(lo['흔들림 횟수(부호 변화)'])} → {int(hi['흔들림 횟수(부호 변화)'])}, "
                    f"정착 시간 {lo['5% 정착 시간(s)']:.2f}s → {hi['5% 정착 시간(s)']:.2f}s. "
                    f"넘어감이 가장 작은 값은 Kd {stp.loc[best_i,'Kd']:g}입니다.")
            if kr["val"] is not None:
                n_va,n_sd=kr["val"].attrs.get("n",("-","-"))
                st.markdown(f"**실제 항해 검증용 관찰 구간 성능과 안정성 검사** (검증용 관찰 구간 {n_va}개 × 노이즈 시드 {n_sd}개)")
                st.dataframe(kr["val"].round(4),hide_index=True,width="stretch")
                st.caption("넘어감을 줄이는 Kd라도 안정성 검사(게인 ×1.5/×2, 서보 속도 변화)에서 실패하면 실제 장치에 쓰기 어렵습니다. "
                           "계단 응답의 넘어감과 검증용 관찰 구간 RMS·최대각, 안정성 판정을 함께 보고 고르세요.")
                st.download_button("D 게인 비교 CSV",kr["val"].merge(kr["step"],on="Kd").to_csv(index=False).encode("utf-8-sig"),
                                   file_name="kd_comparison.csv",mime="text/csv",key="dl_kd")
            else:
                st.download_button("계단 응답 지표 CSV",kr["step"].to_csv(index=False).encode("utf-8-sig"),file_name="kd_step.csv",mime="text/csv",key="dl_kd_step")

with st.expander("모델·데이터 출처와 해석상의 한계"):
    st.markdown("""
**계산 흐름:** 선박 Roll/Pitch → 가상 IMU → 상보필터 → FF+PD → 서보 모델 → 상판 자세 (계산식은 verified_sim.py, 출처는 '📐 계산식 출처와 자체 검증' 참고)

**자동 데이터 관리**
- 저장소의 `data/` 폴더에 경량 데이터(롤·피치·시각만 추출, 18MB 이하 조각 여러 개)가 있으면 그것을 바로 읽고 **다운로드하지 않습니다.** 코드를 고쳐 다시 배포해도 그대로 유지됩니다.
- `data/`가 없을 때만 Zenodo에서 INS 원본 ZIP(약 646.9MB)과 Hs/Tp CSV를 실행 환경 캐시에 받습니다. 한 번 확인한 파일은 체크섬을 다시 계산하지 않습니다.
- 경량 데이터는 원본과 같은 방법으로 연속 구간을 나누며, 원본과의 차이는 각도 float32 반올림(약 10⁻⁷°) 정도입니다. 선수방향·속도 열은 표시용이라 경량 데이터에서 뺐습니다.
- 날짜순으로 연속 구간을 인덱싱하며, 구간 사이 긴 시간 공백을 가로질러 보간하지 않습니다.
- 현재 구간이 끝나면 자동으로 다음 날짜/구간으로 넘어갑니다.

**시뮬레이션 기반 자동 튜닝(무작위 탐색 + 국소 탐색)**
- FF, Kp, Kd, α만 조정합니다. 실험 케이스로 FF=1 고정 또는 FF=0(PD만)도 선택할 수 있습니다.
- τ, 최대 각속도, 기계적 회전한계, 가상 센서 노이즈/바이어스는 사용자가 정한 값으로 고정합니다.
- 일별 파일을 전체 기간에서 균등하게 골라 대표 4분 관찰 구간을 추출합니다. 학습 구간에서 후보를 탐색하고, 일부 날짜는 검증용으로 분리합니다.
- 자동 튜닝 평가는 실시간 재생과 같은 50Hz 제어 루프, 같은 선형 보간, 같은 노이즈 생성 순서를 씁니다(일치 검증 expander에서 확인 가능).
- 탐색 단계는 학습용 관찰 구간 일부(최대 30개)만 쓰고, 최종 검증은 검증용 관찰 구간 전체 × 노이즈 시드 3개 이상의 평균을 씁니다.
- 목적함수는 제약 조건 방식(자세 RMS비 최소화, 포화율 ≤1%·리밋사이클 없음 위반 시 페널티)과 가중합 방식 중 고를 수 있습니다.
- 서보 부담 지표는 관절 각속도 RMS, 각가속도 RMS, 2Hz 이상 고주파 명령 비율입니다(관절각 자체는 수평 유지에 필요한 동작이라 벌점에서 뺐습니다).
- 최종 추천값은 게인 ×1.5/×2, 서보 τ ×0.5/×1.5, 리밋사이클 검사를 통과한 후보 중에서 고릅니다.

**중요한 과학적 한계**
- 공개 INS는 ACE 기간 중 수집한 1Hz 항해 기록입니다. 제공기관은 원자료가 추가 보정되지 않았고 품질검사도 수행되지 않았다고 설명합니다.
- 기록 PC 시간이 GPS 및 다른 장비와 동기화되었는지는 확실하지 않다고 README에 기재되어 있습니다.
- Hs/Tp는 선박 Heave 운동의 5분 구간 스펙트럼과 선박별 RAO를 이용해 재구성한 자료입니다. 독립 부표의 동시 직접 측정값이 아닙니다.
- INS 자료에는 실제 제작할 캡스톤 플랫폼의 IMU 원시 자이로/가속도 신호가 들어 있지 않습니다. 센서 오차는 모델 가정값입니다.
- 본 프로그램의 서보 모델은 1차 지연·속도 제한·각도 제한을 단순화해 표현합니다. 실제 DS51150의 응답, 부하, 기어 유격, 전원 강하, 구조 탄성은 실측으로 검증해야 합니다.
- D항은 상판 자이로(관절 각속도 포함)를 그대로 쓰므로 Kd가 커지면 빠른 내부 루프가 생겨 진동합니다. Kd 탐색 상한은 이 안정 경계를 반영했습니다.
- 서보 1차 모델(기본)은 관성이 없어 목표각을 지나치지 않으므로 D의 감쇠 효과가 나타나지 않습니다. '서보 2차 모델'을 켜면 상판·적재물 관성에 의한 넘어감이 생기고 D가 이를 줄입니다. 2차 모델의 fn·ζ는 가정값이며 실제 서보 계단 응답으로 측정해 바꿔야 합니다.
- 따라서 자동 최적값은 **이 시뮬레이터 모델에 대한 후보값**입니다. 실물 적용 전에는 낮은 이득부터 단계적으로 시험하고, 독립된 실제 IMU·서보 시험 데이터로 재튜닝해야 합니다.

**출처 및 라이선스**
- Thomas & Pina Estany (2020), ACE INS, Zenodo DOI 10.5281/zenodo.3733228, CC BY 4.0.
- Derkani et al. (2021), ACE reconstructed sea state, Zenodo DOI 10.5281/zenodo.4541564, CC BY 4.0.
- 해상상태 분류: WMO Sea State Code(WMO 코드표 3700, Douglas 척도 기반) — 유의파고 Hs 기준 코드 0~9.
- [ACE INS 전체 원자료](https://zenodo.org/records/3733228) · [Hs/Tp 재구성 자료](https://zenodo.org/records/4541564) · [AARI 선박 정보](https://www.aari.ru/eng)
""")
