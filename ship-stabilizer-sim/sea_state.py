"""
sea_state.py — 물리 기반 선박 운동 모델 (Roll·Pitch 독립 1자유도 × 2)

가상 파도 모드에서 '파도 기울기에 배율을 곱해 선박 각도를 만들던' 이전 방식을 대신한다.
파랑 외력 모멘트가 선박을 움직이고, 관성·복원력·감쇠가 그 응답을 정한다.

    Roll :  (Ixx + A44)·φ'' + B44·φ' + C44·φ = M4(t)
    Pitch:  (Iyy + A55)·θ'' + B55·θ' + C55·θ = M5(t)

출처 표기: [확인] 원문을 직접 열어 대조함 · [유도] 이 파일에서 식을 유도하고 verify_engine.py 에서 숫자로 검산
          [실측] Akademik INS 자료에서 직접 계산 · [추정] 자료가 없어 공학적으로 가정 (실측값처럼 쓰지 말 것)

  E1 운동방정식    (m+a)φ'' + bφ' + cφ = M,  ω0² = c/(m+a),  ζ = b/(2√((m+a)c))
                   Journée & Massie, Offshore Hydromechanics (TU Delft, 2001) 6장:
                   식 6.28 (6-13쪽), 식 6.38b·6.39 (6-15쪽)  [확인]
  E2 복원 계수     C44 = ρ·g·∇·GM_T,   C55 = ρ·g·∇·GM_L        같은 책 6-15쪽  [확인]
  E3 만남 주파수   ωe = ω − k·U·cos μ  (μ = 0: 추종파)            같은 책 식 6.6 (6-3쪽)  [확인]
                   이 앱의 파향 β 는 '선수에서 잰 각' (0° = 선수 정면, 90° = 정횡, 180° = 선미) → μ = 180° − β
  E4 롤 고유주기   T = 2·C·B/√GM,  C = 0.373 + 0.023·(B/d) − 0.043·(L_wl/100)
                   IMO 2008 IS Code (MSC.267(85)) Part A 2.3.4, 14쪽  [확인]
                   → Akademik 실측 롤 주기(아래 E7)로부터 GM_T 를 거꾸로 구하는 데 쓴다.
  E5 파랑 외력     M4 = C44·(유효 파경사),  M5 = C55·(유효 파경사)
                   유효 파경사 = k·a·(sin β 또는 cos β)·F(k_y·B/2 또는 k_x·L/2)·e^(−kT)
                   - 긴 파도(ω→0)에서 선박이 수면 기울기를 그대로 따라가는 정적 극한을 정확히 만족  [유도]
                   - F(x) = 3(sin x − x·cos x)/x³ : 상자형 수선면 위에서 기울기가 고르지 않을 때의 평균 효과  [유도, V15]
                   - e^(−kT) : 흘수 깊이에서의 동압 감쇠. J&M 식 6.75(상자형 부유체 상하동요)의 같은 인자를 회전에 적용 [근사]
                   (회절·방사·연성은 무시한 Froude–Krylov형 근사. 실선 RAO가 아니다.)
  E6 정상 응답     H(ω) = 1 / (1 − (ω/ω0)² + 2iζ·ω/ω0)  →  φ = Σ Re[H(ωe)·α_i·e^{iωe t}]
                   선형 방정식이라 성분별 정상해의 합이 정확한 해 (V15에서 시간 적분과 대조)  [유도]
  E7 실측 보정     Akademik INS 30분 블록 5,463개(1 Hz)의 롤 스펙트럼 첨두 주기 중앙값 13.5 s,
                   파도 주기 Tp 7~16 s 에서 거의 변하지 않음 → 선박 고유 롤 주기로 판단  [실측]
                   감쇠비 ζ44, ζ55 는 같은 자료의 Hs 구간별 롤·피치 RMS 중앙값에 맞춤  [실측 맞춤]
"""
from __future__ import annotations

import math
import numpy as np

G = 9.81
RHO = 1025.0          # 해수 밀도 (kg/m³) [추정: 표준값]

# ----------------------------------------------------------------------------- 선박 제원
# AARI(선주, 북극남극연구소) 공식 제원표: 최대 길이 133.53 m, 최대 폭 23.25 m, 배수량 16,539 t, 흘수 8.5 m  [확인]
AKADEMIK = {"name": "R/V Akademik Tryoshnikov", "L": 133.53, "B": 23.25, "T": 8.5, "disp_t": 16539.0,
            "source": "AARI fleet page (aari.ru) — Maximum length 133.53 m, width 23.25 m, displacement 16539 t, draught 8.5 m"}

# Akademik INS(ACE 2016-11~2017-03) 1 Hz 롤·피치 30분 블록 분석 결과 (2026-11-27, 저장소 data/ 의 같은 자료) [실측]
AKADEMIK_MEASURED = {
    "roll_period_s": 13.5,           # 롤 스펙트럼 첨두 주기 중앙값 (블록 5,463개, 평균 스펙트럼 첨두 13.47 s)
    "roll_period_by_tp": {"Tp<7": 14.2, "7-8": 13.8, "8-9": 13.5, "9-10": 13.1, "10-11": 13.1, "11-12": 13.5, "12-14": 13.5, ">14": 13.5},
    "n_blocks": 5463, "n_blocks_with_wave": 2881,
    # Hs 구간 → (블록 수, Tp 중앙값, 롤 RMS 중앙값, 피치 RMS 중앙값) — 평균(정적 경사)을 뺀 표준편차
    "rms_by_hs": [((0.5, 1.25), 725, 7.8, 0.59, 0.28), ((1.25, 2.5), 1172, 8.5, 1.08, 0.46),
                  ((2.5, 4.0), 742, 8.4, 1.55, 0.71), ((4.0, 6.0), 214, 8.9, 2.64, 1.06)],
}


def imo_roll_C(B: float, d: float, L: float) -> float:
    """E4: IMO IS Code 2.3.4 의 계수 C."""
    return 0.373 + 0.023*(B/d) - 0.043*(L/100.0)


def gm_from_roll_period(T: float, B: float, d: float, L: float) -> float:
    """E4 를 GM 에 대해 푼 값: GM = (2·C·B/T)²."""
    return (2.0*imo_roll_C(B, d, L)*B/T)**2


def roll_period_from_gm(GM: float, B: float, d: float, L: float) -> float:
    return 2.0*imo_roll_C(B, d, L)*B/math.sqrt(GM)


# ----------------------------------------------------------------------------- 물리 계수
def default_params(ship: dict = AKADEMIK) -> dict:
    """기본 물리 계수와 각 값의 출처 구분. 모든 '추정' 값은 사이드바에서 바꿀 수 있다."""
    L, B, T = ship["L"], ship["B"], ship["T"]
    GM_T = gm_from_roll_period(AKADEMIK_MEASURED["roll_period_s"], B, T, L)
    return {
        "L": L, "B": B, "T": T, "disp_t": ship["disp_t"],
        "GM_T": round(GM_T, 3),      # [실측 역산] 롤 주기 13.5 s + IMO 식 → 약 1.71 m
        "kxx_B": 0.35,               # [추정] 롤 관성반경/폭. 0.35~0.40B가 흔히 쓰이는 범위 (부가관성 제외 건조 관성)
        "a44_ratio": None,           # None이면 롤 주기가 실측 13.5 s 가 되도록 부가관성을 정한다 [실측 역산]
        "kyy_L": 0.25,               # [추정] 피치 관성반경/길이 (0.25L 은 관례적 근사)
        "a55_ratio": 1.0,            # [추정] 피치 부가관성 / 건조 관성 (상하·피치 부가질량이 선체 질량과 같은 크기라는 근사)
        "CIL": 0.0625,               # [추정] 수선면 길이방향 2차모멘트 계수: I_L = CIL·B·L³ (상자형 1/12=0.083, 선형 0.05~0.07)
        "zeta44": 0.06,              # [실측 맞춤] 롤 감쇠비: calibrate(U=10 kn) 결과 (Hs 구간별 롤 RMS 중앙값과 맞춤)
        "zeta55": 0.12,              # [실측 맞춤] 피치 감쇠비: 같은 방법
        "U_kn": 10.0,                # [추정] 선속(knots). 압축 INS 자료에 속도가 없어 ACE 순항 속도를 10 kn 로 가정
                                     #        (선속 0 으로 맞추면 롤 RMS 가 실측의 1/2~1/3 → 실측 롤 13.5 s 공진은 만남 주파수 효과로 설명됨)
        "imu_h": 5.0,                # [추정] IMU 설치 높이(롤·피치 중심 위, m) → 레버암 가속도 오염
        "mult_I": 1.0, "mult_C": 1.0, "mult_B": 1.0,   # 민감도 시험용 배율 (관성, 복원, 감쇠)
    }


def coefficients(p: dict) -> dict:
    """E1·E2: 질량 관성모멘트, 부가관성, 복원·감쇠 계수, 고유주기. 단위 SI (kg·m², N·m/rad, N·m·s/rad)."""
    L, B, T = p["L"], p["B"], p["T"]
    mass = p["disp_t"]*1000.0                      # kg  [확인: 배수량]
    vol = mass/RHO                                 # ∇ (m³)  [계산]
    # Roll
    I44 = mass*(p["kxx_B"]*B)**2                   # [계산 ← 추정 kxx]
    C44 = RHO*G*vol*p["GM_T"]                      # E2  [계산]
    if p.get("a44_ratio") is None:                 # 실측 롤 주기를 맞추는 부가관성 (총 관성 = C44·(T/2π)²)
        tot44 = C44*(AKADEMIK_MEASURED["roll_period_s"]/(2*math.pi))**2
        A44 = max(tot44 - I44, 0.0)
    else:
        A44 = p["a44_ratio"]*I44
    # Pitch
    I55 = mass*(p["kyy_L"]*L)**2
    A55 = p["a55_ratio"]*I55
    BM_L = p["CIL"]*B*L**3/vol                     # BM_L = I_L/∇ [계산 ← 추정 CIL]
    GM_L = BM_L                                    # GM_L ≈ BM_L (KB−KG ≪ BM_L) [근사]
    C55 = RHO*G*vol*GM_L                           # E2
    # 민감도 배율
    mI, mC, mB = p.get("mult_I", 1.0), p.get("mult_C", 1.0), p.get("mult_B", 1.0)
    J44, J55 = mI*(I44 + A44), mI*(I55 + A55)
    K44, K55 = mC*C44, mC*C55
    B44c = 2*math.sqrt(J44*K44); B55c = 2*math.sqrt(J55*K55)     # 임계 감쇠 (배율 적용 전 계수 기준)
    B44 = mB*p["zeta44"]*2*math.sqrt((I44 + A44)*C44)             # 감쇠 계수는 '기준 상태'의 ζ로 정한 뒤 배율
    B55 = mB*p["zeta55"]*2*math.sqrt((I55 + A55)*C55)
    w44, w55 = math.sqrt(K44/J44), math.sqrt(K55/J55)
    return {"mass": mass, "vol": vol, "I44": I44, "A44": A44, "C44": C44, "B44": B44, "J44": J44, "K44": K44,
            "I55": I55, "A55": A55, "C55": C55, "B55": B55, "J55": J55, "K55": K55, "GM_L": GM_L,
            "w44": w44, "w55": w55, "T44": 2*math.pi/w44, "T55": 2*math.pi/w55,
            "z44": B44/B44c, "z55": B55/B55c,
            # 외력/복원 비: 긴 파도에서 선박은 수면 기울기를 따라 기울어 평형을 이루므로 파랑 모멘트도 같은 C로 스케일된다
            # (GM을 바꾸면 C와 M이 함께 바뀜) → 각도 응답에서는 1. 남기는 이유는 식의 흐름을 보이기 위해.
            "exc44": 1.0, "exc55": 1.0, "hull": {"L": L, "B": B, "T": T}}


def box_factor(x):
    """E5: F(x) = 3(sin x − x cos x)/x³ (x→0 에서 1). 수선면 폭(길이) 위 파경사의 평균 효과."""
    x = np.asarray(x, float)
    out = np.ones_like(x)
    m = np.abs(x) > 1e-4
    xm = x[m]
    out[m] = 3.0*(np.sin(xm) - xm*np.cos(xm))/xm**3
    small = ~m
    out[small] = 1.0 - x[small]**2/10.0
    return out


def rao(w, wn, zeta):
    """E6: 각도/유효파경사 (복소수)."""
    r = np.asarray(w, float)/wn
    return 1.0/(1.0 - r*r + 2j*zeta*r)


# ----------------------------------------------------------------------------- 파도 → 성분별 유효 파경사
def component_terms(comps, direction_deg: float, p: dict, co: dict | None = None):
    """성분마다 만남 주파수, 롤·피치 유효 파경사(복소 진폭), 응답 RAO를 계산해 둔다.
    comps = (freqs Hz, amps m, phases rad, spread rad) — app.wave_components 와 같은 형식.
    파면 η = a·cos(−ω_e·t + φ) (선박 원점), 수면 기울기 = −a·k·sin(−ω_e·t + φ)."""
    co = co or coefficients(p)
    freqs, amps, phases, spread = comps
    w = 2*np.pi*np.asarray(freqs, float); k = w*w/G
    beta = np.deg2rad(direction_deg) + np.asarray(spread, float)          # 선수에서 잰 파향
    U = p.get("U_kn", 0.0)*0.514444
    we = np.abs(w + k*U*np.cos(beta))                                      # E3, μ = π − β
    we = np.maximum(we, 1e-3)
    att = np.exp(-k*p["T"])                                                # E5 깊이 감쇠
    ky, kx = k*np.abs(np.sin(beta)), k*np.abs(np.cos(beta))
    s_roll = k*np.asarray(amps)*np.sin(beta)*box_factor(ky*p["B"]/2)*att*co["exc44"]
    s_pitch = k*np.asarray(amps)*np.cos(beta)*box_factor(kx*p["L"]/2)*att*co["exc55"]
    H44 = rao(we, co["w44"], co["z44"]); H55 = rao(we, co["w55"], co["z55"])
    return {"w": w, "k": k, "we": we, "amps": np.asarray(amps, float), "ph": np.asarray(phases, float), "beta": beta,
            "s_roll": s_roll, "s_pitch": s_pitch, "H44": H44, "H55": H55}


def response(t, terms, imu_h: float = 0.0):
    """E6 정상 응답 (t: 스칼라 또는 배열, 초). 반환 (도 단위): roll, pitch, 각속도, 각가속도, heave(m),
    그리고 레버암 오염각 acc_roll, acc_pitch (= atan(h·각가속도/g), IMU 가 회전 중심 위 h 에 있을 때)."""
    t = np.asarray(t, float)
    sc = np.ndim(t) == 0
    tt = np.atleast_1d(t)[:, None]
    we, ph = terms["we"][None, :], terms["ph"][None, :]
    ang = -we*tt + ph
    # 수면 기울기 −a·k·sin(ang) = Re[ i·a·k·e^{i·ang} ] → 복소 진폭 i·s
    e = np.exp(1j*ang)
    # 시간 인자를 e^{−iωe t} 로 썼으므로 x'' + 2ζω0x' + ω0²x = ω0²u 의 정상해 배율은 H(ω) 의 켤레복소수
    zr = 1j*terms["s_roll"][None, :]*np.conj(terms["H44"])[None, :]*e
    zp = 1j*terms["s_pitch"][None, :]*np.conj(terms["H55"])[None, :]*e
    # d/dt e^{i(−ωe t)} = −iωe
    roll = zr.real.sum(1); pitch = zp.real.sum(1)
    rr = (zr*(-1j*we)).real.sum(1); pr = (zp*(-1j*we)).real.sum(1)
    ra = (zr*(-we*we)).real.sum(1); pa = (zp*(-we*we)).real.sum(1)
    heave = (terms["amps"][None, :]*np.cos(ang)).sum(1)
    d = np.rad2deg
    out = {"roll": d(roll), "pitch": d(pitch), "roll_rate": d(rr), "pitch_rate": d(pr),
           "roll_acc": d(ra), "pitch_acc": d(pa), "heave": heave,
           "acc_roll": d(np.arctan(imu_h*ra/G)), "acc_pitch": d(np.arctan(imu_h*pa/G))}
    if sc:
        return {k_: float(v[0]) for k_, v in out.items()}
    return out


def free_decay(t, phi0_deg: float, wn: float, zeta: float):
    """E1 의 자유 감쇠 해 (초기각 φ0, 초기 각속도 0). 충격·감쇠 시험용 (도, 도/s)."""
    t = np.asarray(t, float)
    wd = wn*math.sqrt(max(1 - zeta*zeta, 1e-12))
    e = np.exp(-zeta*wn*t)
    phi = phi0_deg*e*(np.cos(wd*t) + zeta*wn/wd*np.sin(wd*t))
    rate = phi0_deg*e*(-(wn*wn)/wd*np.sin(wd*t))
    return np.where(t >= 0, phi, 0.0), np.where(t >= 0, rate, 0.0)


def impulse_response(t, peak_deg: float, wn: float, zeta: float):
    """충격 모멘트(짧은 충격)에 대한 응답: 처음 최대값이 대략 peak 가 되도록 맞춘 감쇠 정현파."""
    t = np.asarray(t, float)
    wd = wn*math.sqrt(max(1 - zeta*zeta, 1e-12))
    tpk = math.atan2(wd, zeta*wn)/wd
    A = peak_deg/(math.exp(-zeta*wn*tpk)*math.sin(wd*tpk))
    e = np.exp(-zeta*wn*t)
    v = A*e*np.sin(wd*t); r = A*e*(wd*np.cos(wd*t) - zeta*wn*np.sin(wd*t))
    return np.where(t >= 0, v, 0.0), np.where(t >= 0, r, 0.0)


# ----------------------------------------------------------------------------- 시간 적분 (검증용)
def integrate_sdof(M_over_K, wn: float, zeta: float, dt: float, x0=0.0, v0=0.0):
    """x'' + 2ζωn x' + ωn² x = ωn²·u(t) 를 4차 룽게–쿠타로 적분 (u 는 dt 간격 배열, 중간값은 선형 보간).
    정상해(E6)와 독립적으로 같은 방정식을 푸는 검산용."""
    u = np.asarray(M_over_K, float); n = len(u)
    x = np.empty(n); v = np.empty(n); x[0], v[0] = x0, v0

    def f(xv, uu):
        return np.array([xv[1], wn*wn*(uu - xv[0]) - 2*zeta*wn*xv[1]])
    s = np.array([x0, v0], float)
    for i in range(n-1):
        u0, u1 = u[i], u[i+1]; um = 0.5*(u0 + u1)
        k1 = f(s, u0); k2 = f(s + 0.5*dt*k1, um); k3 = f(s + 0.5*dt*k2, um); k4 = f(s + dt*k3, u1)
        s = s + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        x[i+1], v[i+1] = s
    return x, v


# ----------------------------------------------------------------------------- 파도 스펙트럼 (물리 모델용)
def wave_components_phys(tp: float, hs: float, count: int = 48, seed: int = 123, spread_deg: float = 18.0):
    """app.wave_components 와 같은 PM 스펙트럼 모양 S ∝ f⁻⁵·exp(−1.25(fp/f)⁴), Hs = 4σ.
    성분 수를 늘리고 0.03~0.35 Hz 를 고르게 덮는다(롤 공진 0.074 Hz 가 빠지지 않도록).
    방향 퍼짐은 이전과 같은 ±18° 범위를 성분마다 섞어 배정한다."""
    rng = np.random.default_rng(seed)
    fp = 1.0/tp
    freqs = np.linspace(0.03, max(0.35, 2.5*fp), count)
    df = freqs[1] - freqs[0]
    spectrum = freqs**-5*np.exp(-1.25*(fp/freqs)**4)
    amps = np.sqrt(2*spectrum*df)
    amps = amps/max(np.sqrt(np.sum(amps**2)/2), 1e-12)*(hs/4.0)
    phases = rng.uniform(0, 2*np.pi, count)
    spread = np.deg2rad(rng.permutation(np.linspace(-spread_deg, spread_deg, count)))
    return freqs, amps, phases, spread


def rms_expected(hs: float, tp: float, direction_deg: float, p: dict, count: int = 160, headings=None):
    """선형 정상 응답의 RMS (도) — 시간 모의 없이 스펙트럼 합으로. headings 를 주면 그 방향들의 평균 분산."""
    co = coefficients(p)
    comps = wave_components_phys(tp, hs, count, seed=1, spread_deg=0.0)
    hds = [direction_deg] if headings is None else list(headings)
    vr = vp = 0.0
    for h in hds:
        tm = component_terms(comps, h, p, co)
        vr += np.sum(np.abs(tm["s_roll"]*tm["H44"])**2)/2
        vp += np.sum(np.abs(tm["s_pitch"]*tm["H55"])**2)/2
    vr /= len(hds); vp /= len(hds)
    return math.degrees(math.sqrt(vr)), math.degrees(math.sqrt(vp))


def calibrate(p: dict, zetas=None, measured=AKADEMIK_MEASURED):
    """Hs 구간별 실측 롤·피치 RMS 중앙값에 맞는 ζ44, ζ55 를 찾는다 (선속 0, 파향 0~360° 고르게 가정).
    반환: (ζ44, ζ55, 표)."""
    zetas = zetas or [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8]
    heads = np.arange(0, 360, 15)
    best = {}
    for axis in ("roll", "pitch"):
        errs = []
        for z in zetas:
            q = dict(p, **({"zeta44": z} if axis == "roll" else {"zeta55": z}))
            e = 0.0
            for (lo, hi), n, tp, r_meas, p_meas in measured["rms_by_hs"]:
                hs = 0.5*(lo + hi)
                r, pi = rms_expected(hs, tp, 0.0, q, headings=heads)
                m = r_meas if axis == "roll" else p_meas
                v = r if axis == "roll" else pi
                e += n*math.log(v/m)**2
            errs.append(e)
        best[axis] = zetas[int(np.argmin(errs))]
    return best["roll"], best["pitch"]


# ----------------------------------------------------------------------------- 그래프·검증용 보조
def rao_curves(p: dict, beta_deg: float, periods, co: dict | None = None) -> dict:
    """규칙파 한 성분에 대한 응답 (파도 주기 축).
    roll_per_slope: 롤 진폭 / 파경사 진폭 k·a (무차원, 1이면 긴 파도에서 수면을 그대로 따라감)
    roll_per_amp  : 롤 진폭(도) / 파 진폭(m)"""
    co = co or coefficients(p)
    T = np.asarray(periods, float); w = 2*np.pi/T; k = w*w/G
    b = math.radians(beta_deg); U = p.get("U_kn", 0.0)*0.514444
    we = np.maximum(np.abs(w + k*U*math.cos(b)), 1e-3)
    att = np.exp(-k*p["T"])
    fr = math.sin(b)*box_factor(k*abs(math.sin(b))*p["B"]/2)*att
    fp = math.cos(b)*box_factor(k*abs(math.cos(b))*p["L"]/2)*att
    r = np.abs(fr*rao(we, co["w44"], co["z44"])); q = np.abs(fp*rao(we, co["w55"], co["z55"]))
    return {"T": T, "Te": 2*np.pi/we, "roll_per_slope": r, "pitch_per_slope": q,
            "roll_per_amp": np.degrees(r*k), "pitch_per_amp": np.degrees(q*k)}


def zero_cross_period(t, x, demean: bool = True) -> float:
    """상향 영교차 평균 주기 (초). 교차가 2번 미만이면 nan. 감쇠 진동은 demean=False 로."""
    x = np.asarray(x, float)
    if demean:
        x = x - np.mean(x)
    idx = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if len(idx) < 2:
        return float("nan")
    tc = t[idx] - x[idx]*(t[idx+1] - t[idx])/(x[idx+1] - x[idx])
    return float(np.mean(np.diff(tc)))


def log_decrement(t, x):
    """연속한 양의 극대값들의 대수 감쇠율 δ 평균과, 그로부터의 감쇠비 ζ = δ/√(4π²+δ²)."""
    x = np.asarray(x, float)
    pk = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 0))[0] + 1
    if len(pk) < 2:
        return float("nan"), float("nan")
    d = float(np.mean(np.log(x[pk[:-1]]/x[pk[1:]])))
    return d, d/math.sqrt(4*math.pi**2 + d*d)
