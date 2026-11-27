"""
verified_sim.py — 검증 가능한 식만으로 만든 2축(롤·피치) 상판 수평유지 시뮬레이션 엔진

원칙
  1. 모든 계산 블록은 (a) 출처가 있는 식이거나 (b) 해석해로 정확히 검산되는 식만 쓴다.
  2. 각 블록의 식은 verify_engine.py 에서 해석해(손으로 유도한 식)와 숫자로 대조한다.
  3. 기존 앱(app.py)의 control_step 과 같은 입력·상태·난수 순서를 쓴다.
     legacy 설정(servo_model="euler1", delay_steps=0)이면 기존 결과(0.497, 0.383)를 그대로 재현한다.

블록과 출처 (확인 상태 표기: [확인] 원문을 직접 대조함, [미확인] 원문 쪽수 대조 필요)

  B1 플랜트 구조     y = P2·(P1·u + v),  P2 = 1  →  상판각 = 선박각 + 관절각
     Åström & Murray, Feedback Systems 2판, §12.4, 그림 12.13 (12-20쪽) [확인]
     (v = 측정 외란 = 선박 각도, P1 = 서보, P2 = 1)

  B2 제어 법칙       u = −C(s)·y − F_v(s)·v,  C(s) = Kp + Kd·s,  F_v = Kff
     같은 그림 12.13에서 r = 0 으로 둔 형태 [확인]
     이상적 외란 보상 F_v = 1/P1  (식 12.13, 12-20쪽) [확인]
       → 1차 서보 P1 = 1/(τs+1) 이면 F_v = 1 + τs.  ff_mode="ideal" 로 선택 가능.
       → 정적 Kff = 1 은 이 이상적 보상의 저주파(s→0) 근사.
     D항은 오차가 아니라 측정된 상판 각속도(자이로)에 건다 (기존 앱과 동일).
     PID 일반: Åström & Murray 2판 11장 [절 번호 확인, 쪽수 미확인]

  B3 서보(P1)
     "zoh1"  : 1차 지연 1/(τs+1) 을 명령 고정(ZOH)으로 정확히 이산화
               j[k+1] = a·j[k] + (1−a)·u[k],  a = exp(−dt/τ)
               → j(nτ) = 1 − e^(−n) 이 dt와 무관하게 정확히 성립 (V2에서 검산)
     "zoh2"  : 2차 j'' = ωn²(u − j) − 2ζωn·j' 를 행렬지수로 정확히 이산화 (기존 앱과 동일)
               → 오버슈트 exp(−πζ/√(1−ζ²)) (V3에서 검산)
     "euler1": 기존 앱의 전진 오일러 근사 (실효 시정수가 τ와 약 10% 다름). 재현용.
     속도 제한(±rate)과 각도 제한(±limit)은 하드웨어 사양값 → 다음 학기 실측으로 확정.

  B4 지연            u[k] = cmd[k − delay_steps]  (센서→구동 지연, 샘플 단위)
     교재 12-20쪽: 시간지연이 있으면 이상적 피드포워드(역변환)가 불가능 [확인]
     값은 다음 학기 실측(명령→관절 응답 시각 차)으로 확정.

  B5 가속도계 기울기 (평면 1축)
     비력 f = a − g 를 상판 좌표로 돌린 뒤 θ_acc = atan2(f_y, f_z)
       = θ + atan(a_h / g)   (수평가속 a_h 가 있을 때의 오염; V4에서 항등식 검산)
     tilt 공식: NXP AN3461 "Tilt Sensing Using a Three-Axis Accelerometer" [미확인]

  B6 보완필터        θ̂[k] = α(θ̂[k−1] + ω·dt) + (1−α)·θ_acc[k],   α = τ_f/(τ_f + dt)
     구조 G(s), 1−G(s): Higgins (1975), IEEE TAES AES-11(3), 그림 1, 321~322쪽 [확인]
     정상 바이어스 오차 α·b·dt/(1−α), 오염 통과율 (1−α)/|1−α·e^(−jωdt)| (V5, V6에서 검산)

  B8 기록 시점  timing="consistent": 상판각[k] = 선박각[k] + 관절각[k]  (같은 시각끼리 더함, 센서가 보는 값)
                timing="legacy"    : 상판각[k] = 선박각[k] + 관절각[k+1] (기존 앱; 한 스텝 뒤 관절각과 짝지음)
     기존 방식은 관절이 한 샘플 미래 값으로 기록되어 잔차가 작게 나온다 (V7에서 7자리까지 확인).
     legacy 에서는 상판 각속도도 기존처럼 직전 스텝 평균 관절속도를 쓴다.

  B9 자이로 바이어스 온라인 추정 (ki > 0 일 때만; ki = 0 이면 B6와 완전히 같음)
     연속형  x̂' = y_u − b̂ + k(y_x − x̂),   b̂' = −(y_x − x̂)/λ      (ki = 1/λ)
     Mahony·Hamel·Pflimlin, "Complementary filter design on the special orthogonal group SO(3)",
     44th IEEE CDC-ECC 2005, Appendix A, 식 (22)·(23), 1484쪽 [확인]  — 보완필터 보상기를 C(s) = k + 1/(λs) 로 둔 PI형
     이산형(B6와 같은 후진 오일러):  ω_c = ω − b̂[k−1]
                                     θ̂[k] = α(θ̂[k−1] + ω_c·dt) + (1−α)·θ_acc[k]
                                     b̂[k] = b̂[k−1] − ki·(θ_acc[k] − θ̂[k])·dt
     D항과 이상적 FF에도 보정된 각속도 ω_c 를 쓴다. 상수 바이어스의 정상 오차 → 0 (V14에서 검산).

  B7 성능 지표       RMS비 = RMS(상판각) / RMS(선박각)   (축별, 분모 하한 0.1°) — 기존 앱과 동일 정의

이 엔진에 넣지 않은 것 (근거가 약해서 뺐음)
  - 가상 파도의 흔들림 배율(Hs 0.5/2/4 m → 0.6/2.0/3.5)과 파경사 기반 선박 응답:
    실선 RAO가 아니므로 입력은 '실측 항해 데이터' 또는 '명시한 정현파'만 받는다.
  - 선박 크기에 따른 응답: 근거 있는 RAO 없이 넣지 않는다.
"""
from __future__ import annotations

import math
import numpy as np

G = 9.81
CONTROL_DT = 0.02

DEFAULTS = dict(
    ff=1.0, kp=1.27, kd=0.02, alpha=0.987,          # 최종 추천값
    tau=0.1, rate=300.0, limit=15.0,                  # 서보 (하드웨어 사양값, 다음 학기 실측)
    gyro_noise=0.0, gyro_bias=0.0, acc_noise=0.0,     # 센서 (데이터시트·Allan 분산으로 확정)
    servo_model="zoh1", wn=2*math.pi*3.0, zeta=0.3,   # 서보 모델
    delay_steps=0,                                    # 센서→구동 지연(샘플)
    ff_mode="static",                                 # "static": F_v=Kff, "ideal": F_v=Kff·(1+τs)
    timing="consistent",                              # 기록 시점 (아래 B8 참고)
    ki=0.0,                                           # 자이로 바이어스 추정 이득 (B9, 0이면 끔)
)

LEGACY = dict(servo_model="euler1", delay_steps=0, ff_mode="static", timing="legacy")  # 기존 앱 재현용


# ---------------------------------------------------------------- B5 가속도계
def accel_tilt_deg(theta_deg: float, a_h: float) -> float:
    """평면 1축 가속도계 기울기. 상판이 θ 기울고 수평가속 a_h(m/s², 같은 축 방향)가 있을 때
    비력 (a_h, g)를 상판 좌표로 돌려 atan2로 읽은 각도. = θ + atan(a_h/g)."""
    th = math.radians(theta_deg)
    fy = G*math.sin(th) + a_h*math.cos(th)
    fz = G*math.cos(th) - a_h*math.sin(th)
    return math.degrees(math.atan2(fy, fz))


# ---------------------------------------------------------------- B3 서보 계수
def expm(M: np.ndarray) -> np.ndarray:
    """행렬 지수(스케일링·제곱 + 테일러 20차). 기존 앱 expm_small 과 같은 방법."""
    norm = float(np.abs(M).sum(axis=1).max())
    k = max(0, int(math.ceil(math.log2(norm))) + 1) if norm > 0.5 else 0
    A = M / (2.0**k); E = np.eye(M.shape[0]); term = np.eye(M.shape[0])
    for i in range(1, 21):
        term = term @ A / i; E = E + term
    for _ in range(k):
        E = E @ E
    return E


_CACHE: dict = {}


def servo2_coeffs(wn, zeta, dt):
    key = (float(wn), float(zeta), float(dt))
    if key not in _CACHE:
        M = np.array([[0.0, 1.0, 0.0], [-wn*wn, -2*zeta*wn, wn*wn], [0.0, 0.0, 0.0]]) * dt
        E = expm(M)
        _CACHE[key] = (E[0, 0], E[0, 1], E[1, 0], E[1, 1], E[0, 2], E[1, 2])
    return _CACHE[key]


# ---------------------------------------------------------------- 상태
def new_state(initial=(0.0, 0.0), seed=7):
    """기존 앱 new_sim_state 와 같은 난수 순서(축마다 하부·상판 바이어스 1개씩)."""
    rng = np.random.default_rng(seed)

    def axis(v):
        return {"base_est": float(v), "plate_est": float(v), "joint": 0.0, "joint_rate": 0.0,
                "joint_vel": 0.0, "u_prev": 0.0, "queue": [],
                "base_bias": rng.normal(0, 1), "plate_bias": rng.normal(0, 1),
                "base_bhat": 0.0, "plate_bhat": 0.0}
    return {"t": 0.0, "roll": axis(initial[0]), "pitch": axis(initial[1]), "rng": rng}


def params(**kw):
    p = dict(DEFAULTS); p.update(kw); return p


# ---------------------------------------------------------------- 한 스텝
def step(state, motion, p, controller_on=True, accel_disturb=0.0, dt=CONTROL_DT):
    """motion: roll, pitch (°), roll_rate, pitch_rate (°/s), 선택: ax, ay (m/s², 선박 수평가속).
    기존 control_step 과 같은 계산 순서·난수 순서."""
    rng = state["rng"]
    a_h = {"roll": float(motion.get("ay", 0.0)), "pitch": -float(motion.get("ax", 0.0))}
    extra = {"roll": 0.0, "pitch": 0.0}
    if accel_disturb > 0:   # 기존 앱의 합성 오염(실측 재생용): 0.23 Hz / 0.19 Hz 정현파
        extra["roll"] = accel_disturb*math.sin(2*math.pi*0.23*state["t"])
        extra["pitch"] = accel_disturb*math.sin(2*math.pi*0.19*state["t"]+0.7)
    model = p.get("servo_model", "zoh1")
    row = {"time": state["t"]}
    for axis in ("roll", "pitch"):
        s = state[axis]
        ship = float(motion[axis]); ship_rate = float(motion[f"{axis}_rate"])
        # B1: 상판각 = 선박각 + 관절각
        plate = ship + s["joint"]
        # 상판 실제 각속도 = 선박 각속도 + 관절 각속도(측정 시점의 순간값)
        legacy = p.get("timing") == "legacy"
        if legacy:
            jdot = s["joint_rate"]                            # 기존 앱: 직전 스텝 평균
        elif model == "zoh1":
            jdot = (s["u_prev"] - s["joint"]) / p["tau"]     # 1차계 미분식, 정확
        elif model == "zoh2":
            jdot = s["joint_vel"]                             # 상태변수, 정확
        else:
            jdot = s["joint_rate"]                            # 기존 앱: 직전 스텝 평균
        plate_rate = ship_rate + jdot
        # 센서 (난수 순서: gyro_b, acc_b, gyro_p, acc_p)
        gyro_b = ship_rate + p["gyro_bias"]*s["base_bias"] + rng.normal(0, p["gyro_noise"])
        acc_b = accel_tilt_deg(ship, a_h[axis]) + extra[axis] + rng.normal(0, p["acc_noise"])
        ki = float(p.get("ki", 0.0))
        # B6 보완필터 (하부), B9 바이어스 추정: ω_c = ω − b̂[k−1]
        gyro_b = gyro_b - s["base_bhat"]
        s["base_est"] = p["alpha"]*(s["base_est"] + gyro_b*dt) + (1-p["alpha"])*acc_b
        s["base_bhat"] = s["base_bhat"] - ki*(acc_b - s["base_est"])*dt
        gyro_p = plate_rate + p["gyro_bias"]*s["plate_bias"] + rng.normal(0, p["gyro_noise"])
        acc_p = accel_tilt_deg(plate, a_h[axis]) + extra[axis] + rng.normal(0, p["acc_noise"])
        # B6 보완필터 (상판), B9
        gyro_p = gyro_p - s["plate_bhat"]
        s["plate_est"] = p["alpha"]*(s["plate_est"] + gyro_p*dt) + (1-p["alpha"])*acc_p
        s["plate_bhat"] = s["plate_bhat"] - ki*(acc_p - s["plate_est"])*dt
        # B2 제어 법칙: u = −(Kp + Kd s)·y − F_v·v
        if controller_on:
            ffv = s["base_est"]
            if p.get("ff_mode") == "ideal":                  # F_v = 1/P1 = 1 + τs (식 12.13)
                ffv = s["base_est"] + p["tau"]*gyro_b
            cmd = -p["ff"]*ffv - p["kp"]*s["plate_est"] - p["kd"]*gyro_p
        else:
            cmd = 0.0
        cmd = float(np.clip(cmd, -p["limit"], p["limit"]))
        # B4 지연
        d = int(p.get("delay_steps", 0))
        if d > 0:
            s["queue"].append(cmd)
            u = s["queue"].pop(0) if len(s["queue"]) > d else 0.0
        else:
            u = cmd
        # B3 서보
        j0 = s["joint"]
        if model == "zoh2":
            a11, a12, a21, a22, b1, b2 = servo2_coeffs(p["wn"], p["zeta"], dt)
            v0 = s["joint_vel"]
            new_j = a11*j0 + a12*v0 + b1*u
            vel = float(np.clip(a21*j0 + a22*v0 + b2*u, -p["rate"], p["rate"]))
            new_j = float(np.clip(new_j, -p["limit"], p["limit"]))
            if abs(new_j) >= p["limit"] and new_j*vel > 0:
                vel = 0.0
            s["joint_vel"] = vel
        else:
            if model == "zoh1":
                dj = (1 - math.exp(-dt/p["tau"]))*(u - j0)
            else:  # euler1 (기존 앱)
                dj = (u - j0)*dt/p["tau"]
            dj = float(np.clip(dj, -p["rate"]*dt, p["rate"]*dt))
            new_j = float(np.clip(j0 + dj, -p["limit"], p["limit"]))
        s["joint_rate"] = (new_j - j0)/dt
        s["joint"] = new_j
        s["u_prev"] = u
        row[f"ship_{axis}"] = ship
        row[f"plate_{axis}"] = ship + (new_j if legacy else j0)
        row[f"est_{axis}"] = s["base_est"]
        row[f"acc_{axis}"] = acc_b
        row[f"joint_{axis}"] = new_j if legacy else j0
        row[f"cmd_{axis}"] = cmd
    state["t"] += dt
    return row


# ---------------------------------------------------------------- 실행·지표
def run(ship_roll, ship_pitch, roll_rate, pitch_rate, p, dt=CONTROL_DT, seed=7,
        ax=None, ay=None, accel_disturb=0.0, controller_on=True):
    """50 Hz 배열 입력으로 전체 시뮬레이션. 반환: 열 이름 → 배열."""
    n = len(ship_roll)
    st = new_state((float(ship_roll[0]), float(ship_pitch[0])), seed)
    rows = []
    for k in range(n):
        m = {"roll": ship_roll[k], "pitch": ship_pitch[k],
             "roll_rate": roll_rate[k], "pitch_rate": pitch_rate[k],
             "ax": 0.0 if ax is None else ax[k], "ay": 0.0 if ay is None else ay[k]}
        rows.append(step(st, m, p, controller_on, accel_disturb, dt))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def rms(x):
    x = np.asarray(x, float); return float(np.sqrt(np.mean(x*x)))


def rms_ratio(out, floor=0.1):
    """RMS비 = 상판 RMS / 선박 RMS, 롤·피치 평균. 분모 하한 0.1°(기존 앱과 동일)."""
    r = rms(out["plate_roll"]) / max(rms(out["ship_roll"]), floor)
    q = rms(out["plate_pitch"]) / max(rms(out["ship_pitch"]), floor)
    return 0.5*(r + q), r, q


# ---------------------------------------------------------------- 앱 안에서 돌리는 빠른 자체 검증
def analytic_residual(f, p, dt=CONTROL_DT):
    """이상적 센서·선형 영역에서 step()과 같은 식을 z 영역에서 손으로 푼 해 (상판각/선박각, 복소수).
    verify_engine.py 의 phasor_residual 과 같은 유도."""
    import cmath
    w = 2*math.pi*f; z = cmath.exp(1j*w*dt); S = 1.0; Ws = 1j*w
    al, kp, kd, kff, tau = p["alpha"], p["kp"], p["kd"], p["ff"], p["tau"]
    d = int(p.get("delay_steps", 0)); model = p.get("servo_model", "zoh1"); legacy = p.get("timing") == "legacy"
    # B9: b̂[k−1] = −(ki·dt/(z−1))·(θ_acc − θ̂)  →  보정 각속도 ω_c = ω + kq·(θ_acc − θ̂),  kq = ki·dt/(z−1)
    kq = float(p.get("ki", 0.0))*dt/(z - 1)
    B = (al*dt*Ws + (1-al + al*dt*kq)*S)/(1 - al/z + al*dt*kq)
    Bff = B + (tau*(Ws + kq*(S - B)) if p.get("ff_mode") == "ideal" else 0.0)
    zd = z**(-d)
    if legacy:
        cJ, cV, cC = (1 - 1/z)/dt, 0.0, 0.0
    elif model == "zoh1":
        cJ, cV, cC = -1/tau, 0.0, zd/(z*tau)
    elif model == "zoh2":
        cJ, cV, cC = 0.0, 1.0, 0.0
    else:
        cJ, cV, cC = (1 - 1/z)/dt, 0.0, 0.0
    A = np.zeros((4, 4), complex); b = np.zeros(4, complex)
    A[0] = [1 - al/z + al*dt*kq, -(al*dt*cJ) - (1-al) - al*dt*kq, -al*dt*cV, -al*dt*cC]; b[0] = al*dt*Ws + (1-al + al*dt*kq)*S
    A[1] = [kp - kd*kq, kd*(cJ + kq), kd*cV, 1 + kd*cC]; b[1] = -kff*Bff - kd*Ws - kd*kq*S
    if model == "zoh1":
        a = math.exp(-dt/tau); A[2] = [0, z - a, 0, -(1-a)*zd]; A[3] = [0, 0, 1, 0]
    elif model == "euler1":
        A[2] = [0, z - (1 - dt/tau), 0, -(dt/tau)*zd]; A[3] = [0, 0, 1, 0]
    else:
        a11, a12, a21, a22, b1, b2 = servo2_coeffs(p["wn"], p["zeta"], dt)
        A[2] = [0, z - a11, -a12, -b1*zd]; A[3] = [0, -a21, z - a22, -b2*zd]
    x = np.linalg.solve(A, b)
    return S + (z*x[1] if legacy else x[1])


def self_test(p_user: dict | None = None):
    """화면에서 누르는 빠른 검증. 반환: [(시험, 출처, 코드값, 해석값, 통과)]"""
    out = []
    base = dict(gyro_noise=0.0, gyro_bias=0.0, acc_noise=0.0, rate=1e9, limit=1e9)
    Z = {"roll": 0.0, "pitch": 0.0, "roll_rate": 0.0, "pitch_rate": 0.0}
    # 1) 1차 서보 63.2%
    p = params(**base, tau=0.1, servo_model="zoh1", ff=-1.0, kp=0.0, kd=0.0, alpha=0.0)
    st = new_state((0, 0))
    for _ in range(5):
        step(st, {"roll": 1.0, "pitch": 0.0, "roll_rate": 0.0, "pitch_rate": 0.0}, p)
    out.append(("1차 서보: t=τ 에서 63.2%", "시정수 정의 (Wikipedia Time constant)",
                st["roll"]["joint"], 1-math.exp(-1), abs(st["roll"]["joint"]-(1-math.exp(-1))) < 1e-9))
    # 2) 2차 서보 오버슈트
    dt2 = 0.0005; p = params(**base, servo_model="zoh2", wn=2*math.pi*3, zeta=0.3, ff=-1.0, kp=0.0, kd=0.0, alpha=0.0)
    st = new_state((0, 0)); mx = 0.0
    for _ in range(int(0.5/dt2)):
        step(st, {"roll": 1.0, "pitch": 0.0, "roll_rate": 0.0, "pitch_rate": 0.0}, p, dt=dt2); mx = max(mx, st["roll"]["joint"])
    ex = math.exp(-math.pi*0.3/math.sqrt(1-0.09))
    out.append(("2차 서보 오버슈트 (ζ=0.3)", "Toronto Metropolitan 교재 식 7-10", mx-1, ex, abs(mx-1-ex) < 1e-3))
    # 3) 가속도계 항등식
    e = max(abs(accel_tilt_deg(t, a) - (t + math.degrees(math.atan(a/G)))) for t in (-20, -3, 0, 7, 25) for a in (-3, -0.5, 0.4, 2))
    out.append(("가속도계: atan2 = θ + atan(a/g)", "NXP AN3461 식 1·25", e, 0.0, e < 1e-9))
    # 4) 보완필터 바이어스 오차
    al = 0.987; p = params(alpha=al, gyro_bias=0.3, gyro_noise=0.0, acc_noise=0.0); st = new_state((0, 0)); st["roll"]["base_bias"] = 1.0
    for _ in range(int(150/CONTROL_DT)):
        r = step(st, Z, p, False)
    ex = al*0.3*CONTROL_DT/(1-al)
    out.append(("보완필터 바이어스 오차 α·b·dt/(1−α)", "Higgins 그림 2(B) 이산화", r["est_roll"], ex, abs(r["est_roll"]-ex) < 1e-9))
    # 5) 정상상태 잔차
    p = params(**base, ff=0.9); st = new_state((2.0, 0))
    for _ in range(int(60/CONTROL_DT)):
        r = step(st, {"roll": 2.0, "pitch": 0.0, "roll_rate": 0.0, "pitch_rate": 0.0}, p)
    ex = 2.0*(1-0.9)/(1+p["kp"])
    out.append(("정상상태 잔차 (1−Kff)/(1+Kp)", "A&M 식 (12.11), s→0", r["plate_roll"], ex, abs(r["plate_roll"]-ex) < 1e-9))
    # 6) 닫힌루프 전체: 현재 화면 설정(센서 이상화)으로 0.15 Hz 정현파
    pu = dict(p_user or {}); pu.update(base)
    p = params(**{k: v for k, v in pu.items() if k in DEFAULTS})
    f = 0.15; T = 80.0 if not p.get("ki") else 500.0   # 바이어스 추정의 느린 과도가 사라질 때까지
    n = int(T/CONTROL_DT); t = np.arange(n)*CONTROL_DT; A = 2.0
    th = A*np.sin(2*math.pi*f*t); om = A*2*math.pi*f*np.cos(2*math.pi*f*t); z0 = np.zeros(n)
    o = run(th, z0, om, z0, p)
    m = t > T/2; M = np.column_stack([np.sin(2*math.pi*f*t[m]), np.cos(2*math.pi*f*t[m]), np.ones(m.sum())])
    c, *_ = np.linalg.lstsq(M, o["plate_roll"][m], rcond=None); amp = math.hypot(c[0], c[1])/A
    an = abs(analytic_residual(f, p))
    out.append(("닫힌루프 잔차 0.15 Hz (현재 설정)", "A&M 그림 12.13 구조의 z 영역 해", amp, an, abs(amp-an)/max(an, 1e-12) < 1e-4))
    # 7) 바이어스 추정: 정지 상태에서 상수 바이어스 0.3°/s 의 정상 오차 → 0 (끄면 α·b·dt/(1−α))
    ki_ = float(pu.get("ki", 0.0)) or 0.005; al = 0.998
    p = params(alpha=al, ki=ki_, gyro_bias=0.3, gyro_noise=0.0, acc_noise=0.0); st = new_state((0, 0)); st["roll"]["base_bias"] = 1.0
    for _ in range(int(600/CONTROL_DT)):
        r = step(st, Z, p, False)
    out.append((f"바이어스 추정 (ki={ki_:g}): 정상 오차 → 0, 추정 b̂ → 0.3°/s", "Mahony 외 2005 부록 A 식 (22)·(23)",
                r["est_roll"], 0.0, abs(r["est_roll"]) < 1e-6 and abs(st["roll"]["base_bhat"] - 0.3) < 1e-6))
    return out
