"""
verify_engine.py — verified_sim.py 의 모든 식을 해석해와 숫자로 대조한다.

판정 원칙: 시뮬레이션 출력(시간 적분)과, 같은 식을 손으로 풀어 얻은 해석해를 독립적으로 계산해서
허용오차 안에서 같으면 PASS. 허용오차는 각 시험에 적어 둔다.

실행:  python verify_engine.py                         (엔진 시험 V1~V10, V12)
       python verify_engine.py 이전앱.py                (+ V11: 이전 앱 결과를 엔진이 그대로 재현하는지)
       python verify_engine.py 이전앱.py app.py         (+ V13: 새 앱의 튜닝 루프 = 실시간 루프)
"""
import cmath
import math
import sys

import numpy as np

import verified_sim as vs

DT = vs.CONTROL_DT
results = []


def report(name, ok, detail):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")


def ideal(**kw):
    """이상적 센서(노이즈·바이어스 0), 제한 없음(선형 영역)."""
    base = dict(gyro_noise=0.0, gyro_bias=0.0, acc_noise=0.0, rate=1e9, limit=1e9)
    base.update(kw)
    return vs.params(**base)


def sine_input(f, A, T, dt):
    t = np.arange(int(round(T/dt)))*dt
    w = 2*math.pi*f
    return t, A*np.sin(w*t), A*w*np.cos(w*t)


def fit_amp(t, y, f):
    """정상상태 구간에서 sin·cos 최소제곱으로 진폭 추정."""
    w = 2*math.pi*f
    M = np.column_stack([np.sin(w*t), np.cos(w*t), np.ones_like(t)])
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    return math.hypot(c[0], c[1])


# ------------------------------------------------------------------ 해석해 (이산 위상자)
def phasor_residual(f, p, dt):
    """한 축, 이상적 센서, 선형 영역에서 엔진과 '같은 식'을 z 영역에서 손으로 푼 해.
    입력 선박각 S = e^{jωk·dt}, 선박 각속도 = jω·S.  반환: 상판각 / 선박각 (복소수)."""
    w = 2*math.pi*f
    z = cmath.exp(1j*w*dt)
    S = 1.0; Ws = 1j*w
    al, kp, kd, kff, tau = p["alpha"], p["kp"], p["kd"], p["ff"], p["tau"]
    d = int(p["delay_steps"]); model = p["servo_model"]
    # B9 바이어스 추정: b̂[k] = b̂[k−1] − ki·dt·e[k],  e = θ_acc − θ̂  →  b̂[k−1] = −ki·dt/(z−1)·e
    #   보정 각속도 ω_c = ω − b̂[k−1] = ω + g·e,  g = ki·dt/(z−1)   (ki = 0 이면 g = 0 → 기존 식)
    g = p.get("ki", 0.0)*dt/(z - 1)
    # 하부 보완필터: B(1 − α/z) = α·dt·(Ωs + g(S − B)) + (1−α)·S
    B = (al*dt*(Ws + g*S) + (1-al)*S)/(1 - al/z + al*dt*g)
    Bff = B + (tau*(Ws + g*(S - B)) if p["ff_mode"] == "ideal" else 0.0)
    zd = z**(-d)
    # 미지수 x = [E, J, V, C]  (E: 상판 추정, J: 관절, V: 관절 속도 상태, C: 명령)
    # 관절 순간 각속도 Jd = cJ·J + cV·V + cC·C
    legacy = p.get("timing") == "legacy"
    if legacy:
        cJ, cV, cC = (1 - 1/z)/dt, 0.0, 0.0             # 기존 앱: (j[k] − j[k−1])/dt
    elif model == "zoh1":
        cJ, cV, cC = -1/tau, 0.0, zd/(z*tau)          # (u[k−1] − j[k])/τ
    elif model == "zoh2":
        cJ, cV, cC = 0.0, 1.0, 0.0
    else:
        cJ, cV, cC = (1 - 1/z)/dt, 0.0, 0.0             # (j[k] − j[k−1])/dt
    A = np.zeros((4, 4), complex); b = np.zeros(4, complex)
    # (1) 상판 보완필터: E(1−α/z) = α·dt·(Ωs + Jd + g(S + J − E)) + (1−α)(S + J)
    A[0] = [1 - al/z + al*dt*g, -(al*dt*cJ) - (1-al) - al*dt*g, -al*dt*cV, -al*dt*cC]
    b[0] = al*dt*(Ws + g*S) + (1-al)*S
    # (2) 제어: C = −Kff·Bff − Kp·E − Kd·(Ωs + Jd + g(S + J − E))
    A[1] = [kp - kd*g, kd*(cJ + g), kd*cV, 1 + kd*cC]
    b[1] = -kff*Bff - kd*(Ws + g*S)
    # (3),(4) 서보
    if model == "zoh1":
        a = math.exp(-dt/tau)
        A[2] = [0, z - a, 0, -(1-a)*zd]; A[3] = [0, 0, 1, 0]
    elif model == "euler1":
        A[2] = [0, z - (1 - dt/tau), 0, -(dt/tau)*zd]; A[3] = [0, 0, 1, 0]
    else:
        a11, a12, a21, a22, b1, b2 = vs.servo2_coeffs(p["wn"], p["zeta"], dt)
        A[2] = [0, z - a11, -a12, -b1*zd]
        A[3] = [0, -a21, z - a22, -b2*zd]
    x = np.linalg.solve(A, b)
    return S + (z*x[1] if legacy else x[1])           # legacy: 관절각[k+1]과 짝지어 기록


def continuous_residual(f, p):
    """연속시간 교재식: (1 − F_v·G·e^{−sT}) / (1 + G·e^{−sT}·(Kp + Kd·s)), G = 1/(τs+1)."""
    s = 2j*math.pi*f
    G = 1/(p["tau"]*s + 1)
    D = cmath.exp(-s*p["delay_steps"]*DT) if False else 1.0
    Fv = p["ff"]*((1 + p["tau"]*s) if p["ff_mode"] == "ideal" else 1.0)
    return (1 - Fv*G*D)/(1 + G*D*(p["kp"] + p["kd"]*s))


def continuous_residual_delay(f, p, T):
    s = 2j*math.pi*f
    G = 1/(p["tau"]*s + 1); D = cmath.exp(-s*T)
    Fv = p["ff"]*((1 + p["tau"]*s) if p["ff_mode"] == "ideal" else 1.0)
    return (1 - Fv*G*D)/(1 + G*D*(p["kp"] + p["kd"]*s))


def sim_single_axis(p, f, A=2.0, T=120.0, dt=DT, settle=60.0):
    t, th, om = sine_input(f, A, T, dt)
    z0 = np.zeros_like(t)
    out = vs.run(th, z0, om, z0, p, dt=dt)
    m = t >= settle
    return fit_amp(t[m], out["plate_roll"][m], f)/A


print("=" * 78)
print("verified_sim.py 식별 검증  (dt = %.3f s, 50 Hz)" % DT)
print("=" * 78)

# V1 플랜트 구조 (그림 12.13, P2 = 1): 상판각 − 선박각 = 관절각
t, th, om = sine_input(0.15, 3.0, 20.0, DT)
z0 = np.zeros_like(t)
o = vs.run(th, z0, om, z0, vs.params(gyro_noise=0.05, acc_noise=0.1, gyro_bias=0.2))
err = np.max(np.abs(o["plate_roll"] - o["ship_roll"] - o["joint_roll"]))
report("V1 플랜트: 상판각 = 선박각 + 관절각 (A&M 그림 12.13, P2=1)", err < 1e-12, f"최대 차이 {err:.1e} (허용 1e-12)")

# V2 1차 서보 ZOH: j(nτ) = 1 − e^(−n)  (dt와 무관하게 정확)
for model in ("zoh1", "euler1"):
    for dt in (0.02, 0.01):
        p = ideal(tau=0.1, servo_model=model)
        st = vs.new_state((0, 0)); n_tau = round(0.1/dt)
        vals = []
        for k in range(3*n_tau):
            # 제어기 대신 명령 1을 직접 넣기 위해 kp 등을 0, ff=−1, 선박각 1로 둔다 → cmd = +1
            vs.step(st, {"roll": 1.0, "pitch": 0.0, "roll_rate": 0.0, "pitch_rate": 0.0},
                    dict(p, ff=-1.0, kp=0.0, kd=0.0, alpha=0.0), True, 0.0, dt)
            if (k+1) % n_tau == 0:
                vals.append(st["roll"]["joint"])
        exact = [1 - math.exp(-n) for n in (1, 2, 3)]
        e = max(abs(a-b) for a, b in zip(vals, exact))
        if model == "zoh1":
            report(f"V2 1차 서보 ZOH, dt={dt}: j(nτ)=1−e^(−n)", e < 1e-12,
                   f"코드 {[round(v, 6) for v in vals]} vs 해석 {[round(v, 6) for v in exact]}, 최대오차 {e:.1e}")
        else:
            print(f"   (참고) 기존 오일러, dt={dt}: {[round(v, 4) for v in vals]} vs 해석 {[round(v, 4) for v in exact]}"
                  f" → 오차 {e:.3f}  ← 기존 앱이 근사였다는 증거")

# V3 2차 서보: 오버슈트, 피크 시간
p = ideal(servo_model="zoh2", wn=2*math.pi*3, zeta=0.3)
dt = 0.0005
st = vs.new_state((0, 0)); js = []
for k in range(int(1.0/dt)):
    vs.step(st, {"roll": 1.0, "pitch": 0.0, "roll_rate": 0.0, "pitch_rate": 0.0},
            dict(p, ff=-1.0, kp=0.0, kd=0.0, alpha=0.0), True, 0.0, dt)
    js.append(st["roll"]["joint"])
js = np.array(js); k = int(np.argmax(js))
z_, wn = 0.3, 2*math.pi*3
Mp = math.exp(-math.pi*z_/math.sqrt(1-z_**2)); tp = math.pi/(wn*math.sqrt(1-z_**2))
report("V3 2차 서보 (ζ=0.3, 3 Hz): 오버슈트·피크시간", abs(js[k]-1-Mp) < 1e-4 and abs((k+1)*dt-tp) < dt,
       f"오버슈트 코드 {100*(js[k]-1):.3f}% vs 해석 {100*Mp:.3f}%, 피크 {(k+1)*dt:.4f}s vs {tp:.4f}s")

# V4 가속도계 항등식: atan2(회전된 비력) = θ + atan(a/g)
rng = np.random.default_rng(0); e = 0.0
for _ in range(2000):
    th_, a_ = rng.uniform(-30, 30), rng.uniform(-5, 5)
    e = max(e, abs(vs.accel_tilt_deg(th_, a_) - (th_ + math.degrees(math.atan(a_/vs.G)))))
report("V4 가속도계: atan2(f_y, f_z) = θ + atan(a_h/g) (수평가속 오염)", e < 1e-9, f"무작위 2000개 최대 차이 {e:.1e}°")

# V5 보완필터 바이어스 정상 오차 α·b·dt/(1−α)
for al in (0.99, 0.987):
    p = vs.params(alpha=al, gyro_bias=0.3); st = vs.new_state((0, 0)); st["roll"]["base_bias"] = 1.0
    for k in range(int(200/DT)):
        r = vs.step(st, {"roll": 0, "pitch": 0, "roll_rate": 0, "pitch_rate": 0}, p, False, 0.0)
    ex = al*0.3*DT/(1-al)
    report(f"V5 보완필터 바이어스 오차 (α={al})", abs(r["est_roll"]-ex) < 1e-9,
           f"코드 {r['est_roll']:.6f}° vs 해석 {ex:.6f}°")

# V6 보완필터 오염 통과율 (1−α)/|1−α·e^(−jωdt)|
for al in (0.99, 0.987):
    p = vs.params(alpha=al); st = vs.new_state((0, 0)); ts, es = [], []
    for k in range(int(300/DT)):
        r = vs.step(st, {"roll": 0, "pitch": 0, "roll_rate": 0, "pitch_rate": 0}, p, False, 1.0)
        ts.append(r["time"]); es.append(r["est_roll"])
    ts, es = np.array(ts), np.array(es); m = ts > 100
    amp = fit_amp(ts[m], es[m], 0.23)
    ex = (1-al)/abs(1 - al*cmath.exp(-1j*2*math.pi*0.23*DT))
    report(f"V6 보완필터 오염 통과율 0.23 Hz (α={al})", abs(amp-ex)/ex < 1e-6, f"코드 {amp:.6f} vs 해석 {ex:.6f}")

# V7 닫힌루프 전체: 시간 적분 = z 영역 해석해 (모든 서보·지연·FF 방식)
cases = [dict(servo_model="zoh1", delay_steps=0), dict(servo_model="zoh1", delay_steps=2),
         dict(servo_model="zoh1", delay_steps=0, ff_mode="ideal"), dict(servo_model="euler1", delay_steps=0),
         dict(servo_model="zoh2", delay_steps=1), dict(servo_model="zoh1", delay_steps=0, ff=0.6),
         dict(servo_model="euler1", timing="legacy"), dict(servo_model="zoh2", timing="legacy")]
for c in cases:
    for f in (0.1, 0.25):
        p = ideal(**c)
        sim = sim_single_axis(p, f)
        an = abs(phasor_residual(f, p, DT))
        report(f"V7 닫힌루프 {c}, {f} Hz", abs(sim-an)/an < 1e-6, f"시뮬 {sim:.7f} vs 해석 {an:.7f}")

# V8 dt→0 이면 연속시간 교재식으로 수렴 (지연 시간 T 고정)
print("-" * 78)
print("V8 dt → 0 수렴: 이산 해석해(=시뮬과 일치 확인됨) → 연속 교재식")
for c, T in ((dict(ff_mode="static"), 0.0), (dict(ff_mode="static"), 0.04), (dict(ff_mode="ideal"), 0.0)):
    line = []
    for dt in (0.02, 0.005, 0.001, 0.0002):
        p = ideal(servo_model="zoh1", delay_steps=int(round(T/dt)), **c)
        line.append(abs(phasor_residual(0.15, p, dt)))
    cont = abs(continuous_residual_delay(0.15, ideal(**c), T))
    ok = abs(line[-1]-cont) < 0.02*max(cont, 1e-3) + 2e-4
    report(f"V8 {c}, 지연 {T*1000:.0f} ms, 0.15 Hz", ok,
           "dt=0.02/0.005/0.001/0.0002 → " + ", ".join(f"{v:.4f}" for v in line) + f"  | 연속식 {cont:.4f}")

# V9 DC 잔차 (1−Kff)/(1+Kp)
for kff in (0.6, 0.9, 1.0):
    p = ideal(ff=kff); st = vs.new_state((2.0, 0))
    for k in range(int(60/DT)):
        r = vs.step(st, {"roll": 2.0, "pitch": 0, "roll_rate": 0, "pitch_rate": 0}, p)
    ex = 2.0*(1-kff)/(1+p["kp"])
    report(f"V9 정상상태 잔차 (Kff={kff})", abs(r["plate_roll"]-ex) < 1e-9, f"코드 {r['plate_roll']:.9f}° vs 해석 {ex:.9f}°")

# V10 이상적 피드포워드 F_v = 1/P1 (식 12.13)의 효과
print("-" * 78)
for f in (0.1, 0.2):
    a = abs(phasor_residual(f, ideal(ff_mode="static"), DT))
    b = abs(phasor_residual(f, ideal(ff_mode="ideal"), DT))
    print(f"   V10 (참고) {f} Hz 잔차: 정적 Kff=1 → {a:.4f},  이상적 F_v=1+τs → {b:.5f}")


# V11 기존 앱과 일치 (legacy 설정)
def legacy_check(app_path):
    import types
    src = open(app_path, encoding="utf-8").read()
    cut = src.index("st.set_page_config(")
    st_ = types.ModuleType("streamlit"); st_.delta_generator = types.SimpleNamespace(DeltaGenerator=object)
    go = types.ModuleType("plotly.graph_objects"); sub = types.ModuleType("plotly.subplots")
    sub.make_subplots = lambda *a, **k: None
    pl = types.ModuleType("plotly"); pl.graph_objects = go
    sys.modules.update({"streamlit": st_, "plotly": pl, "plotly.graph_objects": go, "plotly.subplots": sub})
    ns = {"__name__": "appmod"}
    exec(compile(src[:cut], "app", "exec"), ns)
    for servo2 in (False, True):
        pa = dict(ff=1.0, kp=1.27, kd=0.02, alpha=0.987, tau=0.1, rate=60.0, limit=15.0, gyro_noise=0.05,
                  gyro_bias=0.3, acc_noise=0.2, servo2=servo2, wn=2*math.pi*3, zeta=0.3)
        pv = vs.params(**{k: v for k, v in pa.items() if k != "servo2"},
                       servo_model="zoh2" if servo2 else "euler1", delay_steps=0, ff_mode="static",
                       timing="legacy")
        a = ns["new_sim_state"]((1.0, -0.5), seed=11); b = vs.new_state((1.0, -0.5), seed=11)
        e = 0.0
        for k in range(3000):
            tt = k*DT
            m = {"roll": 4*math.sin(0.9*tt), "pitch": 3*math.sin(0.7*tt+1), "roll_rate": 3.6*math.cos(0.9*tt),
                 "pitch_rate": 2.1*math.cos(0.7*tt+1), "ax": 0.3*math.sin(1.3*tt), "ay": 0.4*math.cos(1.1*tt)}
            ra = ns["control_step"](a, m, pa, True, 1.0, DT); rb = vs.step(b, m, pv, True, 1.0, DT)
            e = max(e, *(abs(ra[c] - rb[c]) for c in ("plate_roll", "plate_pitch", "est_roll", "joint_pitch")))
        report(f"V11 기존 앱 control_step 과 일치 ({'2차' if servo2 else '1차 오일러'} 서보, 노이즈·바이어스·오염 포함)",
               e < 1e-9, f"60초 최대 차이 {e:.1e}° → 기존 결과(0.497, 0.383)를 이 엔진으로 재현 가능")


# V14 바이어스 추정 (Mahony 외 2005 부록 A 식 22·23의 이산형, verified_sim B9)
print("-" * 78)
#  (a) 끄면(ki=0) 기존 엔진과 비트 단위로 같다 — V5·V7이 그대로 통과하는 것으로 확인
#  (b) 상수 바이어스의 정상 오차 → 0 (적분기의 최종값 정리), 추정값 b̂ → 실제 바이어스
for al, ki in ((0.987, 0.01), (0.998, 0.005)):
    p = vs.params(alpha=al, ki=ki, gyro_bias=0.3); st = vs.new_state((0, 0)); st["roll"]["base_bias"] = 1.0
    for k in range(int(1500/DT)):
        r = vs.step(st, {"roll": 0, "pitch": 0, "roll_rate": 0, "pitch_rate": 0}, p, False, 0.0)
    report(f"V14 바이어스 추정 정상 오차 → 0 (α={al}, ki={ki})", abs(r["est_roll"]) < 1e-6 and abs(st["roll"]["base_bhat"]-0.3) < 1e-6,
           f"추정각 오차 {r['est_roll']:.2e}° (끄면 {al*0.3*DT/(1-al):.4f}°), b̂ = {st['roll']['base_bhat']:.6f}°/s (실제 0.3)")
#  (c) 닫힌루프 전체: 시간 적분 = z 영역 해석해 (바이어스 추정 포함)
for c in (dict(servo_model="zoh1", delay_steps=1, alpha=0.998, ki=0.005),
          dict(servo_model="zoh2", delay_steps=1, alpha=0.996, ki=0.01),
          dict(servo_model="zoh1", delay_steps=0, alpha=0.998, ki=0.005, ff_mode="ideal")):
    for f in (0.05, 0.2):
        p = ideal(**c)
        sim = sim_single_axis(p, f, T=900.0, settle=600.0)
        an = abs(phasor_residual(f, p, DT))
        report(f"V14 닫힌루프 + 바이어스 추정 {c}, {f} Hz", abs(sim-an)/an < 1e-6, f"시뮬 {sim:.7f} vs 해석 {an:.7f}")

if len(sys.argv) > 1:
    legacy_check(sys.argv[1])


def batch_check(app_path):
    import os, types
    import pandas as pd
    src = open(app_path, encoding="utf-8").read(); cut = src.index("st.set_page_config(")
    st_ = types.ModuleType("streamlit"); st_.delta_generator = types.SimpleNamespace(DeltaGenerator=object)
    go = types.ModuleType("plotly.graph_objects"); sub = types.ModuleType("plotly.subplots"); sub.make_subplots = lambda *a, **k: None
    pl = types.ModuleType("plotly"); pl.graph_objects = go
    sys.modules.update({"streamlit": st_, "plotly": pl, "plotly.graph_objects": go, "plotly.subplots": sub})
    ns = {"__name__": "newapp", "__file__": os.path.abspath(app_path)}
    exec(compile(src[:cut], "app", "exec"), ns)
    t = np.arange(0, 240, 1.0)
    seg = pd.DataFrame({"elapsed_s": t, "ship_roll": 3*np.sin(2*np.pi*0.11*t) + np.sin(2*np.pi*0.27*t),
                        "ship_pitch": 1.5*np.sin(2*np.pi*0.09*t + 1)})
    seg["timestamp"] = pd.Timestamp("2017-01-01", tz="UTC") + pd.to_timedelta(t, unit="s")
    prm = dict(ff=1.0, kp=1.27, kd=0.02, alpha=0.987)
    base = dict(tau=0.1, rate=60.0, limit=15.0, gyro_noise=0.3, gyro_bias=0.3, acc_noise=0.5, accel_disturb=1.0,
                wn=2*math.pi*3, zeta=0.3)
    for name, kw in [("1차 ZOH, 지연 0", dict(servo2=False, servo_model="zoh1", delay_steps=0)),
                     ("1차 ZOH, 지연 3", dict(servo2=False, servo_model="zoh1", delay_steps=3)),
                     ("2차, 지연 1", dict(servo2=True, servo_model="zoh2", delay_steps=1)),
                     ("기존 앱 재현", dict(servo2=False, servo_model="euler1", delay_steps=0, timing="legacy"))]:
        r = ns["consistency_check"](seg, prm, dict(base, **kw))
        d = float(r["차이(%)"].max())
        report(f"V13 새 앱 튜닝 루프 = 실시간 루프 ({name}, 노이즈·바이어스·오염 포함)", d < 1e-6,
               f"RMS비 {r['튜닝 평가 RMS비'].iloc[0]:.6f} vs {r['실시간 RMS비'].iloc[0]:.6f}, 차이 {d:.1e}%")
    for name, kw in [("1차 ZOH, 지연 1, kᵢ 0.005", dict(servo2=False, servo_model="zoh1", delay_steps=1, ki=0.005)),
                     ("2차, 지연 2, kᵢ 0.01", dict(servo2=True, servo_model="zoh2", delay_steps=2, ki=0.01))]:
        r = ns["consistency_check"](seg, dict(prm, alpha=0.998), dict(base, **kw))
        d = float(r["차이(%)"].max())
        report(f"V14 새 앱 튜닝 루프 = 실시간 루프 ({name}, α 0.998)", d < 1e-6,
               f"RMS비 {r['튜닝 평가 RMS비'].iloc[0]:.6f} vs {r['실시간 RMS비'].iloc[0]:.6f}, 차이 {d:.1e}%")


if len(sys.argv) > 2:
    batch_check(sys.argv[2])

print("-" * 78)
print("V12 (참고) 대표 조건에서 RMS비 변화: 선박 0.12+0.2 Hz 혼합, 노이즈·바이어스, 가속도 오염 1°")
t = np.arange(int(600/DT))*DT
roll = 3*np.sin(2*math.pi*0.12*t) + 1.5*np.sin(2*math.pi*0.2*t+0.5)
rr = 3*2*math.pi*0.12*np.cos(2*math.pi*0.12*t) + 1.5*2*math.pi*0.2*np.cos(2*math.pi*0.2*t+0.5)
pitch = 0.6*roll[::-1].copy(); prr = -0.6*rr[::-1].copy()
sens = dict(gyro_noise=0.05, gyro_bias=0.3, acc_noise=0.2)
for name, kw in [("기존 앱과 같음 (euler, 시점 혼합)", dict(servo_model="euler1", timing="legacy")),
                 ("시점 일치만 고침", dict(servo_model="euler1", timing="consistent")),
                 ("시점 일치 + ZOH 서보", dict(servo_model="zoh1")),
                 ("+ 지연 1샘플(20 ms)", dict(servo_model="zoh1", delay_steps=1)),
                 ("+ 지연 3샘플(60 ms)", dict(servo_model="zoh1", delay_steps=3))]:
    vals = [vs.rms_ratio(vs.run(roll, pitch, rr, prr, vs.params(**sens, **kw), seed=sd, accel_disturb=1.0))[0]
            for sd in (1, 2, 3)]
    print(f"   {name:<28s} RMS비 {np.mean(vals):.4f}  (시드 3개: {', '.join(f'{v:.4f}' for v in vals)})")

print("=" * 78)
n_ok = sum(ok for _, ok in results)
print(f"결과: {n_ok}/{len(results)} PASS")
