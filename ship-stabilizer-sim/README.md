# 선박 2축(롤·피치) 수평유지 시뮬레이터 v3

- `app.py` : Streamlit 앱 (실행 파일)
- `verified_sim.py` : 계산 엔진 — 출처가 확인된 식만 사용 (app.py 와 같은 폴더에 있어야 함)
- `verify_engine.py` : 검증 스크립트 (`python verify_engine.py`)
- `docs/수식_출처_원문캡처.pdf` : 각 식의 원문 쪽 캡처
- `data/` : (선택) 경량 항해 데이터. 있으면 647MB 원본을 다시 받지 않음

실행: `pip install -r requirements.txt` 후 `streamlit run app.py`

데이터 출처: ACE INS (Thomas & Pina Estany 2020, Zenodo 10.5281/zenodo.3733228, CC BY 4.0),
재구성 Hs/Tp (Derkani et al. 2021, Zenodo 10.5281/zenodo.4541564, CC BY 4.0)
