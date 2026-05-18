# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Stock prediction project in Python 3.9. The repository is in early setup — no source files exist yet.

## Environment

- Python 3.9.13 (CPython)
- Virtual environment at `.venv/` managed by `virtualenv`

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install dependencies (once a requirements file exists)
pip install -r requirements.txt
```

## Tooling (from .gitignore)

The project is configured to support:
- **Ruff** — linting and formatting (`ruff check .`, `ruff format .`)
- **Jupyter Notebooks** — `.ipynb_checkpoints/` is ignored
- **Streamlit** — `.streamlit/secrets.toml` is gitignored (keep secrets out of the repo)
- **Marimo** — notebook alternative

Run linting once Ruff is installed:
```bash
ruff check .
ruff format .
```

---

## 프로젝트 파일 구조

| 파일 | 역할 |
|------|------|
| `analyzer.py` | 실시간 매수/매도 추천 메인 스크립트 (pykrx 기반) |
| `stock_patterns.md` | 캔들·차트 패턴 및 기술적 지표 레퍼런스 |

실행:
```bash
source .venv/bin/activate
pip install pykrx pandas numpy
python analyzer.py
```

---

## 주식 분석 태스크 (Stock Analysis Task)

추가 지시가 있을 때마다 아래 지침에 따라 삼성전자 주식 분석을 수행한다.

### 분석 대상

- **기업명:** 삼성전자
- **티커:** Samsung Electronics Co Ltd
- **주식 번호:** KRX: 005930

### 역할 설정

20년 경력의 전문 펀드매니저 관점으로 분석한다.

### 수행 절차

1. **패턴 학습** — 아래 URL에서 주식 패턴(캔들, 이동평균, 거래량 등) 원리를 먼저 파악한다.
   - https://okuk.tistory.com/487

2. **실시간 데이터 수집** — 아래 항목을 웹 검색·페이지 조회로 수집한다.
   - 현재가, 일중 범위, 52주 고가/저가, 거래량
   - 기술적 지표: RSI(14), MACD(12,26), 5일·20일·50일·200일 이동평균선
   - 주요 지지선·저항선
   - 최근 뉴스: 실적, 수급(외국인/기관), 이벤트(파업·정책·경쟁사 등)
   - 주요 증권사 목표주가 및 투자의견

3. **검증 원칙**
   - 모든 판단은 수집된 실제 수치 기반으로 한다. 유추만으로 결론 내리지 않는다.
   - 수치가 상충하면 가장 최신·신뢰 출처를 우선하고 이유를 명시한다.
   - 연산이 가능한 값은 직접 계산해 검증한다.

4. **출력 구조** — 아래 섹션을 순서대로 작성한다.
   1. 현재 주가 스냅샷 (테이블)
   2. 최근 주가 흐름 패턴 재구성 (시계열 + 패턴명)
   3. 기술적 지표 종합 (테이블)
   4. 펀더멘털 분석 (실적·수급)
   5. 핵심 리스크
   6. 주요 가격 레벨 (지지·저항 레이어 도식)
   7. 시나리오별 예측 (강세·중립·약세, 각 확률 포함)
   8. 매매 전략 결론 (단기·중기, 진입가·목표가·손절가 명시)
   9. 종합 판단 (테이블 + 한 문단 의견)
   10. Sources (웹 출처 링크)

### 매매 전략 필수 포함 항목

- 매수 진입 타깃 가격
- 1차 차익실현 목표 가격
- 손절 기준 가격
- 분할 매수/매도 비중 권고
