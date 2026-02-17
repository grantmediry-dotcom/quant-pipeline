"""
Data Agent - data collection, cleaning, and caching.
"""
import json
import os
import sys
import time

import akshare as ak
import numpy as np
import pandas as pd
import tushare as ts

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    API_DELAY,
    DAILY_QUOTES_PROVIDER,
    DATA_DIR,
    END_DATE,
    PRICE_ADJ_MODE,
    START_DATE,
    STOCK_INFO_PROVIDER,
    TUSHARE_TOKEN,
)
from core.agent_base import BaseAgent
from utils.helpers import ensure_dir, load_parquet, save_parquet


class DataAgent(BaseAgent):
    """Data Agent: mixed providers + cleaning pipeline."""

    agent_name = "DataAgent"

    def __init__(self):
        super().__init__()

        self.price_adj_mode = PRICE_ADJ_MODE if PRICE_ADJ_MODE in {"qfq", "hfq", "none"} else "qfq"
        if self.price_adj_mode != PRICE_ADJ_MODE:
            print(f"[DataAgent] warning: invalid PRICE_ADJ_MODE={PRICE_ADJ_MODE}, fallback=qfq")

        self.stock_info_provider = (
            STOCK_INFO_PROVIDER if STOCK_INFO_PROVIDER in {"auto", "tushare", "akshare"} else "auto"
        )
        if self.stock_info_provider != STOCK_INFO_PROVIDER:
            print(f"[DataAgent] warning: invalid STOCK_INFO_PROVIDER={STOCK_INFO_PROVIDER}, fallback=auto")

        self.daily_quotes_provider = (
            DAILY_QUOTES_PROVIDER if DAILY_QUOTES_PROVIDER in {"auto", "tushare", "akshare"} else "auto"
        )
        if self.daily_quotes_provider != DAILY_QUOTES_PROVIDER:
            print(f"[DataAgent] warning: invalid DAILY_QUOTES_PROVIDER={DAILY_QUOTES_PROVIDER}, fallback=auto")

        self.pro = None
        if TUSHARE_TOKEN:
            ts.set_token(TUSHARE_TOKEN)
            self.pro = ts.pro_api()
        else:
            print("[DataAgent] warning: missing TUSHARE_TOKEN, remote tushare calls disabled")

        ensure_dir(DATA_DIR)
        print(
            "[DataAgent] init done "
            f"(PRICE_ADJ_MODE={self.price_adj_mode}, "
            f"STOCK_INFO_PROVIDER={self.stock_info_provider}, "
            f"DAILY_QUOTES_PROVIDER={self.daily_quotes_provider})"
        )

    def _require_pro_api(self, action: str) -> None:
        if self.pro is None:
            raise RuntimeError(f"[DataAgent] {action} needs valid TUSHARE_TOKEN")

    @staticmethod
    def _load_cache_meta(meta_path: str) -> dict:
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_cache_meta(meta_path: str, data: dict) -> None:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _to_ts_code(code: str) -> str:
        code = str(code).zfill(6)
        if code.startswith(("6", "9", "5")):
            return f"{code}.SH"
        return f"{code}.SZ"

    @staticmethod
    def _normalize_stock_info(df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["ts_code", "name", "list_date", "delist_date", "list_status", "is_st"]
        out = df.copy()
        for col in required_cols:
            if col not in out.columns:
                out[col] = ""
        out["name"] = out["name"].fillna("").astype(str)
        out["list_date"] = out["list_date"].fillna("").astype(str)
        out["delist_date"] = out["delist_date"].fillna("").astype(str)
        out["list_status"] = out["list_status"].fillna("").astype(str)
        out["is_st"] = out["is_st"].fillna(0).astype(int)
        out = out[required_cols]
        out = out.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
        return out

    def _fetch_stock_info_tushare(self) -> pd.DataFrame | None:
        self._require_pro_api("get_stock_info(tushare)")
        frames = []
        for status in ["L", "D", "P"]:
            part = self.pro.stock_basic(
                exchange="",
                list_status=status,
                fields="ts_code,name,list_date,delist_date,list_status",
            )
            time.sleep(API_DELAY)
            if part is not None and not part.empty:
                frames.append(part)

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        status_priority = {"L": 0, "P": 1, "D": 2}
        df["_status_priority"] = df["list_status"].map(status_priority).fillna(9)
        df = (
            df.sort_values(["ts_code", "_status_priority"])
            .drop_duplicates(subset=["ts_code"], keep="first")
            .drop(columns=["_status_priority"])
        )
        df = df[~df["ts_code"].str.endswith(".BJ")]
        df["is_st"] = df["name"].str.contains("ST", na=False).astype(int)
        return self._normalize_stock_info(df)

    def _fetch_stock_info_akshare(self) -> pd.DataFrame | None:
        raw = None
        try:
            raw = ak.stock_info_a_code_name()
        except Exception:
            raw = None

        if raw is None or raw.empty:
            try:
                raw = ak.stock_zh_a_spot_em()[["代码", "名称"]]
            except Exception:
                return None

        code_col = "code" if "code" in raw.columns else "代码"
        name_col = "name" if "name" in raw.columns else "名称"
        if code_col not in raw.columns or name_col not in raw.columns:
            return None

        df = raw[[code_col, name_col]].copy()
        df.columns = ["code", "name"]
        df["code"] = df["code"].astype(str).str.zfill(6)
        df = df[~df["code"].str.startswith(("4", "8"))]
        df["ts_code"] = df["code"].map(self._to_ts_code)
        df["is_st"] = df["name"].str.contains("ST", na=False).astype(int)
        df["list_date"] = ""
        df["delist_date"] = ""
        df["list_status"] = "L"
        df = df.drop(columns=["code"])
        return self._normalize_stock_info(df)

    def get_stock_info(self) -> pd.DataFrame:
        cache_path = os.path.join(DATA_DIR, "stock_info.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            required_cols = {"ts_code", "name", "list_date", "delist_date", "list_status", "is_st"}
            if required_cols.issubset(set(cached.columns)):
                print(f"[DataAgent] stock_info cache hit symbols={len(cached)}")
                return cached
            cached = self._normalize_stock_info(cached)
            if self.pro is None:
                print("[DataAgent] warning: stock_info cache schema upgraded from local cache")
                return cached

        order = ["tushare", "akshare"] if self.stock_info_provider == "auto" else [self.stock_info_provider]
        for provider in order:
            try:
                print(f"[DataAgent] fetching stock_info provider={provider}")
                df = self._fetch_stock_info_tushare() if provider == "tushare" else self._fetch_stock_info_akshare()
                if df is not None and not df.empty:
                    save_parquet(df, cache_path)
                    print(f"[DataAgent] stock_info done provider={provider}, symbols={len(df)}, ST={int(df['is_st'].sum())}")
                    return df
            except Exception as e:
                print(f"[DataAgent] stock_info failed provider={provider}: {e}")

        if cached is not None:
            print("[DataAgent] warning: fallback to local stock_info cache")
            return cached

        print("[DataAgent] warning: no stock_info available")
        return pd.DataFrame(columns=["ts_code", "name", "list_date", "delist_date", "list_status", "is_st"])

    def _get_trade_dates_from_akshare(self) -> list[str]:
        df = ak.stock_zh_index_daily(symbol="sh000852")
        if df is None or df.empty:
            raise RuntimeError("[DataAgent] AKShare unable to fetch trade calendar")
        dates = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y%m%d")
        dates = dates[(dates >= START_DATE) & (dates <= END_DATE)]
        return sorted(dates.dropna().unique().tolist())

    def _get_trade_dates_from_daily(self) -> list[str]:
        cache_path = os.path.join(DATA_DIR, "trade_dates.parquet")
        cached = load_parquet(cache_path)
        if cached is not None and "trade_date" in cached.columns:
            return cached["trade_date"].astype(str).tolist()

        dates = []
        if self.pro is not None:
            print("[DataAgent] fetching trade calendar provider=tushare")
            try:
                df = self.pro.daily(ts_code="000001.SZ", start_date=START_DATE, end_date=END_DATE)
                time.sleep(API_DELAY)
                if df is not None and not df.empty:
                    dates = sorted(df["trade_date"].astype(str).unique().tolist())
            except Exception as e:
                print(f"[DataAgent] trade calendar failed provider=tushare: {e}")

        if not dates:
            print("[DataAgent] fetching trade calendar provider=akshare")
            dates = self._get_trade_dates_from_akshare()

        save_parquet(pd.DataFrame({"trade_date": dates}), cache_path)
        print(f"[DataAgent] trade calendar done days={len(dates)}")
        return dates

    def _fetch_raw_quotes_tushare(self) -> pd.DataFrame:
        self._require_pro_api("_fetch_raw_quotes_tushare")
        trade_dates = self._get_trade_dates_from_daily()
        print(f"[DataAgent] quotes fetch provider=tushare, trade_dates={len(trade_dates)}")

        all_data = []
        failed_dates = []
        for i, date in enumerate(trade_dates):
            try:
                df = self.pro.daily(trade_date=date)
                time.sleep(API_DELAY)
                if df is not None and not df.empty:
                    df = df[~df["ts_code"].str.endswith(".BJ")]
                    all_data.append(df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pre_close"]])
                if (i + 1) % 100 == 0:
                    print(f"  processed {i + 1}/{len(trade_dates)}")
            except Exception as e:
                failed_dates.append(date)
                print(f"  fetch failed date={date}: {e}")
                time.sleep(2)

        for retry in range(3):
            if not failed_dates:
                break
            still_failed = []
            print(f"  retry {retry + 1}, pending={len(failed_dates)}")
            for date in failed_dates:
                try:
                    time.sleep(2)
                    df = self.pro.daily(trade_date=date)
                    time.sleep(API_DELAY)
                    if df is not None and not df.empty:
                        df = df[~df["ts_code"].str.endswith(".BJ")]
                        all_data.append(df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pre_close"]])
                    else:
                        still_failed.append(date)
                except Exception:
                    still_failed.append(date)
            failed_dates = still_failed

        if not all_data:
            raise RuntimeError("[DataAgent] tushare quotes returned no data")
        if failed_dates:
            print(f"[DataAgent] warning: unresolved trade dates={len(failed_dates)}")

        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        for col in ["open", "high", "low", "close", "vol", "amount", "pre_close"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        return result

    def _fetch_raw_quotes_akshare(self) -> pd.DataFrame:
        print("[DataAgent] quotes fetch provider=akshare (symbol loop)")

        stock_info = self.get_stock_info()
        if stock_info is None or stock_info.empty or "ts_code" not in stock_info.columns:
            raise RuntimeError("[DataAgent] akshare quotes require stock_info")

        stock_info = stock_info.copy()
        stock_info["code"] = stock_info["ts_code"].astype(str).str[:6]
        stock_info = stock_info[stock_info["code"].str.match(r"^\d{6}$", na=False)]
        stock_info = stock_info[~stock_info["code"].str.startswith(("4", "8"))]
        stock_info = stock_info.drop_duplicates(subset=["code"])

        code_to_ts = dict(zip(stock_info["code"], stock_info["ts_code"]))
        codes = stock_info["code"].tolist()

        def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        all_data = []
        failed = []
        for i, code in enumerate(codes):
            try:
                hist = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=START_DATE,
                    end_date=END_DATE,
                    adjust="",
                )
                if hist is None or hist.empty:
                    continue

                date_col = pick_col(hist, ["日期", "date", "Date"])
                open_col = pick_col(hist, ["开盘", "open", "Open"])
                high_col = pick_col(hist, ["最高", "high", "High"])
                low_col = pick_col(hist, ["最低", "low", "Low"])
                close_col = pick_col(hist, ["收盘", "close", "Close"])
                vol_col = pick_col(hist, ["成交量", "vol", "volume", "Volume"])
                amount_col = pick_col(hist, ["成交额", "amount", "Amount"])

                if not all([date_col, open_col, high_col, low_col, close_col]):
                    failed.append(code)
                    continue

                frame = pd.DataFrame(
                    {
                        "trade_date": pd.to_datetime(hist[date_col], errors="coerce").dt.strftime("%Y%m%d"),
                        "open": pd.to_numeric(hist[open_col], errors="coerce"),
                        "high": pd.to_numeric(hist[high_col], errors="coerce"),
                        "low": pd.to_numeric(hist[low_col], errors="coerce"),
                        "close": pd.to_numeric(hist[close_col], errors="coerce"),
                        "vol": pd.to_numeric(hist[vol_col], errors="coerce") if vol_col else np.nan,
                        "amount": pd.to_numeric(hist[amount_col], errors="coerce") if amount_col else np.nan,
                    }
                )
                frame["ts_code"] = code_to_ts.get(code, self._to_ts_code(code))
                frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date").reset_index(drop=True)
                frame["pre_close"] = frame["close"].shift(1)
                frame = frame[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "pre_close"]]
                if not frame.empty:
                    all_data.append(frame)

                if (i + 1) % 200 == 0:
                    print(f"  processed {i + 1}/{len(codes)} symbols")
                time.sleep(0.02)
            except Exception:
                failed.append(code)

        if not all_data:
            raise RuntimeError("[DataAgent] akshare quotes returned no data")
        if failed:
            print(f"[DataAgent] warning: failed symbols={len(set(failed))}")

        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        for col in ["open", "high", "low", "close", "vol", "amount", "pre_close"]:
            result[col] = pd.to_numeric(result[col], errors="coerce")
        return result

    def _fetch_raw_quotes(self) -> pd.DataFrame:
        cache_path = os.path.join(DATA_DIR, "daily_quotes_raw.parquet")
        meta_path = os.path.join(DATA_DIR, "daily_quotes_raw.meta.json")
        cached = load_parquet(cache_path)
        meta = self._load_cache_meta(meta_path)

        cached_provider = str(meta.get("provider", "")).lower()
        if cached is not None:
            cache_match = (
                cached_provider == self.daily_quotes_provider
                or (self.daily_quotes_provider == "auto" and cached_provider in {"tushare", "akshare"})
            )
            if cache_match:
                print(f"[DataAgent] raw quotes cache hit rows={len(cached)} provider={cached_provider or 'unknown'}")
                return cached
            print(
                f"[DataAgent] raw quotes cache provider mismatch "
                f"(cached={cached_provider or 'unknown'}, current={self.daily_quotes_provider}), rebuilding"
            )

        order = ["tushare", "akshare"] if self.daily_quotes_provider == "auto" else [self.daily_quotes_provider]
        for provider in order:
            try:
                result = self._fetch_raw_quotes_tushare() if provider == "tushare" else self._fetch_raw_quotes_akshare()
                save_parquet(result, cache_path)
                self._save_cache_meta(
                    meta_path,
                    {
                        "provider": provider,
                        "start_date": START_DATE,
                        "end_date": END_DATE,
                    },
                )
                print(f"[DataAgent] raw quotes fetched rows={len(result)} provider={provider}")
                return result
            except Exception as e:
                print(f"[DataAgent] raw quotes failed provider={provider}: {e}")

        if cached is not None:
            print("[DataAgent] warning: fallback to existing raw quotes cache")
            return cached
        raise RuntimeError("[DataAgent] unable to fetch raw quotes from all providers")

    def _get_adj_factors(self) -> pd.DataFrame:
        cache_path = os.path.join(DATA_DIR, "adj_factors.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[DataAgent] adj factors cache hit rows={len(cached)}")
            return cached

        self._require_pro_api("_get_adj_factors")
        trade_dates = self._get_trade_dates_from_daily()
        print(f"[DataAgent] fetching adj factors trade_dates={len(trade_dates)}")

        all_data = []
        failed_dates = []
        for i, date in enumerate(trade_dates):
            try:
                df = self.pro.adj_factor(trade_date=date, fields="ts_code,trade_date,adj_factor")
                time.sleep(API_DELAY)
                if df is not None and not df.empty:
                    df = df[~df["ts_code"].str.endswith(".BJ")]
                    all_data.append(df)
                if (i + 1) % 100 == 0:
                    print(f"  processed {i + 1}/{len(trade_dates)}")
            except Exception as e:
                failed_dates.append(date)
                print(f"  fetch failed date={date}: {e}")
                time.sleep(2)

        for retry in range(2):
            if not failed_dates:
                break
            still_failed = []
            print(f"  retry {retry + 1}, pending={len(failed_dates)}")
            for date in failed_dates:
                try:
                    time.sleep(2)
                    df = self.pro.adj_factor(trade_date=date, fields="ts_code,trade_date,adj_factor")
                    time.sleep(API_DELAY)
                    if df is not None and not df.empty:
                        df = df[~df["ts_code"].str.endswith(".BJ")]
                        all_data.append(df)
                    else:
                        still_failed.append(date)
                except Exception:
                    still_failed.append(date)
            failed_dates = still_failed

        if not all_data:
            raise RuntimeError("[DataAgent] adj factors returned no data")
        if failed_dates:
            print(f"[DataAgent] warning: unresolved adj-factor dates={len(failed_dates)}")

        result = pd.concat(all_data, ignore_index=True)
        result = result.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        result["adj_factor"] = pd.to_numeric(result["adj_factor"], errors="coerce")
        save_parquet(result, cache_path)
        print(f"[DataAgent] adj factors fetched rows={len(result)}")
        return result

    def _apply_price_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.price_adj_mode == "none":
            print("[DataAgent] PRICE_ADJ_MODE=none, skip price adjustment")
            return df

        adj_df = self._get_adj_factors()
        work = df.merge(adj_df, on=["ts_code", "trade_date"], how="left")
        work["adj_factor"] = pd.to_numeric(work["adj_factor"], errors="coerce")
        work["adj_factor"] = work.groupby("ts_code")["adj_factor"].transform(lambda x: x.ffill().bfill())

        if self.price_adj_mode == "qfq":
            anchor = work.groupby("ts_code")["adj_factor"].transform("last")
        else:
            anchor = work.groupby("ts_code")["adj_factor"].transform("first")

        scale = work["adj_factor"] / anchor
        scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        for col in ["open", "high", "low", "close", "pre_close"]:
            work[col] = pd.to_numeric(work[col], errors="coerce") * scale

        work = work.drop(columns=["adj_factor"])
        print(f"[DataAgent] applied price adjustment mode={self.price_adj_mode}")
        return work

    def get_daily_quotes(self) -> pd.DataFrame:
        cache_path = os.path.join(DATA_DIR, "daily_quotes.parquet")
        meta_path = os.path.join(DATA_DIR, "daily_quotes.meta.json")
        cached = load_parquet(cache_path)

        if cached is not None:
            meta = self._load_cache_meta(meta_path)
            cached_mode = str(meta.get("price_adj_mode", "")).lower()
            cached_provider = str(meta.get("daily_quotes_provider", "")).lower()
            provider_match = (
                cached_provider == self.daily_quotes_provider
                or (self.daily_quotes_provider == "auto" and cached_provider in {"tushare", "akshare"})
            )
            if cached_mode == self.price_adj_mode and provider_match:
                print(
                    f"[DataAgent] daily quotes cache hit rows={len(cached)} "
                    f"mode={cached_mode}, provider={cached_provider or 'unknown'}"
                )
                return cached

            if self.pro is None and self.price_adj_mode != "none":
                print("[DataAgent] warning: cannot rebuild adjusted prices without TUSHARE_TOKEN, reuse cache")
                return cached

            print(
                f"[DataAgent] daily quotes cache mismatch "
                f"(mode {cached_mode}->{self.price_adj_mode}, "
                f"provider {cached_provider or 'unknown'}->{self.daily_quotes_provider}), rebuilding"
            )

        raw = self._fetch_raw_quotes()
        stock_info = self.get_stock_info()

        print("[DataAgent] cleaning daily quotes...")
        df = raw.copy()
        n_before = len(df)

        if len(stock_info) > 0 and "is_st" in stock_info.columns:
            st_codes = stock_info[stock_info["is_st"] == 1]["ts_code"].tolist()
            df = df[~df["ts_code"].isin(st_codes)]
            print(f"  ST filtered rows={n_before - len(df)} symbols={len(st_codes)}")

        if len(stock_info) > 0 and "list_date" in stock_info.columns:
            df = df.merge(stock_info[["ts_code", "list_date"]], on="ts_code", how="left")
            df["list_date"] = df["list_date"].fillna("").astype(str)

            has_list_date = df["list_date"].str.len().eq(8)
            listed_in_sample = has_list_date & (df["list_date"] >= START_DATE)
            after_list = has_list_date & (df["trade_date"] >= df["list_date"])

            df["_ipo_rank"] = np.nan
            df.loc[after_list, "_ipo_rank"] = df.loc[after_list].groupby("ts_code").cumcount() + 1
            df["is_ipo_period"] = (listed_in_sample & after_list & (df["_ipo_rank"] <= 5)).astype(int)

            df = df.drop(columns=["list_date", "_ipo_rank"])
            print(f"  IPO marked rows={int(df['is_ipo_period'].sum())}")
        else:
            df["is_ipo_period"] = 0

        if "pre_close" in df.columns and df["pre_close"].notna().sum() > 0:
            df["pct_chg"] = (df["close"] / df["pre_close"] - 1) * 100

            is_star = df["ts_code"].str.startswith("688")
            is_gem = df["ts_code"].str.startswith("300")
            gem_reform = df["trade_date"] >= "20200824"

            limit_pct = pd.Series(9.5, index=df.index)
            limit_pct[is_star] = 19.5
            limit_pct[is_gem & gem_reform] = 19.5

            df["is_limit_up"] = (df["pct_chg"] >= limit_pct).astype(int)
            df["is_limit_down"] = (df["pct_chg"] <= -limit_pct).astype(int)
            print(
                f"  limit marks: up={int(df['is_limit_up'].sum())}, "
                f"down={int(df['is_limit_down'].sum())}"
            )
        else:
            df["pct_chg"] = 0
            df["is_limit_up"] = 0
            df["is_limit_down"] = 0

        df = self._apply_price_adjustment(df)

        print(f"  cleaning done rows: {n_before} -> {len(df)}")
        save_parquet(df, cache_path)

        raw_meta = self._load_cache_meta(os.path.join(DATA_DIR, "daily_quotes_raw.meta.json"))
        raw_provider = str(raw_meta.get("provider", "")).lower()
        self._save_cache_meta(
            meta_path,
            {
                "price_adj_mode": self.price_adj_mode,
                "daily_quotes_provider": raw_provider or self.daily_quotes_provider,
                "start_date": START_DATE,
                "end_date": END_DATE,
            },
        )
        return df

    def get_benchmark(self) -> pd.DataFrame:
        cache_path = os.path.join(DATA_DIR, "benchmark.parquet")
        cached = load_parquet(cache_path)
        if cached is not None:
            print(f"[DataAgent] benchmark cache hit rows={len(cached)}")
            return cached

        print("[DataAgent] fetching benchmark from AKShare sh000852")
        df = ak.stock_zh_index_daily(symbol="sh000852")
        if df is None or df.empty:
            raise RuntimeError("[DataAgent] unable to fetch benchmark")

        df["trade_date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y%m%d")
        df = df.rename(columns={"close": "bench_close"})
        df = df[["trade_date", "bench_close"]]
        df = df[(df["trade_date"] >= START_DATE) & (df["trade_date"] <= END_DATE)]
        df = df.sort_values("trade_date").reset_index(drop=True)
        df["bench_close"] = pd.to_numeric(df["bench_close"], errors="coerce")
        save_parquet(df, cache_path)
        print(f"[DataAgent] benchmark fetched rows={len(df)}")
        return df

    def update_all(self) -> dict:
        print("=" * 50)
        print("[DataAgent] full update start")
        print("=" * 50)

        quotes = self.get_daily_quotes()
        benchmark = self.get_benchmark()

        summary = {
            "股票数量": int(quotes["ts_code"].nunique()),
            "行情数据行数": int(len(quotes)),
            "基准数据行数": int(len(benchmark)),
            "日期范围": f"{START_DATE} ~ {END_DATE}",
        }

        stock_info = load_parquet(os.path.join(DATA_DIR, "stock_info.parquet"))
        if stock_info is not None and "is_st" in stock_info.columns:
            summary["ST过滤"] = f"已完成（ST样本 {int(stock_info['is_st'].sum())} 只）"
        else:
            summary["ST过滤"] = "未提供（缺少stock_info缓存）"

        if "is_ipo_period" in quotes.columns:
            summary["IPO期标记"] = f"{int(quotes['is_ipo_period'].sum())} 行"
        if "is_limit_up" in quotes.columns:
            summary["涨停标记"] = f"{int(quotes['is_limit_up'].sum())} 行"

        print("\n[DataAgent] data update done:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        self.emit("data.ready", summary)
        return summary


if __name__ == "__main__":
    agent = DataAgent()
    agent.update_all()
