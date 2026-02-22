"""
HTML 交互式报告生成器

使用 Plotly 生成嵌入图表的单文件 HTML 报告。
若 Plotly 不可用，生成纯 HTML 文本报告。
"""

import os
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import ensure_dir
from utils.log import get_logger

logger = get_logger("analytics.html_report")


class HTMLReportGenerator:
    """HTML 交互式报告生成器"""

    def generate(
        self,
        metrics: dict,
        nav_df: pd.DataFrame,
        factor_test: pd.DataFrame,
        output_path: str,
    ):
        """生成单文件 HTML 报告"""
        ensure_dir(os.path.dirname(output_path))

        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            html = self._generate_plotly(metrics, nav_df, factor_test)
        except ImportError:
            logger.info("Plotly 不可用，生成纯 HTML 报告")
            html = self._generate_fallback(metrics, nav_df, factor_test)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML 报告已保存: {os.path.basename(output_path)}")

    def _generate_plotly(self, metrics: dict, nav_df: pd.DataFrame, factor_test: pd.DataFrame) -> str:
        """使用 Plotly 生成交互式报告"""
        import plotly.graph_objects as go
        import plotly.offline as pyo

        # NAV 曲线
        dates = pd.to_datetime(nav_df["trade_date"], format="%Y%m%d")
        nav_fig = go.Figure()
        nav_fig.add_trace(go.Scatter(x=dates, y=nav_df["nav"], name="Strategy", line=dict(color="#e74c3c")))
        nav_fig.add_trace(go.Scatter(x=dates, y=nav_df["bench_nav"], name="Benchmark", line=dict(color="#3498db")))
        nav_fig.update_layout(title="Strategy NAV vs Benchmark", xaxis_title="Date", yaxis_title="NAV", height=400)
        nav_html = pyo.plot(nav_fig, output_type="div", include_plotlyjs=False)

        # 超额净值
        excess_fig = go.Figure()
        if "excess_nav" in nav_df.columns:
            excess_fig.add_trace(go.Scatter(x=dates, y=nav_df["excess_nav"], name="Excess NAV", line=dict(color="#2ecc71")))
            excess_fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        excess_fig.update_layout(title="Excess NAV", xaxis_title="Date", yaxis_title="Excess NAV", height=400)
        excess_html = pyo.plot(excess_fig, output_type="div", include_plotlyjs=False)

        # 回撤曲线
        nav_vals = nav_df["nav"].values
        peak = pd.Series(nav_vals).cummax()
        dd = (pd.Series(nav_vals) - peak) / peak
        dd_fig = go.Figure()
        dd_fig.add_trace(go.Scatter(x=dates, y=dd.values, fill="tozeroy", name="Drawdown", line=dict(color="#e74c3c")))
        dd_fig.update_layout(title="Drawdown", xaxis_title="Date", yaxis_title="Drawdown", height=300)
        dd_html = pyo.plot(dd_fig, output_type="div", include_plotlyjs=False)

        # 指标表格
        metrics_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())

        # 因子检验表格
        factor_rows = ""
        if not factor_test.empty:
            for _, row in factor_test.iterrows():
                cells = "".join(f"<td>{row[c]}</td>" for c in factor_test.columns)
                factor_rows += f"<tr>{cells}</tr>"
            factor_headers = "".join(f"<th>{c}</th>" for c in factor_test.columns)
        else:
            factor_headers = ""

        html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>Quant Strategy Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 20px; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .chart {{ margin: 20px 0; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Quant Multi-Factor Strategy Report</h1>
        <p class="timestamp">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {metrics_rows}
        </table>

        <div class="chart">
            <h2>NAV Curve</h2>
            {nav_html}
        </div>

        <div class="chart">
            <h2>Excess NAV</h2>
            {excess_html}
        </div>

        <div class="chart">
            <h2>Drawdown</h2>
            {dd_html}
        </div>

        <h2>Single Factor Test</h2>
        <table>
            <tr>{factor_headers}</tr>
            {factor_rows}
        </table>
    </div>
</body>
</html>"""
        return html

    def _generate_fallback(self, metrics: dict, nav_df: pd.DataFrame, factor_test: pd.DataFrame) -> str:
        """纯 HTML 报告（无 Plotly 时使用）"""
        metrics_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())

        factor_rows = ""
        factor_headers = ""
        if not factor_test.empty:
            factor_headers = "".join(f"<th>{c}</th>" for c in factor_test.columns)
            for _, row in factor_test.iterrows():
                cells = "".join(f"<td>{row[c]}</td>" for c in factor_test.columns)
                factor_rows += f"<tr>{cells}</tr>"

        return f"""<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="utf-8">
    <title>Quant Strategy Report</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #3498db; color: white; }}
    </style>
</head>
<body>
    <h1>Quant Multi-Factor Strategy Report</h1>
    <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><em>Install plotly for interactive charts: pip install plotly</em></p>

    <h2>Performance Metrics</h2>
    <table><tr><th>Metric</th><th>Value</th></tr>{metrics_rows}</table>

    <h2>Single Factor Test</h2>
    <table><tr>{factor_headers}</tr>{factor_rows}</table>
</body>
</html>"""
