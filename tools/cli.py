# cli.py
import click, requests

import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://api:8000")


@click.group()
def cli():
    pass


@cli.command()
@click.option('--start', help='起始时间 (ISO format, e.g. 2026-01-01T00:00:00)')
@click.option('--end', help='结束时间 (ISO format)')
@click.option('--source', type=click.Choice(['robot_state', 'ros_data']), help='数据来源（不指定则导出全部）')
def export_sensor_logs(start, end, source):
    """导出传感器日志 CSV（从 Cassandra）"""
    payload = {}
    if start:
        payload["start"] = start
    if end:
        payload["end"] = end
    if source:
        payload["source"] = source
    resp = requests.post(f"{API_URL}/export/sensor-logs", json=payload)
    if resp.status_code == 200:
        data = resp.json()
        filename = f"sensor_logs{'_' + source if source else ''}.csv"
        with open(filename, "w") as f:
            f.write(data["csv"])
        click.echo(f"传感器日志已导出，共 {data['count']} 条记录 -> {filename}")
    else:
        click.echo(f"导出失败: {resp.text}")


@cli.command()
@click.option('--model-version', help='模型版本')
def trigger_prelabel(model_version):
    """触发全量数据重新预标注"""
    requests.post(f"{API_URL}/prelabel/trigger", json={"model": model_version})
    click.echo("预标注流水线已启动")


if __name__ == '__main__':
    cli()
