from typing import Dict, List, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def print_panel(title: str, text: str, style: str = "bold cyan"):
    console.print(Panel(text, title=title, title_align="left", style=style))

def print_table(title: str, columns: List[str], rows: List[List[Any]]):
    table = Table(title=title, box=box.SIMPLE_HEAVY)
    for c in columns:
        table.add_column(c, overflow="fold")
    for r in rows:
        table.add_row(*[str(x) if x is not None else "-" for x in r])
    console.print(table)

def info(msg: str):
    console.print(f"[bold cyan]INFO[/]: {msg}")

def warn(msg: str):
    console.print(f"[bold yellow]WARN[/]: {msg}")

def error(msg: str):
    console.print(f"[bold red]ERROR[/]: {msg}")
